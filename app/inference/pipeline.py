"""Qwen3-TTS ONNX 推理引擎。

移植自 onnx-community/Qwen3-TTS 仓库的自带 inference.py：
- manifest 驱动加载 9/8 个子模型（text_embed / codec_embed / talker / talker_cache /
  code_predictor / residual_embed / tok_encoder / tok_decoder / speaker_encoder）
- numpy 自回归循环（镜像 Qwen3TTSForConditionalGeneration.generate，non_streaming）
- 语音克隆（base）+ 声音设计（voice_design）能力门控

运行时依赖：纯 onnxruntime（>=1.20），CPUExecutionProvider。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

SR = 24000
DEC_FRAMES = 25            # tok_decoder 固定 25 帧
N_GROUPS = 16              # 16 个 codebook


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── numpy 采样工具（镜像 HF generate 逻辑） ───────────────────────────────
def _apply_repetition_penalty(logits, prev_ids, penalty):
    if penalty == 1.0 or not prev_ids:
        return logits
    idx = np.array(sorted(set(int(i) for i in prev_ids)), dtype=np.int64)
    sc = logits[idx]
    logits[idx] = np.where(sc < 0, sc * penalty, sc / penalty)
    return logits


def _sample(logits, do_sample, top_k, top_p, temperature, rng):
    logits = logits.astype(np.float64)
    if not do_sample or temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = np.partition(logits, -k)[-k]
        logits = np.where(logits < kth, -np.inf, logits)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    if top_p and top_p < 1.0:
        order = np.argsort(probs)[::-1]
        csum = np.cumsum(probs[order])
        cut = np.searchsorted(csum, top_p) + 1
        keep = order[:cut]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        probs = mask / mask.sum()
    return int(rng.choice(len(probs), p=probs))


class Qwen3TtsOnnxPipeline:
    """manifest 驱动加载的 Qwen3-TTS 子模型集合 + 生成逻辑。"""

    def __init__(self, model_path: str, tts_dir: str | None = None):
        import onnxruntime as ort

        self.root = Path(model_path)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        sm = self.manifest["sub_models"]
        prov = self.manifest.get("execution_provider", "CPUExecutionProvider")
        avail = ort.get_available_providers()
        if prov not in avail:
            print(f"  [warn] manifest EP {prov} unavailable; falling back to CPU", file=sys.stderr)
            prov = "CPUExecutionProvider"
        self.provider = prov

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 0   # ORT 默认; 由主机调度

        def sess(name: str):
            if name not in sm:
                return None
            return ort.InferenceSession(str(self.root / sm[name]["filename"]), so, providers=[prov])

        self.text_embed = sess("text_embed")
        self.codec_embed = sess("codec_embed")
        self.talker = sess("talker")
        self.code_predictor = sess("code_predictor")
        self.residual_embed = sess("residual_embed")
        self.tok_encoder = sess("tok_encoder")
        self.tok_decoder = sess("tok_decoder")
        self.speaker_encoder = sess("speaker_encoder")    # 仅 base（克隆 x-vector）
        self.talker_cache = sess("talker_cache")          # 可选 O(n) KV-cache
        if self.talker_cache is not None:
            self._past_names = [i.name for i in self.talker_cache.get_inputs()][3:]

        # config + tokenizer（完整生成时才需要）
        self.tts_dir = tts_dir
        self._cfg = None
        self._tok = None
        if tts_dir:
            cfg_p = Path(tts_dir) / "config.json"
            if not cfg_p.exists():
                raise FileNotFoundError(f"config.json 不存在于 tts_dir: {tts_dir}")
            self._cfg = json.loads(cfg_p.read_text(encoding="utf-8"))

    # ── building blocks ──────────────────────────────────────────────────
    def embed_text(self, text_ids):
        return self.text_embed.run(None, {"text_ids": np.asarray(text_ids, np.int64)})[0]

    def embed_codec(self, codec_ids):
        return self.codec_embed.run(None, {"codec_ids": np.asarray(codec_ids, np.int64)})[0]

    def talker_step(self, inputs_embeds, position_ids, attention_mask):
        return self.talker.run(None, {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "position_ids": np.asarray(position_ids, np.int64),
            "attention_mask": np.asarray(attention_mask, np.int64)})

    def talker_cache_step(self, inputs_embeds, position_ids, attention_mask, past):
        feed = {"inputs_embeds": inputs_embeds.astype(np.float32),
                "position_ids": np.asarray(position_ids, np.int64),
                "attention_mask": np.asarray(attention_mask, np.int64)}
        for name, t in zip(self._past_names, past):
            feed[name] = t.astype(np.float32)
        out = self.talker_cache.run(None, feed)
        return out[0], out[1], list(out[2:])

    def predict_residual(self, talker_hidden, codec_ids):
        return self.code_predictor.run(None, {
            "talker_hidden": talker_hidden.astype(np.float32),
            "codec_ids": np.asarray(codec_ids, np.int64)})[0]

    def step_embed(self, codec_ids):
        return self.residual_embed.run(None, {"codec_ids": np.asarray(codec_ids, np.int64)})[0]

    # ── 音频编解码 ────────────────────────────────────────────────────────
    @staticmethod
    def _load_ref_wav(path):
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)

    def encode(self, audio):
        return self.tok_encoder.run(None, {"audio": audio.astype(np.float32)})[0]

    def encode_chunked(self, wav):
        out = []
        for s in range(0, max(len(wav), 1), SR):
            c = wav[s:s + SR]
            if len(c) < SR:
                c = np.pad(c, (0, SR - len(c)))
            out.append(self.encode(c.reshape(1, 1, SR).astype(np.float32))[0])
        return np.concatenate(out, axis=0).astype(np.int64)

    def decode(self, codes):
        return self.tok_decoder.run(None, {"audio_codes": np.asarray(codes, np.int64)})[0]

    def decode_chunked(self, codes):
        F = codes.shape[1]
        outs = []
        for s in range(0, F, DEC_FRAMES):
            chunk = codes[:, s:s + DEC_FRAMES]
            if chunk.shape[1] < DEC_FRAMES:
                idx = np.arange(DEC_FRAMES) % chunk.shape[1]
                chunk = chunk[:, idx]
                wav = self.decode(chunk)
                keep = int(round(wav.shape[-1] * (F - s) / DEC_FRAMES))
                outs.append(wav[..., :keep])
                break
            outs.append(self.decode(chunk))
        return np.concatenate(outs, axis=-1)

    # ── config / tokenizer 访问 ─────────────────────────────────────────
    @property
    def cfg(self):
        if self._cfg is None:
            raise RuntimeError("需要 tts_dir（原版 HF model dir）：config token ids + tokenizer 在那里")
        return self._cfg

    @property
    def model_type(self) -> str:
        c = self.cfg
        return (c.get("tts_model_type")
                or c.get("talker_config", {}).get("tts_model_type") or "unknown")

    def capability(self) -> str:
        mt = self.model_type
        if mt == "base":
            return "voice_clone"
        if mt == "voice_design":
            return "voice_design"
        if mt == "custom_voice":
            return "custom_voice"
        return "unknown"

    def _check_features(self, instruct=None, speaker=None, ref_audio=None, ref_text=None):
        mt = self.model_type
        if speaker and mt != "custom_voice":
            raise ValueError(f"--speaker 是 CustomVoice 功能，但本模型是 '{mt}'。请使用 customvoice checkpoint。")
        if instruct and mt not in ("voice_design", "custom_voice"):
            raise ValueError(f"--instruct 是 VoiceDesign/CustomVoice 功能，但本模型是 '{mt}'。"
                             f"Base 模型请用 --ref-audio 克隆。")
        if ref_audio and mt != "base":
            raise ValueError(f"语音克隆(--ref-audio) 是 Base 模型功能，但本模型是 '{mt}'。请用 base checkpoint。")
        if ref_text and not ref_audio:
            raise ValueError("ref_text 需要 ref_audio 一起提供（克隆）。")

    @property
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.tts_dir, trust_remote_code=True)
        return self._tok

    def _ids(self, text):
        enc = self.tokenizer(text, return_tensors="np")
        ids = enc["input_ids"]
        return ids if ids.ndim == 2 else ids[None]

    # ── generate 主入口 ─────────────────────────────────────────────────
    def generate(self, text, language="Auto", instruct=None, speaker=None,
                 ref_audio=None, ref_text=None,
                 max_new_tokens=2048, do_sample=True, top_k=50, top_p=1.0,
                 temperature=0.9, repetition_penalty=1.05,
                 sub_do_sample=True, sub_top_k=50, sub_top_p=1.0, sub_temperature=0.9,
                 seed=0, verbose=True) -> np.ndarray:
        """返回 codes [T,16]（int64）。用 decode_chunked(codes[None]) 解码为波形。"""
        cfg = self.cfg
        self._check_features(instruct=instruct, speaker=speaker,
                             ref_audio=ref_audio, ref_text=ref_text)
        if ref_audio:
            return self._generate_clone(text, ref_audio, ref_text, language=language,
                                        max_new_tokens=max_new_tokens, do_sample=do_sample,
                                        top_k=top_k, top_p=top_p, temperature=temperature,
                                        repetition_penalty=repetition_penalty,
                                        sub_do_sample=sub_do_sample, sub_top_k=sub_top_k,
                                        sub_top_p=sub_top_p, sub_temperature=sub_temperature,
                                        seed=seed, verbose=verbose)
        tc = cfg["talker_config"]
        H = tc["hidden_size"]
        rng = np.random.default_rng(seed)

        tts_bos, tts_eos, tts_pad = (cfg["tts_bos_token_id"], cfg["tts_eos_token_id"],
                                     cfg["tts_pad_token_id"])
        codec_eos = tc["codec_eos_token_id"]
        codec_pad, codec_bos = tc["codec_pad_id"], tc["codec_bos_id"]
        vocab = tc["vocab_size"]

        assistant = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_id = self._ids(assistant)
        if input_id.shape[1] < 9:
            raise ValueError("text tokenized too short for the assistant template")

        spec = self.embed_text([[tts_bos, tts_eos, tts_pad]])
        bos_e, eos_e, pad_e = spec[:, 0:1], spec[:, 1:2], spec[:, 2:3]

        lang = (language or "auto").lower()
        if lang == "auto" or lang not in tc.get("codec_language_id", {}):
            language_id = None
        else:
            language_id = tc["codec_language_id"][lang]
        if language_id is None:
            codec_prefill = [[tc["codec_nothink_id"], tc["codec_think_bos_id"],
                              tc["codec_think_eos_id"]]]
        else:
            codec_prefill = [[tc["codec_think_id"], tc["codec_think_bos_id"],
                              language_id, tc["codec_think_eos_id"]]]
        codec0 = self.embed_codec(codec_prefill)
        codec1 = self.embed_codec([[codec_pad, codec_bos]])

        speaker_embed = None
        if speaker:
            spk_map = tc.get("spk_id", {})
            if speaker.lower() not in spk_map:
                raise ValueError(f"Speaker '{speaker}' 不在 spk_id {list(spk_map)[:8]}…")
            speaker_embed = self.embed_codec([[spk_map[speaker.lower()]]])

        if speaker_embed is None:
            codec_input = np.concatenate([codec0, codec1], axis=1)
        else:
            codec_input = np.concatenate([codec0, speaker_embed, codec1], axis=1)

        prefix = []
        if instruct:
            instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
            prefix.append(self.embed_text(self._ids(instruct_text)))

        role = self.embed_text(input_id[:, :3])
        pad_block = np.concatenate(
            [np.repeat(pad_e, codec_input.shape[1] - 2, axis=1), bos_e], axis=1)
        talker_in = np.concatenate([role, pad_block + codec_input[:, :-1]], axis=1)

        body_ids = input_id[:, 3:-5]
        Ltext = body_ids.shape[1]
        text_body = self.embed_text(body_ids)
        block1 = (np.concatenate([text_body, eos_e], axis=1)
                  + self.embed_codec([[codec_pad] * (Ltext + 1)]))
        block2 = pad_e + self.embed_codec([[codec_bos]])
        talker_in = np.concatenate([talker_in, block1, block2], axis=1)
        if prefix:
            talker_in = np.concatenate(prefix + [talker_in], axis=1)
        trailing = pad_e[:, 0]

        return self._ar_loop(talker_in, trailing, vocab, codec_eos, max_new_tokens,
                             do_sample, top_k, top_p, temperature, repetition_penalty,
                             sub_do_sample, sub_top_k, sub_top_p, sub_temperature, seed, verbose)

    def _ar_loop(self, talker_in, trailing, vocab, codec_eos, max_new_tokens, do_sample,
                 top_k, top_p, temperature, repetition_penalty, sub_do_sample, sub_top_k,
                 sub_top_p, sub_temperature, seed, verbose):
        if self.talker_cache is not None:
            return self._ar_loop_cached(talker_in, trailing, vocab, codec_eos, max_new_tokens,
                                        do_sample, top_k, top_p, temperature, repetition_penalty,
                                        sub_do_sample, sub_top_k, sub_top_p, sub_temperature,
                                        seed, verbose)
        rng = np.random.default_rng(seed)
        suppress = np.array([i for i in range(vocab - 1024, vocab) if i != codec_eos],
                            dtype=np.int64)
        all_codes, prev_first = [], []
        for step in range(max_new_tokens):
            T = talker_in.shape[1]
            pos = np.broadcast_to(np.arange(T), (3, 1, T)).copy()
            mask = np.ones((1, T), dtype=np.int64)
            logits, hidden = self.talker_step(talker_in, pos, mask)
            first = logits[0, -1].astype(np.float64).copy()
            first[suppress] = -np.inf
            first = _apply_repetition_penalty(first, prev_first, repetition_penalty)
            code0 = _sample(first, do_sample, top_k, top_p, temperature, rng)
            if code0 == codec_eos:
                break
            prev_first.append(code0)
            th = hidden[0, -1][None].astype(np.float32)
            codes16 = np.zeros((1, N_GROUPS), dtype=np.int64)
            codes16[0, 0] = code0
            for j in range(1, N_GROUPS):
                gl = self.predict_residual(th, codes16)
                codes16[0, j] = _sample(gl[0, j - 1], sub_do_sample, sub_top_k, sub_top_p,
                                        sub_temperature, rng)
            all_codes.append(codes16[0].copy())
            nxt = self.step_embed(codes16)[:, None] + trailing[:, None]
            talker_in = np.concatenate([talker_in, nxt], axis=1)
            if verbose and (step + 1) % 25 == 0:
                print(f"    …{step + 1} frames", file=sys.stderr)
        codes = np.stack(all_codes, axis=0).astype(np.int64) if all_codes \
            else np.zeros((0, N_GROUPS), np.int64)
        if verbose:
            print(f"  generated {codes.shape[0]} frames", file=sys.stderr)
        return codes

    def _ar_loop_cached(self, talker_in, trailing, vocab, codec_eos, max_new_tokens, do_sample,
                        top_k, top_p, temperature, repetition_penalty, sub_do_sample, sub_top_k,
                        sub_top_p, sub_temperature, seed, verbose):
        rng = np.random.default_rng(seed)
        suppress = np.array([i for i in range(vocab - 1024, vocab) if i != codec_eos],
                            dtype=np.int64)
        past = [np.zeros((1, 8, 0, 128), np.float32) for _ in self._past_names]
        T0 = talker_in.shape[1]
        pos = np.broadcast_to(np.arange(T0), (3, 1, T0)).copy()
        logits, hidden, past = self.talker_cache_step(
            talker_in, pos, np.ones((1, T0), np.int64), past)
        total = T0
        all_codes, prev_first = [], []
        for step in range(max_new_tokens):
            first = logits[0, -1].astype(np.float64).copy()
            first[suppress] = -np.inf
            first = _apply_repetition_penalty(first, prev_first, repetition_penalty)
            code0 = _sample(first, do_sample, top_k, top_p, temperature, rng)
            if code0 == codec_eos:
                break
            prev_first.append(code0)
            th = hidden[0, -1][None].astype(np.float32)
            codes16 = np.zeros((1, N_GROUPS), dtype=np.int64)
            codes16[0, 0] = code0
            for j in range(1, N_GROUPS):
                gl = self.predict_residual(th, codes16)
                codes16[0, j] = _sample(gl[0, j - 1], sub_do_sample, sub_top_k, sub_top_p,
                                        sub_temperature, rng)
            all_codes.append(codes16[0].copy())
            nxt = self.step_embed(codes16)[:, None] + trailing[:, None]
            pos = np.broadcast_to(np.array([total]), (3, 1, 1)).copy()
            logits, hidden, past = self.talker_cache_step(
                nxt, pos, np.ones((1, total + 1), np.int64), past)
            total += 1
            if verbose and (step + 1) % 25 == 0:
                print(f"    …{step + 1} frames (cached)", file=sys.stderr)
        codes = np.stack(all_codes, axis=0).astype(np.int64) if all_codes \
            else np.zeros((0, N_GROUPS), np.int64)
        if verbose:
            print(f"  generated {codes.shape[0]} frames (KV-cache)", file=sys.stderr)
        return codes

    def _generate_clone(self, text, ref_audio, ref_text, language="Auto", max_new_tokens=2048,
                        do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
                        repetition_penalty=1.05, sub_do_sample=True, sub_top_k=50, sub_top_p=1.0,
                        sub_temperature=0.9, seed=0, verbose=True):
        if self.speaker_encoder is None:
            raise RuntimeError("speaker_encoder.onnx 缺失 —— 需要为 Base 模型导出该组件。")
        if not ref_text:
            raise ValueError("voice clone 需要 --ref-text（--ref-audio 的转写文本）。")
        cfg = self.cfg
        tc = cfg["talker_config"]
        H = tc["hidden_size"]
        tts_bos, tts_eos, tts_pad = (cfg["tts_bos_token_id"], cfg["tts_eos_token_id"],
                                     cfg["tts_pad_token_id"])
        codec_eos = tc["codec_eos_token_id"]
        codec_pad, codec_bos = tc["codec_pad_id"], tc["codec_bos_id"]
        vocab = tc["vocab_size"]

        wav = self._load_ref_wav(ref_audio)
        ref_code = self.encode_chunked(wav)
        spk = self.speaker_encoder.run(None, {"audio": wav[None].astype(np.float32)})[0]
        spk = spk.reshape(1, 1, H)

        assistant = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_id = self._ids(assistant)
        ref_id = self._ids(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n")[:, 3:-2]
        text_id = input_id[:, 3:-5]

        spec = self.embed_text([[tts_bos, tts_eos, tts_pad]])
        bos_e, eos_e, pad_e = spec[:, 0:1], spec[:, 1:2], spec[:, 2:3]

        lang = (language or "auto").lower()
        language_id = (tc["codec_language_id"][lang]
                       if lang != "auto" and lang in tc.get("codec_language_id", {}) else None)
        codec_prefill = ([[tc["codec_nothink_id"], tc["codec_think_bos_id"],
                           tc["codec_think_eos_id"]]]
                         if language_id is None else
                         [[tc["codec_think_id"], tc["codec_think_bos_id"], language_id,
                           tc["codec_think_eos_id"]]])
        codec0 = self.embed_codec(codec_prefill)
        codec1 = self.embed_codec([[codec_pad, codec_bos]])
        codec_input = np.concatenate([codec0, spk, codec1], axis=1)

        role = self.embed_text(input_id[:, :3])
        pad_block = np.concatenate(
            [np.repeat(pad_e, codec_input.shape[1] - 2, axis=1), bos_e], axis=1)
        base = np.concatenate([role, pad_block + codec_input[:, :-1]], axis=1)

        text_embed = np.concatenate([self.embed_text(np.concatenate([ref_id, text_id], axis=1)),
                                     eos_e], axis=1)
        T1 = text_embed.shape[1]
        codec_embed = np.concatenate([self.embed_codec([[codec_bos]]),
                                      self.step_embed(ref_code)[None]], axis=1)
        icl = text_embed + self.embed_codec([[codec_pad] * T1])
        icl = np.concatenate([icl, codec_embed + pad_e], axis=1)
        talker_in = np.concatenate([base, icl], axis=1)
        trailing = pad_e[:, 0]
        if verbose:
            print(f"  [clone] ref {ref_code.shape[0]} frames + ref_text {ref_id.shape[1]} toks "
                  f"+ text {text_id.shape[1]} toks -> prefill {talker_in.shape[1]}", file=sys.stderr)
        return self._ar_loop(talker_in, trailing, vocab, codec_eos, max_new_tokens, do_sample,
                             top_k, top_p, temperature, repetition_penalty, sub_do_sample,
                             sub_top_k, sub_top_p, sub_temperature, seed, verbose)

    # ── 能力门控（服务层调用，预校验）────────────────────────────────────
    def check_capability(self, capability: str):
        """服务期望能力 vs 模型实际能力的一致性校验。不匹配抛出 ValueError。"""
        mt = self.model_type
        actual = self.capability()
        if actual != capability:
            raise ValueError(f"模型 '{self.tts_dir}' 的 tts_model_type='{mt}'（能力 {actual}），"
                             f"与配置要求的能力 '{capability}' 不匹配。请检查 onnx_dir/tts_dir 配对了。")

    def selftest(self):
        """codec 往返 + 构建块冒烟测试（加载到目标 EP）。返回 dict 供日志。"""
        import numpy as np
        rng = np.random.default_rng(0)
        t = np.arange(SR) / SR
        audio = (0.6 * np.sin(2 * np.pi * (180 + 300 * t) * t)
                 + 0.01 * rng.standard_normal(SR)).astype(np.float32)[None, None, :]

        codes = self.encode(audio)
        wav = self.decode_chunked(codes)
        te = self.embed_codec(rng.integers(0, 2048, (1, 8)))
        ce = self.embed_codec(rng.integers(0, 2048, (1, 8)))
        th = rng.standard_normal((1, 2048)).astype(np.float32)
        gl = self.predict_residual(th, rng.integers(0, 2048, (1, N_GROUPS)))
        se = self.step_embed(rng.integers(0, 2048, (1, N_GROUPS)))
        return {
            "provider": self.provider,
            "model_type": self.model_type,
            "codec_roundtrip": (codes.shape, wav.shape),
            "text_embed": te.shape,
            "codec_embed": ce.shape,
            "code_predictor": gl.shape,
            "residual_embed": se.shape,
        }