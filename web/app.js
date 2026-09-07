/* Qwen3-TTS ONNX — 管理界面逻辑（原生 JS，无构建） */
"use strict";

const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const TOKEN_KEY = "q3tts_token";

/* ── 鉴权 ─────────────────────────────────────────────── */
function getToken() { return localStorage.getItem(TOKEN_KEY) || ""; }
function setToken(t) { localStorage.setItem(TOKEN_KEY, t); }

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const t = getToken();
  if (t) headers["Authorization"] = "Bearer " + t;
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401 && !path.startsWith("/healthz")) {
    showLogin();
    throw new Error("unauthorized");
  }
  return res;
}

async function apiJSON(path, opts = {}) {
  const res = await api(path, opts);
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-json */ }
  if (!res.ok) {
    const msg = (body && body.error && body.error.message) || res.status;
    throw new Error(msg);
  }
  return body;
}

/* ── 登录 ─────────────────────────────────────────────── */
function showLogin() { $("#app").classList.add("hidden"); $("#login-overlay").classList.remove("hidden"); }
function enter() {
  const t = $("#login-token").value.trim();
  if (!t) return;
  setToken(t);
  $("#login-overlay").classList.add("hidden");
  $("#app").classList.remove("hidden");
  refreshAll();
}
function checkAuth() {
  if (!getToken()) { showLogin(); return; }
  // 探一下
  api("/v1/models").then(() => {
    $("#app").classList.remove("hidden");
    $("#login-overlay").classList.add("hidden");
  }).catch(() => showLogin());
}

/* ── 导航 ─────────────────────────────────────────────── */
function switchTab(name) {
  $$(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach(t => t.classList.toggle("active", t.id === name));
  if (name === "tabs/voices") loadVoices();
  if (name === "tabs/tasks") loadTasks();
  if (name === "tabs/models") loadModels();
}

/* ── 合成 ─────────────────────────────────────────────── */
async function loadVoicesInto(sel, preferred) {
  try {
    const r = await apiJSON("/v1/voices");
    const opts = r.data.map(v => `<option value="${esc(v.voice_id)}">${esc(v.voice_id)} · ${v.type} · ${v.language}</option>`).join("");
    sel.innerHTML = opts || "<option value=''>（暂无语音，请先注册）</option>";
    if (preferred && $$("option", sel).some(o => o.value === preferred)) sel.value = preferred;
  } catch (e) { sel.innerHTML = "<option>加载失败</option>"; }
}

async function synth(sync) {
  const model = $("#syn-model").value;
  const text = $("#syn-text").value.trim();
  if (!text) { flash("请输入合成文本", "warn"); return; }
  const fmt = $("#syn-format").value;
  const language = $("#syn-language").value;
  const src = $("#syn-voice-src").value;
  const fd = new FormData();
  fd.append("text", text);
  fd.append("model", model);
  fd.append("language", language === "Auto" ? "Chinese" : language);
  fd.append("response_format", fmt);
  const voice = $("#syn-voice").value;
  if (src === "registered" && voice) fd.append("voice", voice);
  if (src === "temp") {
    const ref = $("#syn-ref-audio").files && $("#syn-ref-audio").files[0];
    if (ref) fd.append("ref_audio", ref);
    if ($("#syn-ref-text").value.trim()) fd.append("ref_text", $("#syn-ref-text").value.trim());
    if ($("#syn-instruct").value.trim()) fd.append("instruct", $("#syn-instruct").value.trim());
  }
  setStatus(sync ? "合成中…（CPU 较慢，短文本几秒～数十秒）" : "提交中…");
  try {
    const res = await api(sync ? "/v1/audio/speech" : "/api/tasks", {
      method: "POST", body: sync ? fd : JSON.stringify(Object.fromEntries(fd)),
      headers: sync ? {} : { "Content-Type": "application/json" }
    });
    if (sync) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      $("#syn-audio").src = url;
      $("#syn-download").href = url;
      $("#syn-download").setAttribute("download", `tts-${Date.now()}.${fmt}`);
      $("#syn-result").classList.remove("hidden");
      setStatus(`完成（${Math.round(blob.size / 1024)} KB, ${fmt}）`, "ok");
    } else {
      const j = await res.json();
      setStatus(`已提交任务 ${j.task_id.slice(0, 8)}…`, "ok");
      switchTab("tabs/tasks");
    }
  } catch (e) { setStatus("失败: " + e.message, "err"); }
}

/* ── 语音库 ───────────────────────────────────────────── */
let voicesCache = [];
let modalMode = "clone";

async function loadVoices() {
  try {
    const r = await apiJSON("/v1/voices");
    voicesCache = r.data;
    const tb = $("#voices-table tbody");
    tb.innerHTML = r.data.map(v => {
      const info = v.type === "design" ? esc((v.instruct || "").slice(0, 40)) : esc((v.ref_text || "").slice(0, 60));
      return `<tr>
        <td>${esc(v.voice_id)}</td>
        <td><span class="badge ${v.type}">${v.type}</span></td>
        <td>${esc(v.language)}</td>
        <td class="meta">${info}</td>
        <td class="meta">${esc(v.updated_at)}</td>
        <td>
          <button class="small" data-test="${v.voice_id}">试听</button>
          <button class="small" data-del="${v.voice_id}">删除</button>
        </td></tr>`;
    }).join("") || "<tr><td colspan=6 class=meta>（空）</td></tr>";
  } catch (e) { flash("加载语音库失败: " + e.message); }
}

function openModal(mode) {
  modalMode = mode;
  $("#modal-title").textContent = mode === "clone" ? "注册克隆语音" : "注册设计语音";
  $("#modal-clone").classList.toggle("hidden", mode !== "clone");
  $("#modal-design").classList.toggle("hidden", mode !== "design");
  $("#modal-err").classList.add("hidden");
  $("#voices-modal").classList.remove("hidden");
}

async function saveVoice() {
  const vid = $("#modal-vid").value.trim();
  const lang = $("#modal-language").value;
  const desc = $("#modal-desc").value.trim();
  if (!vid) { $("#modal-err").textContent = "voice_id 必填"; $("#modal-err").classList.remove("hidden"); return; }
  try {
    if (modalMode === "clone") {
      const f = $("#modal-ref-audio").files && $("#modal-ref-audio").files[0];
      const rt = $("#modal-ref-text").value.trim();
      if (!f) throw new Error("参考音频必填（wav/mp3）");
      if (!rt) throw new Error("参考文本必填");
      const fd = new FormData();
      fd.append("type", "clone"); fd.append("voice_id", vid); fd.append("language", lang);
      fd.append("description", desc); fd.append("ref_text", rt); fd.append("ref_audio", f);
      await apiJSON("/v1/voices", { method: "POST", body: fd });
    } else {
      const ins = $("#modal-instruct").value.trim();
      if (!ins) throw new Error("设计指令必填");
      await apiJSON("/v1/voices", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "design", voice_id: vid, language: lang, instruct: ins, description: desc }) });
    }
    closeModal(); loadVoices();
  } catch (e) { $("#modal-err").textContent = "保存失败: " + e.message; $("#modal-err").classList.remove("hidden"); }
}
function closeModal() { $("#voices-modal").classList.add("hidden"); }

async function testVoice(id) {
  try {
    const res = await api("/v1/voices/" + encodeURIComponent(id) + "/test", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: "{}" });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    $("#syn-audio").src = url; $("#syn-audio").controls = true;
    $("#syn-result").classList.remove("hidden");
    $("#syn-meta").textContent = "试听: " + id;
  } catch (e) { flash("试听失败: " + e.message); }
}

/* ── 任务 ─────────────────────────────────────────────── */
async function loadTasks() {
  const st = $("#tasks-filter").value;
  const path = "/api/tasks" + (st ? "?status=" + st : "");
  try {
    const r = await apiJSON(path);
    const tb = $("#tasks-table tbody");
    tb.innerHTML = r.data.map(t => {
      let prog = "";
      if (t.progress_frame != null && t.max_frames) {
        prog = `<progress value="${t.progress_frame}" max="${t.max_frames}"></progress>`;
      }
      const stat = t.status === "failed" ? `<span class="badge failed" title="${esc(t.error_message || "")}">failed</span>`
        : `<span class="badge ${t.status}">${t.status}</span>`;
      let ops = "";
      if (t.status === "completed") ops += `<button class="small" data-task-audio="${t.task_id}">下载</button>`;
      if (t.status === "running") ops += `<button class="small" data-task-cancel="${t.task_id}">取消</button>`;
      return `<tr><td class="meta">${esc(t.task_id.slice(0, 8))}</td><td class="meta">${esc(t.model_id)}</td>
        <td>${stat}</td><td>${prog}</td>
        <td class="meta">${esc((t.input_text || "").slice(0, 30))}</td><td>${ops}</td></tr>`;
    }).join("") || "<tr><td colspan=6 class=meta>（空）</td></tr>";
  } catch (e) { /* 已处理 */ }
}

/* ── 模型/服务 ─────────────────────────────────────────── */
async function loadModels() {
  try {
    const r = await apiJSON("/api/models/status");
    const cards = r.models.map(m => `
      <div class="card" style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <b>${esc(m.id)}</b> <span class="badge ${m.status}">${m.status}</span>
          <span class="badge ${m.capability}">${m.capability}</span><br>
          <span class="meta">RAM≈${m.estimated_ram_mb || "-"} MB · 加载 ${(m.load_time_ms || 0).toFixed(0)}ms
            · 推理 ${m.infer_count} 次 · 空闲 ${m.idle_seconds == null ? "-" : m.idle_seconds + "s"}</span>
        </div>
        <div>
          ${m.status === "loaded" ? `<button data-unload="${m.id}">卸载</button>` : ""}
          ${m.status !== "loaded" && m.status !== "loading" ? `<button class="primary" data-load="${m.id}">加载</button>` : ""}
        </div>
      </div>`).join("");
    $("#models-cards").innerHTML = cards;
  } catch (e) { /* ignore */ }
  try {
    const r = await apiJSON("/api/models/status");
    $("#server-info").textContent = JSON.stringify({ models_status: r.models.length, uptime_hint: "见 /healthz" }, null, 2);
  } catch (e) { /* ignore */ }
  try {
    const c = await apiJSON("/api/config");
    const redacted = { ...c, models: c.models.map(m => ({ id: m.id, capability: m.capability,
      load_on_start: m.load_on_start, max_new_tokens: m.max_new_tokens })) };
    $("#cfg-view").textContent = JSON.stringify(redacted, null, 2);
  } catch (e) { $("#cfg-view").textContent = "配置加载失败"; }
}

/* ── utils ─────────────────────────────────────────────── */
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
  .replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
function flash(msg, kind = "") { const el = $("#syn-status"); el.textContent = msg; el.className = "status " + kind; }
function setStatus(msg, kind = "") { flash(msg, kind); }
function refreshAll() { loadModels(); loadVoices(); loadTasks(); loadVoicesInto($("#syn-voice")); }

/* ── 事件绑定 ─────────────────────────────────────────── */
document.addEventListener("click", e => {
  const t = e.target;
  if (t.dataset && t.dataset.tab) switchTab(t.dataset.tab);
  if (t.dataset && t.dataset.test) testVoice(t.dataset.test);
  if (t.dataset && t.dataset.del) { if (confirm("删除语音 " + t.dataset.del + " 及其参考音频？")) { fetch(`/v1/voices/${encodeURIComponent(t.dataset.del)}`, { method: "DELETE", headers: { Authorization: "Bearer " + getToken() } }).then(() => { loadVoices(); loadVoicesInto($("#syn-voice")); }); } }
  if (t.dataset && t.dataset.load) { fetch(`/api/models/${t.dataset.load}/load`, { method: "POST", headers: { Authorization: "Bearer " + getToken() } }).then(() => loadModels()); }
  if (t.dataset && t.dataset.unload) { fetch(`/api/models/${t.dataset.unload}/unload`, { method: "POST", headers: { Authorization: "Bearer " + getToken() } }).then(() => loadModels()); }
  if (t.dataset && t.dataset.taskAudio) { window.open(`/api/tasks/${t.dataset.taskAudio}/audio`); }
  if (t.dataset && t.dataset.taskCancel) { fetch(`/api/tasks/${t.dataset.taskCancel}`, { method: "DELETE", headers: { Authorization: "Bearer " + getToken() } }).then(() => loadTasks()); }
  if (t.id === "login-btn") enter();
  if (t.id === "logout") { localStorage.removeItem(TOKEN_KEY); location.reload(); }
  if (t.id === "syn-go") synth(true);
  if (t.id === "syn-async") synth(false);
  if (t.id === "syn-listen") { const v = $("#syn-voice").value; if (v) testVoice(v); }
  if (t.id === "voices-refresh") loadVoices();
  if (t.id === "voices-add-clone") openModal("clone");
  if (t.id === "voices-add-design") openModal("design");
  if (t.id === "modal-save") saveVoice();
  if (t.id === "modal-cancel") closeModal();
  if (t.id === "tasks-refresh") loadTasks();
});

document.addEventListener("change", e => {
  const el = e.target;
  if (el.id === "syn-voice-src") {
    const temp = el.value === "temp";
    $("#syn-registered").classList.toggle("hidden", temp);
    $("#syn-temp").classList.toggle("hidden", !temp);
  }
  if (el.id === "syn-model") loadVoicesInto($("#syn-voice"));
  if (el.id === "tasks-filter") loadTasks();
});

/* ── 启动 ─────────────────────────────────────────────── */
$(document).readyState;
checkAuth();
setInterval(() => { if (!document.hidden && $("#tabs/tasks").classList.contains("active")) loadTasks(); }, 3000);
window.URL = window.URL || window.URL ?? {}; // no-op guard
window.addEventListener("beforeunload", () => { /* keep token */ });