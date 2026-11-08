// Shared helpers for the inference pages. All user-controlled text goes through
// textContent (never innerHTML).
const API = "/infer/api";

function getCookie(name) {
  const m = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
  return m ? decodeURIComponent(m[1]) : "";
}

function setToken(t) {
  document.cookie = "footy_token=" + encodeURIComponent(t) +
    "; path=/; SameSite=Strict; max-age=2592000";
}

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  const tok = getCookie("footy_token");
  if (tok) headers["Authorization"] = "Bearer " + tok;
  let res = await fetch(API + path, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    const t = prompt("This server needs an access token:");
    if (t) {
      setToken(t);
      return api(path, opts);
    }
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k);
  return n;
}

function fmtSecs(s) {
  if (s == null) return "-";
  s = Math.round(s);
  return s >= 60 ? Math.floor(s / 60) + "m " + (s % 60) + "s" : s + "s";
}

function fmtDate(t) {
  return t ? new Date(t * 1000).toLocaleString() : "-";
}

// Short plain-language line for where a job is now.
function stageLine(job) {
  const seg = job.stages.segment, trk = job.stages.track;
  if (job.status === "queued") return "Waiting for the GPU (another job may be running)";
  if (job.status === "cancelled") return "Cancelled";
  if (job.status === "failed") return "Failed";
  if (seg.status === "running") return "Cutting the video into shots";
  if (trk.status === "running")
    return "Tracking clip " + Math.min(trk.clips_done + 1, trk.clips_total) +
      " of " + trk.clips_total;
  if (job.status === "done") return "Done: " + trk.clips_total + " clips";
  return job.status;
}

function renderStages(box, job, uploadPct) {
  box.replaceChildren();
  const seg = job.stages.segment, trk = job.stages.track;
  const cls = (s) => (s === "done" ? "done" : s === "running" ? "active" : s === "failed" ? "failed" : "");
  box.append(
    el("span", { class: "stage " + (uploadPct == null || uploadPct >= 100 ? "done" : "active") }, "Uploading"),
    el("span", { class: "stage " + cls(seg.status) }, "Cutting"),
    el("span", { class: "stage " + cls(trk.status) },
      trk.clips_total ? "Tracking " + trk.clips_done + "/" + trk.clips_total : "Tracking"),
  );
}

const NAV = [["/infer", "Upload"], ["/infer/jobs", "Jobs"]];
function renderHeader(title) {
  const h = document.querySelector("header");
  h.append(el("h1", {}, title));
  for (const [href, label] of NAV) {
    h.append(el("a", { href, class: location.pathname === href ? "cur" : "" }, label));
  }
}
