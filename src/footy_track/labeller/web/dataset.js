// Carry ?dataset=<name> through every same-origin request a labeller page makes.
//
// One server serves several datasets (see labeller/datasets.py); the dataset is
// chosen per request by the `dataset` query parameter, default = the registry
// default. Pages are opened as /ball_check?dataset=norwich; this shim (loaded
// first in <head>) adds the same parameter to fetch(), WebSocket, <img>/<video>
// /<source> src, <a> href, setAttribute('src'|'href') and history.*State, so the
// page code itself does not have to know. With no ?dataset= it does nothing.
// Use window.footyUrl(u) for URLs built into innerHTML strings.
(function () {
  var ds = "";
  try { ds = new URLSearchParams(location.search).get("dataset") || ""; } catch (e) {}
  window.FOOTY_DATASET = ds;
  window.footyUrl = function (u) { return u; };
  if (!ds) return;

  function tag(u) {
    try {
      if (typeof u !== "string" && !(u instanceof URL)) return u;
      var s = String(u);
      if (s === "" || s[0] === "#" || /^(data|blob|javascript|mailto):/i.test(s)) return u;
      var x = new URL(s, location.href);
      if (x.host !== location.host || x.pathname.indexOf("/static/") === 0 || x.searchParams.has("dataset")) return u;
      x.searchParams.set("dataset", ds);
      return s[0] === "/" && s[1] !== "/" ? x.pathname + x.search + x.hash : x.href;
    } catch (e) { return u; }
  }
  window.footyUrl = tag;

  var _fetch = window.fetch;
  window.fetch = function (input, init) {
    return _fetch.call(this, typeof input === "string" ? tag(input) : input, init);
  };

  var _WS = window.WebSocket;
  function WS(u, p) { return p === undefined ? new _WS(tag(u)) : new _WS(tag(u), p); }
  WS.prototype = _WS.prototype;
  ["CONNECTING", "OPEN", "CLOSING", "CLOSED"].forEach(function (k) { WS[k] = _WS[k]; });
  window.WebSocket = WS;

  function patchSetter(proto, prop) {
    var d = Object.getOwnPropertyDescriptor(proto, prop);
    if (!d || !d.set) return;
    Object.defineProperty(proto, prop, {
      configurable: true, enumerable: d.enumerable, get: d.get,
      set: function (v) { d.set.call(this, tag(v)); },
    });
  }
  patchSetter(HTMLImageElement.prototype, "src");
  patchSetter(HTMLMediaElement.prototype, "src");
  patchSetter(HTMLSourceElement.prototype, "src");
  patchSetter(HTMLAnchorElement.prototype, "href");

  var _set = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function (name, value) {
    var n = String(name).toLowerCase();
    return _set.call(this, name, n === "src" || n === "href" ? tag(value) : value);
  };

  ["replaceState", "pushState"].forEach(function (k) {
    var orig = history[k];
    history[k] = function (state, title, url) {
      return orig.call(this, state, title, url == null ? url : tag(url));
    };
  });

  // Static markup: links and images already in the page when it loads.
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll('a[href^="/"], img[src^="/"]').forEach(function (e) {
      var attr = e.tagName === "A" ? "href" : "src";
      _set.call(e, attr, tag(e.getAttribute(attr)));
    });
  });
})();
