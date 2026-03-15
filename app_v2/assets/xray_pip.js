/**
 * X-ray Activity Monitor — floating PiP panel.
 * Builds the panel UI inside #xray-pip, polls /xray/api/events,
 * supports drag-to-move, resize, minimize, filter, pause, clear.
 */
(function () {
  "use strict";

  // ── Source metadata: icon, human label, CSS class suffix ──
  var SOURCES = {
    ingest: { icon: "📥", label: "Plaud", cls: "ingest" },
    gemini: { icon: "🤖", label: "Gemini AI", cls: "gemini" },
    embed: { icon: "🧬", label: "Embedding", cls: "embed" },
    qdrant: { icon: "💎", label: "Qdrant", cls: "qdrant" },
    graph: { icon: "🕸", label: "Graph", cls: "graph" },
    search: { icon: "🔍", label: "Search", cls: "search" },
    data: { icon: "📊", label: "Data", cls: "data" },
    nav: { icon: "🧭", label: "Navigation", cls: "nav" },
    pipeline: { icon: "⚙️", label: "Pipeline", cls: "pipeline" },
    detail: { icon: "📋", label: "Detail", cls: "detail" },
    day: { icon: "📅", label: "Day View", cls: "day" },
    sync: { icon: "🔄", label: "Sync", cls: "sync" },
  };

  // ── Human-readable operation labels ──
  var OP_LABELS = {
    "plaud-api": "API",
    download: "Download",
    skip: "Skip",
    store: "Save",
    start: "Start",
    done: "Done",
    fail: "Fail",
    error: "Error",
    prompt: "Prompt",
    stream: "Stream",
    tokens: "Tokens",
    "json-repair": "Repair",
    extract: "Extract",
    retry: "Retry",
    text: "Text",
    batch: "Batch",
    multimodal: "Audio+Text",
    fallback: "Fallback",
    create: "Setup",
    upsert: "Save",
    search: "Search",
    build: "Build",
    communities: "Clusters",
    "cache-hit": "Cached",
    "cache-miss": "Loading",
    loaded: "Loaded",
    embed: "Embed",
    query: "Query",
    results: "Results",
    vector: "Search",
    ai: "AI",
    "ai-answer": "AI",
    layout: "Layout",
    "node-tap": "Click",
    view: "View",
    category: "Edit",
    select: "Open",
    "extract-error": "Error",
    switch: "Switch",
    topic: "Topic",
    detail: "Detail",
    data: "Data",
    total: "Done",
    render: "Render",
    fetch: "Fetch",
    openai: "AI",
    ingest: "Import",
    process: "Analyze",
    index: "Index",
  };

  // Filter groups — which source keys belong to each tab
  var FILTER_GROUPS = {
    all: null, // show everything
    pipeline: ["ingest", "gemini", "embed", "qdrant", "pipeline", "sync"],
    search: ["search"],
    graph: ["graph"],
    data: ["data", "day", "detail", "nav"],
    errors: null, // special: filter by level
  };

  // Wait for Dash to render #xray-pip
  function boot() {
    var pip = document.getElementById("xray-pip");
    if (!pip) {
      setTimeout(boot, 200);
      return;
    }
    init(pip);
  }
  setTimeout(boot, 300);

  function init(pip) {
    // ── Build inner HTML ──
    pip.innerHTML =
      '<div class="xp-resize" id="xp-resize"></div>' +
      '<div class="xp-titlebar" id="xp-titlebar">' +
      '<span class="xp-logo">⚡</span>' +
      '<span class="xp-title">Activity Monitor</span>' +
      '<span class="xp-dot live" id="xp-dot"></span>' +
      '<div class="xp-stats" id="xp-stats"></div>' +
      '<button class="xp-winbtn" id="xp-min" title="Minimize">−</button>' +
      "</div>" +
      '<div class="xp-toolbar" id="xp-toolbar">' +
      '<div class="xp-filters" id="xp-filters">' +
      '<button class="xp-fbtn active" data-f="all">All</button>' +
      '<button class="xp-fbtn" data-f="pipeline">Pipeline</button>' +
      '<button class="xp-fbtn" data-f="search">Search</button>' +
      '<button class="xp-fbtn" data-f="graph">Graph</button>' +
      '<button class="xp-fbtn" data-f="data">Data</button>' +
      '<button class="xp-fbtn" data-f="errors">Errors</button>' +
      "</div>" +
      '<span class="xp-spacer" style="flex:1"></span>' +
      '<span class="xp-count" id="xp-count">0</span>' +
      '<button class="xp-btn" id="xp-pause">Pause</button>' +
      '<button class="xp-btn" id="xp-clear">Clear</button>' +
      "</div>" +
      '<div class="xp-body" id="xp-body">' +
      '<div class="xp-list" id="xp-list">' +
      '<div class="xp-empty"><span class="xp-empty-icon">📡</span>Listening\u2026</div>' +
      "</div>" +
      "</div>";

    var list = document.getElementById("xp-list");
    var stats = document.getElementById("xp-stats");
    var countEl = document.getElementById("xp-count");
    var pauseBtn = document.getElementById("xp-pause");
    var dot = document.getElementById("xp-dot");
    var minBtn = document.getElementById("xp-min");

    var filter = "all",
      paused = false;
    var allEvents = []; // accumulates ALL events ever seen this session
    var highestSeq = 0; // last seq we received — for incremental polling
    var MAX_CLIENT = 2000; // keep up to 2000 events client-side

    // ── Filters ──
    document
      .getElementById("xp-filters")
      .addEventListener("click", function (e) {
        var btn = e.target.closest(".xp-fbtn");
        if (!btn) return;
        pip.querySelectorAll(".xp-fbtn").forEach(function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
        filter = btn.dataset.f;
        renderAll();
      });

    // ── Pause ──
    pauseBtn.addEventListener("click", function () {
      paused = !paused;
      pauseBtn.textContent = paused ? "Resume" : "Pause";
      dot.className = paused ? "xp-dot paused" : "xp-dot live";
    });

    // ── Clear ──
    document.getElementById("xp-clear").addEventListener("click", function () {
      fetch("/xray/api/clear", { method: "POST" });
      allEvents = [];
      highestSeq = 0;
      renderAll();
    });

    // ── Minimize / restore ──
    minBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      pip.classList.toggle("minimized");
      minBtn.textContent = pip.classList.contains("minimized") ? "+" : "−";
    });
    document
      .getElementById("xp-titlebar")
      .addEventListener("dblclick", function () {
        if (pip.classList.contains("minimized")) {
          pip.classList.remove("minimized");
          minBtn.textContent = "−";
        }
      });

    // ── Drag (titlebar) ──
    (function () {
      var bar = document.getElementById("xp-titlebar");
      var dragging = false,
        sx = 0,
        sy = 0,
        ox = 0,
        oy = 0;

      bar.addEventListener("mousedown", function (e) {
        if (e.target.closest(".xp-winbtn")) return;
        dragging = true;
        sx = e.clientX;
        sy = e.clientY;
        var rect = pip.getBoundingClientRect();
        ox = rect.left;
        oy = rect.top;
        document.body.style.userSelect = "none";
      });

      document.addEventListener("mousemove", function (e) {
        if (!dragging) return;
        var nx = ox + (e.clientX - sx);
        var ny = oy + (e.clientY - sy);
        nx = Math.max(0, Math.min(window.innerWidth - 60, nx));
        ny = Math.max(0, Math.min(window.innerHeight - 34, ny));
        pip.style.left = nx + "px";
        pip.style.top = ny + "px";
        pip.style.right = "auto";
        pip.style.bottom = "auto";
      });

      document.addEventListener("mouseup", function () {
        if (dragging) {
          dragging = false;
          document.body.style.userSelect = "";
        }
      });
    })();

    // ── Resize (top-left handle) ──
    (function () {
      var handle = document.getElementById("xp-resize");
      var resizing = false,
        sx = 0,
        sy = 0,
        sw = 0,
        sh = 0,
        origRight = 0,
        origBottom = 0;

      handle.addEventListener("mousedown", function (e) {
        e.stopPropagation();
        resizing = true;
        sx = e.clientX;
        sy = e.clientY;
        var rect = pip.getBoundingClientRect();
        sw = rect.width;
        sh = rect.height;
        origRight = window.innerWidth - rect.right;
        origBottom = window.innerHeight - rect.bottom;
        document.body.style.userSelect = "none";
      });

      document.addEventListener("mousemove", function (e) {
        if (!resizing) return;
        var dx = sx - e.clientX;
        var dy = sy - e.clientY;
        var nw = Math.max(280, sw + dx);
        var nh = Math.max(160, sh + dy);
        pip.style.width = nw + "px";
        pip.style.height = nh + "px";
        pip.style.right = origRight + "px";
        pip.style.bottom = origBottom + "px";
        pip.style.left = "auto";
        pip.style.top = "auto";
      });

      document.addEventListener("mouseup", function () {
        if (resizing) {
          resizing = false;
          document.body.style.userSelect = "";
        }
      });
    })();

    // ── Rendering helpers ──
    function tier(ms) {
      return ms < 100 ? "fast" : ms < 500 ? "med" : "slow";
    }
    function barPct(ms) {
      if (ms <= 0) return 5;
      return (
        Math.min(100, Math.max(5, (Math.log10(ms) / Math.log10(2000)) * 100)) |
        0
      );
    }
    function fmtDur(ms) {
      return ms < 1000 ? ms.toFixed(0) + "ms" : (ms / 1000).toFixed(1) + "s";
    }
    function fmtTs(ts) {
      var d = new Date(ts * 1000);
      return [d.getHours(), d.getMinutes(), d.getSeconds()]
        .map(function (v) {
          return String(v).padStart(2, "0");
        })
        .join(":");
    }
    function esc(s) {
      var d = document.createElement("div");
      d.textContent = s;
      return d.innerHTML;
    }

    function srcMeta(key) {
      return SOURCES[key] || { icon: "●", label: key, cls: "default" };
    }
    function opLabel(op) {
      return OP_LABELS[op] || op;
    }

    function filterEvents() {
      if (filter === "all") return allEvents;
      if (filter === "errors")
        return allEvents.filter(function (e) {
          return e.level === "error" || e.level === "warn";
        });
      var group = FILTER_GROUPS[filter];
      if (group) {
        return allEvents.filter(function (e) {
          return group.indexOf(e.source) !== -1;
        });
      }
      return allEvents.filter(function (e) {
        return e.source === filter;
      });
    }

    function buildRow(e) {
      var src = e.source || "?";
      var meta = srcMeta(src);
      var srcCls = "xp-src xp-src-" + meta.cls;
      var lvl = e.level || "info";
      var lvlCls =
        lvl === "warn"
          ? " warn"
          : lvl === "error"
            ? " error"
            : lvl === "perf"
              ? " perf"
              : "";

      var dur = "";
      if (e.duration_ms != null) {
        var t = tier(e.duration_ms),
          p = barPct(e.duration_ms);
        dur =
          '<span class="xp-dur"><span class="xp-bar"><span class="xp-fill ' +
          t +
          '" style="width:' +
          p +
          '%"></span></span><span class="xp-dlabel ' +
          t +
          '">' +
          fmtDur(e.duration_ms) +
          "</span></span>";
      }
      var dtl = e.detail
        ? '<span class="xp-dtl">' + esc(e.detail) + "</span>"
        : "";

      return (
        '<div class="xp-row' +
        lvlCls +
        '">' +
        '<span class="xp-ts">' +
        fmtTs(e.ts) +
        "</span>" +
        '<span class="' +
        srcCls +
        '"><span style="margin-right:3px">' +
        meta.icon +
        "</span>" +
        esc(meta.label) +
        "</span>" +
        '<span class="xp-op">' +
        esc(opLabel(e.op || "")) +
        "</span>" +
        '<span class="xp-msg">' +
        esc(e.message || "") +
        "</span>" +
        dur +
        dtl +
        "</div>"
      );
    }

    function renderAll() {
      var evts = filterEvents();
      if (!evts.length) {
        list.innerHTML =
          '<div class="xp-empty"><span class="xp-empty-icon">📡</span>No activity yet.</div>';
      } else {
        list.innerHTML = evts.map(buildRow).join("");
      }
      updateStats();
      var n = filter === "all" ? allEvents.length : evts.length;
      countEl.textContent = n + (n === 1 ? " event" : " events");
    }

    function updateStats() {
      var total = allEvents.length;
      var timed = allEvents
        .filter(function (e) {
          return e.duration_ms != null;
        })
        .map(function (e) {
          return e.duration_ms;
        });
      var avg = timed.length
        ? timed.reduce(function (a, b) {
            return a + b;
          }, 0) / timed.length
        : 0;
      var peak = timed.length ? Math.max.apply(null, timed) : 0;
      var errs = allEvents.filter(function (e) {
        return e.level === "error";
      }).length;
      var warns = allEvents.filter(function (e) {
        return e.level === "warn";
      }).length;

      var avgC = avg < 100 ? "g" : avg < 500 ? "w" : "r";
      var peakC = peak < 200 ? "g" : peak < 1000 ? "w" : "r";

      var h = '<span class="xp-stat"><b class="g">' + total + "</b> ops</span>";
      if (timed.length) {
        h +=
          '<span class="xp-stat">avg <b class="' +
          avgC +
          '">' +
          fmtDur(avg) +
          "</b></span>";
        h +=
          '<span class="xp-stat">peak <b class="' +
          peakC +
          '">' +
          fmtDur(peak) +
          "</b></span>";
      }
      if (errs)
        h += '<span class="xp-stat"><b class="r">' + errs + "</b> err</span>";
      if (warns)
        h += '<span class="xp-stat"><b class="w">' + warns + "</b> warn</span>";
      stats.innerHTML = h;
    }

    // ── Polling — incremental: only fetch events newer than highestSeq ──
    function poll() {
      if (paused) return;
      var url = "/xray/api/events";
      if (highestSeq > 0) url += "?since=" + highestSeq;
      fetch(url)
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          if (!data.length) return; // nothing new
          // data arrives newest-first; find the max seq in this batch
          var batchMax = 0;
          for (var i = 0; i < data.length; i++) {
            if (data[i].seq > batchMax) batchMax = data[i].seq;
          }
          // Merge into allEvents (newest-first order)
          // data is newest-first, allEvents is newest-first
          allEvents = data.concat(allEvents);
          // Trim to client max
          if (allEvents.length > MAX_CLIENT) allEvents.length = MAX_CLIENT;
          highestSeq = batchMax;
          renderAll();
        })
        .catch(function () {});
    }

    setInterval(poll, 800);
    poll();
  }
})();
