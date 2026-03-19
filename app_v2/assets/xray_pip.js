/**
 * X-ray Activity Monitor — 10x floating PiP debug panel.
 *
 * Features:
 *   - Live event stream with incremental DOM updates
 *   - Throughput sparkline (30-bar rolling event rate)
 *   - Session cost ticker (live $ from /xray/api/costs)
 *   - 7 filter tabs (All, Pipeline, Search, Graph, Data, Costs, Errors)
 *   - Click-to-expand event detail (raw metadata JSON)
 *   - Keyboard shortcuts: X toggle, Space pause, C clear, Esc close
 *   - Drag-to-move, resize, minimize
 *   - Source grouping for consecutive same-source bursts
 */
(function () {
  ("use strict");

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
    notion: { icon: "📝", label: "Notion", cls: "notion" },
    openai: { icon: "💬", label: "OpenAI", cls: "openai" },
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
    "ai-answer": "AI Answer",
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
    openai: "GPT",
    ingest: "Import",
    process: "Analyze",
    index: "Index",
    import: "Import",
    sync: "Sync",
    match: "Match",
    cost: "Cost",
  };

  // Filter groups — which source keys belong to each tab
  var FILTER_GROUPS = {
    all: null,
    pipeline: [
      "ingest",
      "gemini",
      "embed",
      "qdrant",
      "pipeline",
      "sync",
      "notion",
    ],
    search: ["search"],
    graph: ["graph"],
    data: ["data", "day", "detail", "nav"],
    costs: null, // special: show cost-related events
    errors: null, // special: filter by level
  };

  // ── State ──
  var allEvents = [];
  var highestSeq = 0;
  var filter = "all";
  var paused = false;
  var sessionCost = 0;
  var MAX_CLIENT = 2000;
  var MAX_VISIBLE = 500;
  var POLL_INTERVAL = 800;
  var COST_POLL_INTERVAL = 5000;
  var SPARKLINE_BARS = 30;
  var expandedSeq = null; // which event row is expanded (seq or null)

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
      // ── Titlebar ──
      '<div class="xp-titlebar" id="xp-titlebar">' +
      '<span class="xp-logo">⚡</span>' +
      '<span class="xp-title">X-ray</span>' +
      '<span class="xp-dot live" id="xp-dot"></span>' +
      '<div class="xp-stats" id="xp-stats"></div>' +
      '<span class="xp-cost-ticker" id="xp-cost-ticker" title="Session API cost">$0.00</span>' +
      '<button class="xp-winbtn" id="xp-min" title="Minimize (Esc)">−</button>' +
      "</div>" +
      // ── Toolbar ──
      '<div class="xp-toolbar" id="xp-toolbar">' +
      '<div class="xp-filters" id="xp-filters">' +
      '<button class="xp-fbtn active" data-f="all">All</button>' +
      '<button class="xp-fbtn" data-f="pipeline">Pipeline</button>' +
      '<button class="xp-fbtn" data-f="search">Search</button>' +
      '<button class="xp-fbtn" data-f="graph">Graph</button>' +
      '<button class="xp-fbtn" data-f="data">Data</button>' +
      '<button class="xp-fbtn" data-f="costs">💲 Costs</button>' +
      '<button class="xp-fbtn" data-f="errors">Errors</button>' +
      "</div>" +
      '<span style="flex:1"></span>' +
      '<div class="xp-sparkline" id="xp-sparkline" title="Events/sec (30s)"></div>' +
      '<span class="xp-count" id="xp-count">0</span>' +
      '<button class="xp-btn" id="xp-pause" title="Space">Pause</button>' +
      '<button class="xp-btn" id="xp-clear" title="C">Clear</button>' +
      "</div>" +
      // ── Body ──
      '<div class="xp-body" id="xp-body">' +
      '<div class="xp-list" id="xp-list">' +
      '<div class="xp-empty"><span class="xp-empty-icon">📡</span>Listening\u2026<div class="xp-hint">Press <kbd>X</kbd> to toggle \u2022 <kbd>Space</kbd> to pause</div></div>' +
      "</div>" +
      "</div>";

    var list = document.getElementById("xp-list");
    var stats = document.getElementById("xp-stats");
    var countEl = document.getElementById("xp-count");
    var pauseBtn = document.getElementById("xp-pause");
    var dot = document.getElementById("xp-dot");
    var minBtn = document.getElementById("xp-min");
    var costEl = document.getElementById("xp-cost-ticker");
    var sparkEl = document.getElementById("xp-sparkline");

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
    pauseBtn.addEventListener("click", togglePause);
    function togglePause() {
      paused = !paused;
      pauseBtn.textContent = paused ? "Resume" : "Pause";
      dot.className = paused ? "xp-dot paused" : "xp-dot live";
    }

    // ── Clear ──
    document.getElementById("xp-clear").addEventListener("click", doClear);
    function doClear() {
      fetch("/xray/api/clear", { method: "POST" });
      allEvents = [];
      highestSeq = 0;
      expandedSeq = null;
      renderAll();
    }

    // ── Minimize / restore ──
    minBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      toggleMinimize();
    });
    document
      .getElementById("xp-titlebar")
      .addEventListener("dblclick", function () {
        if (pip.classList.contains("minimized")) toggleMinimize();
      });
    function toggleMinimize() {
      pip.classList.toggle("minimized");
      minBtn.textContent = pip.classList.contains("minimized") ? "+" : "−";
    }

    // ── Keyboard shortcuts ──
    document.addEventListener("keydown", function (e) {
      // Don't fire when user is typing in inputs
      if (
        e.target.tagName === "INPUT" ||
        e.target.tagName === "TEXTAREA" ||
        e.target.tagName === "SELECT" ||
        e.target.isContentEditable
      )
        return;

      var key = e.key.toLowerCase();

      if (key === "x") {
        e.preventDefault();
        toggleMinimize();
      } else if (key === " " && !pip.classList.contains("minimized")) {
        // Only capture Space when panel is visible
        e.preventDefault();
        togglePause();
      } else if (
        key === "c" &&
        !e.ctrlKey &&
        !e.metaKey &&
        !pip.classList.contains("minimized")
      ) {
        e.preventDefault();
        doClear();
      } else if (key === "escape" && !pip.classList.contains("minimized")) {
        e.preventDefault();
        toggleMinimize();
      }
    });

    // ── Click-to-expand event detail ──
    list.addEventListener("click", function (e) {
      var row = e.target.closest(".xp-row");
      if (!row) return;
      var seq = parseInt(row.dataset.seq, 10);
      if (!seq) return;

      if (expandedSeq === seq) {
        // Collapse
        expandedSeq = null;
        var dtlEl = row.querySelector(".xp-expand");
        if (dtlEl) dtlEl.remove();
        row.classList.remove("expanded");
      } else {
        // Collapse previous
        var prev = list.querySelector(".xp-row.expanded");
        if (prev) {
          prev.classList.remove("expanded");
          var prevDtl = prev.querySelector(".xp-expand");
          if (prevDtl) prevDtl.remove();
        }
        // Expand this one
        expandedSeq = seq;
        row.classList.add("expanded");
        var evt = allEvents.find(function (ev) {
          return ev.seq === seq;
        });
        if (evt) {
          var detail = document.createElement("div");
          detail.className = "xp-expand";
          detail.innerHTML =
            '<div class="xp-expand-grid">' +
            '<span class="xp-expand-key">seq</span><span class="xp-expand-val">' +
            evt.seq +
            "</span>" +
            '<span class="xp-expand-key">source</span><span class="xp-expand-val">' +
            esc(evt.source) +
            "</span>" +
            '<span class="xp-expand-key">op</span><span class="xp-expand-val">' +
            esc(evt.op || "") +
            "</span>" +
            '<span class="xp-expand-key">level</span><span class="xp-expand-val xp-lvl-' +
            (evt.level || "info") +
            '">' +
            esc(evt.level || "info") +
            "</span>" +
            '<span class="xp-expand-key">time</span><span class="xp-expand-val">' +
            new Date(evt.ts * 1000).toISOString() +
            "</span>" +
            (evt.duration_ms != null
              ? '<span class="xp-expand-key">duration</span><span class="xp-expand-val">' +
                fmtDur(evt.duration_ms) +
                "</span>"
              : "") +
            (evt.detail
              ? '<span class="xp-expand-key">detail</span><span class="xp-expand-val xp-expand-detail">' +
                esc(evt.detail) +
                "</span>"
              : "") +
            '<span class="xp-expand-key">message</span><span class="xp-expand-val xp-expand-msg">' +
            esc(evt.message || "") +
            "</span>" +
            "</div>";
          row.appendChild(detail);
        }
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
        if (
          e.target.closest(".xp-winbtn") ||
          e.target.closest(".xp-cost-ticker")
        )
          return;
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

    // ══════════════════════════════════════════════════════════════════════
    // Rendering
    // ══════════════════════════════════════════════════════════════════════

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
    function fmtTsMs(ts) {
      var d = new Date(ts * 1000);
      var ms = String(d.getMilliseconds()).padStart(3, "0");
      return fmtTs(ts) + "." + ms;
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
      if (filter === "costs")
        return allEvents.filter(function (e) {
          return (
            e.op === "cost" ||
            e.op === "ai-answer" ||
            e.op === "openai" ||
            (e.detail && e.detail.indexOf("token") !== -1) ||
            (e.detail && e.detail.indexOf("cost") !== -1)
          );
        });
      var group = FILTER_GROUPS[filter];
      if (group)
        return allEvents.filter(function (e) {
          return group.indexOf(e.source) !== -1;
        });
      return allEvents.filter(function (e) {
        return e.source === filter;
      });
    }

    function matchesFilter(e) {
      if (filter === "all") return true;
      if (filter === "errors") return e.level === "error" || e.level === "warn";
      if (filter === "costs") {
        return (
          e.op === "cost" ||
          e.op === "ai-answer" ||
          e.op === "openai" ||
          (e.detail && e.detail.indexOf("token") !== -1) ||
          (e.detail && e.detail.indexOf("cost") !== -1)
        );
      }
      var group = FILTER_GROUPS[filter];
      if (group) return group.indexOf(e.source) !== -1;
      return e.source === filter;
    }

    function buildRow(e, fresh) {
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
      var freshCls = fresh ? " xp-fresh" : "";
      var expandCls = expandedSeq === e.seq ? " expanded" : "";

      var dur = "";
      if (e.duration_ms != null) {
        var t = tier(e.duration_ms),
          p = barPct(e.duration_ms);
        dur =
          '<span class="xp-dur">' +
          '<span class="xp-bar"><span class="xp-fill ' +
          t +
          '" style="width:' +
          p +
          '%"></span></span>' +
          '<span class="xp-dlabel ' +
          t +
          '">' +
          fmtDur(e.duration_ms) +
          "</span>" +
          "</span>";
      }
      var dtl = e.detail
        ? ' <span class="xp-dtl">' + esc(e.detail) + "</span>"
        : "";

      var msgContent = e.message || "";
      var msgLine =
        msgContent || dtl
          ? '<div class="xp-msg">' + esc(msgContent) + dtl + "</div>"
          : "";

      return (
        '<div class="xp-row' +
        lvlCls +
        freshCls +
        expandCls +
        '" data-seq="' +
        e.seq +
        '">' +
        '<div class="xp-meta">' +
        '<span class="xp-ts">' +
        fmtTsMs(e.ts) +
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
        '<span style="flex:1"></span>' +
        dur +
        "</div>" +
        msgLine +
        "</div>"
      );
    }

    function updateCount() {
      var evts = filterEvents();
      var n = filter === "all" ? allEvents.length : evts.length;
      countEl.textContent = n + (n === 1 ? " event" : " events");
    }

    function renderAll() {
      expandedSeq = null;
      var evts = filterEvents();
      if (!evts.length) {
        list.innerHTML =
          '<div class="xp-empty"><span class="xp-empty-icon">📡</span>Listening\u2026' +
          '<div class="xp-hint">Press <kbd>X</kbd> to toggle \u2022 <kbd>Space</kbd> to pause</div></div>';
      } else {
        list.innerHTML = evts
          .map(function (e) {
            return buildRow(e, false);
          })
          .join("");
      }
      updateStats();
      updateCount();
    }

    function appendNewRows(newEvents) {
      var empty = list.querySelector(".xp-empty");
      if (empty) empty.remove();

      var wasAtTop = list.scrollTop < 10;

      var filtered = newEvents.filter(matchesFilter);
      if (!filtered.length) {
        updateCount();
        return;
      }

      var html = filtered
        .map(function (e) {
          return buildRow(e, true);
        })
        .join("");
      var temp = document.createElement("div");
      temp.innerHTML = html;

      var frag = document.createDocumentFragment();
      while (temp.firstChild) frag.appendChild(temp.firstChild);
      list.insertBefore(frag, list.firstChild);

      while (list.children.length > MAX_VISIBLE)
        list.removeChild(list.lastChild);
      if (wasAtTop) list.scrollTop = 0;
      updateCount();
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

    // ══════════════════════════════════════════════════════════════════════
    // Sparkline — 30-bar throughput visualization
    // ══════════════════════════════════════════════════════════════════════

    function renderSparkline(buckets) {
      if (!buckets || !buckets.length) return;
      var max = Math.max.apply(null, buckets) || 1;
      var bars = buckets.map(function (v) {
        var pct = Math.max(2, (v / max) * 100);
        var cls =
          v === 0
            ? "xp-spark-bar empty"
            : v > max * 0.7
              ? "xp-spark-bar hot"
              : "xp-spark-bar";
        return (
          '<div class="' +
          cls +
          '" style="height:' +
          pct +
          '%" title="' +
          v +
          ' evt/s"></div>'
        );
      });
      sparkEl.innerHTML = bars.join("");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Cost ticker — session spend from /xray/api/costs
    // ══════════════════════════════════════════════════════════════════════

    function updateCostTicker(data) {
      if (!data || !data.session) return;
      var s = data.session;
      var total = s.total_cost_usd != null ? s.total_cost_usd : 0;
      sessionCost = total;
      if (total < 0.01) {
        costEl.textContent = "$0.00";
        costEl.className = "xp-cost-ticker";
      } else if (total < 0.1) {
        costEl.textContent = "$" + total.toFixed(3);
        costEl.className = "xp-cost-ticker cost-low";
      } else if (total < 1.0) {
        costEl.textContent = "$" + total.toFixed(2);
        costEl.className = "xp-cost-ticker cost-med";
      } else {
        costEl.textContent = "$" + total.toFixed(2);
        costEl.className = "xp-cost-ticker cost-high";
      }
    }

    function pollCosts() {
      fetch("/xray/api/costs?days=1")
        .then(function (r) {
          return r.json();
        })
        .then(updateCostTicker)
        .catch(function () {});
    }

    // ══════════════════════════════════════════════════════════════════════
    // Polling — events, throughput, costs
    // ══════════════════════════════════════════════════════════════════════

    function pollEvents() {
      if (paused) return;
      var url = "/xray/api/events";
      if (highestSeq > 0) url += "?since=" + highestSeq;
      fetch(url)
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          if (!data.length) return;
          var batchMax = 0;
          for (var i = 0; i < data.length; i++) {
            if (data[i].seq > batchMax) batchMax = data[i].seq;
          }
          allEvents = data.concat(allEvents);
          if (allEvents.length > MAX_CLIENT) allEvents.length = MAX_CLIENT;
          highestSeq = batchMax;
          appendNewRows(data);
          updateStats();
        })
        .catch(function () {});
    }

    function pollThroughput() {
      if (paused) return;
      fetch("/xray/api/throughput?buckets=" + SPARKLINE_BARS)
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          renderSparkline(data.buckets);
        })
        .catch(function () {});
    }

    // Start polling loops
    setInterval(pollEvents, POLL_INTERVAL);
    setInterval(pollThroughput, 2000);
    setInterval(pollCosts, COST_POLL_INTERVAL);
    pollEvents();
    pollThroughput();
    pollCosts();
  }
})();
