/* Chronos keyboard shortcuts */
document.addEventListener("keydown", function (e) {
  /* Esc — close detail panel */
  if (e.key === "Escape") {
    var backBtn = document.querySelector(".back-btn");
    if (backBtn) {
      backBtn.click();
      return;
    }
  }

  /* / — focus search (unless already typing) */
  if (
    e.key === "/" &&
    !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)
  ) {
    e.preventDefault();
    var searchInput = document.getElementById("search-input");
    if (searchInput) {
      searchInput.focus();
    }
  }
});

/* ── Auto-scroll to highlighted event card ───────────────────────────── */
(function () {
  var observer = new MutationObserver(function () {
    var el = document.querySelector(".event-card.highlighted");
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      observer.disconnect();
    }
  });
  /* Watch for detail panel opening (content changes) */
  observer.observe(document.body, { childList: true, subtree: true });

  /* Re-attach after Dash re-renders */
  document.addEventListener("DOMContentLoaded", function () {
    new MutationObserver(function () {
      var highlighted = document.querySelector(".event-card.highlighted");
      if (highlighted) {
        setTimeout(function () {
          highlighted.scrollIntoView({ behavior: "smooth", block: "center" });
        }, 100);
      }
    }).observe(document.body, { childList: true, subtree: true });
  });
})();

/* ── Topic grid: live search + sort ──────────────────────────────────────── */
(function () {
  function filterTopics() {
    var searchEl = document.getElementById("topic-search-input");
    var grid = document.getElementById("topics-grid-container");
    if (!searchEl || !grid) return;

    var query = (searchEl.value || "").toLowerCase();
    var cards = grid.querySelectorAll(".topic-card");

    cards.forEach(function (card) {
      var name = card.querySelector(".topic-name");
      if (!name) return;
      var text = name.textContent.toLowerCase();
      card.style.display = text.indexOf(query) !== -1 ? "" : "none";
    });
  }

  /* Debounced input handler */
  var timer;
  document.addEventListener("input", function (e) {
    if (e.target && e.target.id === "topic-search-input") {
      clearTimeout(timer);
      timer = setTimeout(filterTopics, 120);
    }
  });
})();
