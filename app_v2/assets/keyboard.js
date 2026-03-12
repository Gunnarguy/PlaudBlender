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
