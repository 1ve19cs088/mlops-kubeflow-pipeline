(function () {
  const root = document.documentElement;
  const stored = localStorage.getItem("theme");

  if (stored) {
    root.setAttribute("data-bs-theme", stored);
  }

  const toggle = document.getElementById("theme-toggle");

  if (toggle) {
    toggle.addEventListener("click", function () {
      const current = root.getAttribute("data-bs-theme");
      const next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-bs-theme", next);
      localStorage.setItem("theme", next);
    });
  }
})();
