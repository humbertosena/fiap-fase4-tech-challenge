/* Painel de saúde: filtro de alertas por severidade e destaque de uma janela. */
(function () {
  "use strict";
  const buttons = Array.from(document.querySelectorAll("[data-filter-severity]"));
  const alerts = Array.from(document.querySelectorAll("#alertas [data-severity]"));
  const counter = document.getElementById("alert-count");

  function filterAlerts(severity) {
    let shown = 0;
    alerts.forEach((item) => {
      const visible = severity === "todos" || item.dataset.severity === severity;
      item.hidden = !visible;
      if (visible) shown += 1;
    });
    buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.filterSeverity === severity)));
    if (counter) counter.textContent = shown + (shown === 1 ? " alerta" : " alertas");
  }
  buttons.forEach((button) => button.addEventListener("click", () => filterAlerts(button.dataset.filterSeverity)));
  if (buttons.length) filterAlerts("todos");

  const picker = document.getElementById("window-picker");
  function highlight(window) {
    document.querySelectorAll("[data-window]").forEach((node) => {
      const cell = node.matches("th, td");
      if (cell) node.classList.toggle("is-selected", window !== "" && node.dataset.window === window);
    });
  }
  if (picker) picker.addEventListener("change", () => highlight(picker.value));
})();
