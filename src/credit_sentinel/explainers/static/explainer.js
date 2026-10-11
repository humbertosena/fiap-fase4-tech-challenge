/* Credit Sentinel — utilitários comuns das páginas didáticas (JS vanilla). */
(function () {
  "use strict";
  const root = document.documentElement;
  const payloadNode = document.getElementById("payload");
  const payload = payloadNode ? JSON.parse(payloadNode.textContent) : {};
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const charts = [];

  const store = {
    get(key) { try { return window.localStorage.getItem(key); } catch (e) { return null; } },
    set(key, value) { try { window.localStorage.setItem(key, value); } catch (e) { /* sem armazenamento */ } },
  };

  const nf = (digits) => new Intl.NumberFormat("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const fmt = {
    num: (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : nf(d).format(v)),
    int: (v) => (v === null || v === undefined ? "—" : new Intl.NumberFormat("pt-BR").format(Math.round(v))),
    pct: (v, d = 1) => (v === null || v === undefined ? "—" : nf(d).format(v * 100) + " %"),
    brl: (v) => (v === null || v === undefined ? "—" : new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL", maximumFractionDigits: 0 }).format(v)),
    compact: (v) => new Intl.NumberFormat("pt-BR", { notation: "compact", maximumFractionDigits: 1 }).format(v),
  };

  function cssVar(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      if (key === "text") node.textContent = value;
      else if (key === "class") node.className = value;
      else node.setAttribute(key, value);
    });
    (children || []).forEach((child) => node.append(child));
    return node;
  }

  function renderTable(container, headers, rows) {
    if (!container) return;
    const head = el("tr", {}, headers.map((h) => el("th", { text: h.label, class: h.num ? "num" : "" })));
    const body = rows.map((row) => el("tr", {}, row.map((cell, i) => el("td", { text: cell, class: headers[i].num ? "num" : "" }))));
    container.replaceChildren(el("table", {}, [el("thead", {}, [head]), el("tbody", {}, body)]));
  }

  function applyChartDefaults() {
    if (!window.Chart) return;
    const Chart = window.Chart;
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.color = cssVar("--ink-2");
    Chart.defaults.borderColor = cssVar("--grid");
    Chart.defaults.animation = reducedMotion ? false : { duration: 350 };
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.labels.boxWidth = 12;
    Chart.defaults.plugins.tooltip.backgroundColor = cssVar("--ink");
    Chart.defaults.plugins.tooltip.titleColor = cssVar("--page");
    Chart.defaults.plugins.tooltip.bodyColor = cssVar("--page");
  }

  /**
   * Registra um gráfico com tabela equivalente. Sem Chart.js (offline), só a tabela aparece.
   * spec: { canvasId, tableId, config: () => chartConfig, table: () => ({headers, rows}) }
   */
  function chart(spec) {
    const entry = { spec, instance: null };
    charts.push(entry);
    draw(entry);
    return {
      update() { draw(entry); },
    };
  }

  function draw(entry) {
    const { spec } = entry;
    const canvas = document.getElementById(spec.canvasId);
    const table = document.getElementById(spec.tableId);
    const data = spec.table();
    renderTable(table, data.headers, data.rows);
    if (!window.Chart || !canvas) {
      if (canvas) canvas.closest(".chart-box").hidden = true;
      return;
    }
    if (table) table.hidden = true;
    if (entry.instance) entry.instance.destroy();
    entry.instance = new window.Chart(canvas, spec.config());
  }

  function redrawAll() {
    applyChartDefaults();
    charts.forEach(draw);
  }

  function setupTheme() {
    const button = document.querySelector(".theme-toggle");
    const saved = store.get("cs-theme");
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
    const current = () => root.dataset.theme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const label = () => { if (button) button.textContent = current() === "dark" ? "☀︎ Tema claro" : "☾ Tema escuro"; };
    label();
    if (!button) return;
    button.addEventListener("click", () => {
      root.dataset.theme = current() === "dark" ? "light" : "dark";
      store.set("cs-theme", root.dataset.theme);
      label();
      redrawAll();
    });
  }

  function setupCopy() {
    document.querySelectorAll("[data-copy]").forEach((button) => {
      button.addEventListener("click", async () => {
        const text = document.getElementById(button.dataset.copy).textContent;
        try {
          await navigator.clipboard.writeText(text);
        } catch (e) {
          const area = el("textarea", {});
          area.value = text;
          document.body.append(area);
          area.select();
          document.execCommand("copy");
          area.remove();
        }
        const original = button.textContent;
        button.textContent = "Copiado ✓";
        setTimeout(() => { button.textContent = original; }, 1400);
      });
    });
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, reducedMotion ? 0 : ms));
  }

  applyChartDefaults();
  setupTheme();
  setupCopy();
  const offline = document.getElementById("offline-note");
  if (offline && !window.Chart) offline.hidden = false;

  window.CS = { payload, fmt, cssVar, el, renderTable, chart, sleep, reducedMotion };
})();
