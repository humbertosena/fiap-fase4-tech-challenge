/* Etapa 2 — linha do tempo, calculadora de PSI, mapa de calor, concept drift e degradação. */
(function () {
  "use strict";
  const { payload, fmt, cssVar, el, chart, renderTable } = window.CS;
  const report = payload.report;
  const KEYS = payload.windows.map((w) => w.key);
  const WINDOWS = KEYS.filter((key) => key !== "reference");
  const LABEL = Object.fromEntries(payload.windows.map((w) => [w.key, w.label]));
  const SHORT = { reference: "Referência", mes_1: "Mês 1", mes_2: "Mês 2", mes_3_crise: "Mês 3" };
  const FEATURE_LABELS = {
    renda_mensal: "Renda mensal", score_bureau: "Score de bureau",
    taxa_endividamento: "Taxa de endividamento", utilizacao_credito: "Utilização de crédito",
  };
  const LEVEL = { estavel: "estável", atencao: "atenção", critico: "crítico" };
  const POINTS = { reference: "rect", mes_1: "circle", mes_2: "triangle", mes_3_crise: "rectRot" };
  const sum = (values) => values.reduce((a, b) => a + b, 0);
  const color = (key) => ({
    reference: cssVar("--muted"), mes_1: cssVar("--series-1"), mes_2: cssVar("--series-2"), mes_3_crise: cssVar("--critical"),
  })[key];
  const pValue = (p) => (p < 1e-3 ? p.toExponential(1).replace(".", ",") : fmt.num(p, 4));
  const levelOf = (psi) => (psi < 0.1 ? "estavel" : psi < 0.25 ? "atencao" : "critico");
  const featureLabel = (name) => FEATURE_LABELS[name] || name;
  const numericOf = (window, feature) => report.windows[window].drift.numeric.find((r) => r.feature === feature);

  function binLabel(feature, low, high) {
    const f = (x) => (feature === "renda_mensal" ? fmt.compact(x) : feature === "score_bureau" ? fmt.int(x) : fmt.num(x, 2));
    return `${f(low)}–${f(high)}`;
  }

  /* ---------- 1. Linha do tempo ---------- */
  let selected = "mes_3_crise";
  const tlFeature = document.getElementById("tl-feature");
  const buttons = document.getElementById("window-buttons");

  function shares(counts) {
    const total = sum(counts);
    return counts.map((c) => (100 * c) / total);
  }

  function timelineRows() {
    const feature = tlFeature.value;
    const h = payload.histograms[feature];
    const ref = shares(h.counts.reference);
    const cur = shares(h.counts[selected]);
    return ref.map((value, i) => ({ label: binLabel(feature, h.edges[i], h.edges[i + 1]), ref: value, cur: cur[i] }));
  }

  function renderBadges() {
    const box = document.getElementById("tl-badges");
    const card = payload.windows.find((w) => w.key === selected);
    const badge = (label, value) => el("span", { class: "badge" }, [label, el("strong", { text: value })]);
    const items = [badge("Inadimplência", fmt.pct(card.base_rate, 1))];
    if (selected !== "reference") {
      const r = numericOf(selected, tlFeature.value);
      items.push(
        badge("KS p", pValue(r.ks_p_value)),
        badge("PSI", `${fmt.num(r.psi, 3)} · ${LEVEL[r.psi_level]}`),
        badge("Média", `${fmt.num(r.mean_ref, 2)} → ${fmt.num(r.mean_cur, 2)}`),
      );
    }
    box.replaceChildren(...items);
    document.getElementById("tl-caption").textContent = selected === "reference"
      ? "A referência sozinha: clientes por faixa, em % do conjunto."
      : `Referência contra ${LABEL[selected]}: clientes por faixa, em % de cada conjunto.`;
  }

  function setupTimeline() {
    Object.keys(payload.histograms).forEach((name) => tlFeature.append(el("option", { value: name, text: featureLabel(name) })));
    KEYS.forEach((key) => {
      const button = el("button", { class: "btn", type: "button", "aria-pressed": String(key === selected), text: LABEL[key] });
      button.addEventListener("click", () => {
        selected = key;
        buttons.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
        renderBadges();
        timeline.update();
      });
      buttons.append(button);
    });
    const timeline = chart({
      canvasId: "tl-chart",
      tableId: "tl-table",
      table: () => ({
        headers: [{ label: "Faixa" }, { label: "Referência (%)", num: true }, { label: `${SHORT[selected]} (%)`, num: true }],
        rows: timelineRows().map((r) => [r.label, fmt.num(r.ref, 1), fmt.num(r.cur, 1)]),
      }),
      config: () => {
        const rows = timelineRows();
        const datasets = [{ label: "Referência", data: rows.map((r) => r.ref), backgroundColor: color("reference") + "99", borderColor: color("reference"), borderWidth: 1, borderRadius: 3, borderSkipped: "start" }];
        if (selected !== "reference") {
          datasets.push({ label: SHORT[selected], data: rows.map((r) => r.cur), backgroundColor: color(selected) + "cc", borderRadius: 3, borderSkipped: "start" });
        }
        return {
          type: "bar",
          data: { labels: rows.map((r) => r.label), datasets },
          options: {
            interaction: { mode: "index", intersect: false },
            plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt.num(c.parsed.y, 1)} %` } } },
            scales: { x: { grid: { display: false }, ticks: { maxRotation: 60, autoSkip: true } }, y: { title: { display: true, text: "% do conjunto" }, beginAtZero: true } },
          },
        };
      },
    });
    tlFeature.addEventListener("change", () => { renderBadges(); timeline.update(); });
    renderBadges();
  }

  /* ---------- 2. Calculadora de PSI ---------- */
  const psiFeature = document.getElementById("psi-feature");
  const slider = document.getElementById("psi-slider");
  let psiChart;
  const prepared = {};

  function prepare(calc) {
    const total = sum(calc.counts);
    const cum = [0];
    calc.counts.forEach((c) => cum.push(cum[cum.length - 1] + c / total));
    const edges = calc.edges;
    const cdf = (x) => {
      if (x <= edges[0]) return 0;
      if (x >= edges[edges.length - 1]) return 1;
      let lo = 0;
      let hi = edges.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (edges[mid] <= x) lo = mid; else hi = mid; }
      return cum[lo] + ((x - edges[lo]) / (edges[lo + 1] - edges[lo])) * (cum[lo + 1] - cum[lo]);
    };
    const cuts = [];
    for (let k = 1; k < 10; k += 1) {
      let lo = edges[0];
      let hi = edges[edges.length - 1];
      for (let i = 0; i < 60; i += 1) { const mid = (lo + hi) / 2; if (cdf(mid) < k / 10) lo = mid; else hi = mid; }
      cuts.push((lo + hi) / 2);
    }
    return { cdf, cuts };
  }

  function psiAt(feature, shiftSd) {
    const calc = payload.psi_calculator[feature];
    const { cdf, cuts } = prepared[feature];
    const delta = shiftSd * calc.std;
    const shifted = (x) => (x === -Infinity ? 0 : x === Infinity ? 1 : cdf(x - delta));
    const bounds = [-Infinity, ...cuts, Infinity];
    let psi = 0;
    for (let i = 0; i < 10; i += 1) {
      const share = Math.max(shifted(bounds[i + 1]) - shifted(bounds[i]), 1e-6);
      psi += (share - 0.1) * Math.log(share / 0.1);
    }
    return psi;
  }

  function shiftedBins(feature, shiftSd) {
    const calc = payload.psi_calculator[feature];
    const { cdf } = prepared[feature];
    const delta = shiftSd * calc.std;
    const total = sum(calc.counts);
    return calc.counts.map((count, i) => ({
      mid: (calc.edges[i] + calc.edges[i + 1]) / 2,
      ref: (100 * count) / total,
      cur: 100 * (cdf(calc.edges[i + 1] - delta) - cdf(calc.edges[i] - delta)),
    }));
  }

  function realShift(key) {
    return payload.psi_calculator[psiFeature.value].observed_shift_sd[key];
  }

  function renderPsi() {
    const feature = psiFeature.value;
    const shift = Number(slider.value);
    const calc = payload.psi_calculator[feature];
    const psi = psiAt(feature, shift);
    document.getElementById("psi-shift").textContent = `${fmt.num(shift, 2)} desvio-padrão (${fmt.num(shift * calc.std, 3)} na unidade da feature)`;
    document.getElementById("psi-value").textContent = fmt.num(psi, 3);
    const level = levelOf(psi);
    const status = el("span", { class: "status " + (level === "estavel" ? "ok" : level === "atencao" ? "na" : "fail"), text: LEVEL[level] });
    document.getElementById("psi-level").replaceChildren(status);
    document.getElementById("psi-needle").style.left = Math.min(psi, 1) * 100 + "%";
    const bits = WINDOWS.map((key) => `${SHORT[key]}: PSI real ${fmt.num(numericOf(key, feature).psi, 3)} com deslocamento de ${fmt.num(realShift(key), 2)} dp`);
    document.getElementById("psi-note").textContent = `A calculadora só desloca a média, sem mudar a forma. ${bits.join(" · ")}.`;
    if (psiChart) psiChart.update();
  }

  function setupPsi() {
    Object.keys(payload.psi_calculator).forEach((name) => {
      prepared[name] = prepare(payload.psi_calculator[name]);
      psiFeature.append(el("option", { value: name, text: featureLabel(name) }));
    });
    const labelButtons = () => WINDOWS.slice(1).forEach((key) => {
      document.getElementById("psi-real-" + key).textContent = `Aplicar o deslocamento de média do ${SHORT[key]} (${fmt.num(realShift(key), 2)} dp)`;
    });
    const apply = (value) => { slider.value = String(Math.max(-1, Math.min(1, value))); renderPsi(); };
    WINDOWS.slice(1).forEach((key) => document.getElementById("psi-real-" + key).addEventListener("click", () => apply(realShift(key))));
    document.getElementById("psi-zero").addEventListener("click", () => apply(0));
    slider.addEventListener("input", renderPsi);
    psiFeature.addEventListener("change", () => { labelButtons(); renderPsi(); });
    labelButtons();
    psiChart = chart({
      canvasId: "psi-chart",
      tableId: "psi-table",
      table: () => ({
        headers: [{ label: "Valor central da faixa" }, { label: "Referência (%)", num: true }, { label: "Deslocada (%)", num: true }],
        rows: shiftedBins(psiFeature.value, Number(slider.value)).filter((_, i) => i % 3 === 0).map((b) => [fmt.num(b.mid, 2), fmt.num(b.ref, 2), fmt.num(b.cur, 2)]),
      }),
      config: () => {
        const bins = shiftedBins(psiFeature.value, Number(slider.value));
        return {
          type: "line",
          data: {
            labels: bins.map((b) => fmt.num(b.mid, 2)),
            datasets: [
              { label: "Referência", data: bins.map((b) => b.ref), borderColor: color("reference"), borderDash: [6, 4], borderWidth: 2, pointRadius: 0, tension: 0.3 },
              { label: "Deslocada", data: bins.map((b) => b.cur), borderColor: color("mes_3_crise"), borderWidth: 2.5, pointRadius: 0, tension: 0.3 },
            ],
          },
          options: {
            interaction: { mode: "index", intersect: false },
            plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt.num(c.parsed.y, 2)} %` } } },
            scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 0 } }, y: { title: { display: true, text: "% por faixa" }, beginAtZero: true } },
          },
        };
      },
    });
    renderPsi();
  }

  /* ---------- 3. Mapa de calor ---------- */
  const heat = document.getElementById("heatmap");
  const detail = document.getElementById("heat-detail");

  function cell(window, row, categorical) {
    const level = categorical ? (row.jensen_shannon >= 0.1 ? "atencao" : "estavel") : row.psi_level;
    const value = categorical ? row.jensen_shannon : row.psi;
    const sub = categorical
      ? (row.chi2_drift ? "JS · χ² significativo" : "JS · χ² não signif.")
      : LEVEL[level];
    const button = el("button", { type: "button", class: "lvl-" + level, "data-window": window, "data-feature": row.feature, "aria-pressed": "false", "aria-label": `${row.feature}, ${SHORT[window]}: ${fmt.num(value, 3)}, ${sub}` }, [
      fmt.num(value, 3), el("span", { class: "sub", text: sub }),
    ]);
    button.addEventListener("click", () => {
      heat.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      showDetail(window, row, categorical);
    });
    return el("td", {}, [button]);
  }

  function showDetail(window, row, categorical) {
    const flag = report.windows[window].evidently.flags[row.feature];
    const verdict = `Evidently: ${flag ? "drift detectado" : "sem drift"}.`;
    if (categorical) {
      const shares = (obj) => Object.entries(obj).map(([k, v]) => `${k} ${fmt.pct(v, 1)}`).join(", ");
      detail.textContent = `${row.feature} · ${LABEL[window]}: qui-quadrado ${fmt.num(row.chi2_statistic, 1)}, p = ${pValue(row.chi2_p_value)}; Jensen-Shannon ${fmt.num(row.jensen_shannon, 3)}. Referência: ${shares(row.shares_ref)}. Janela: ${shares(row.shares_cur)}. ${verdict}`;
      return;
    }
    detail.textContent = `${row.feature} · ${LABEL[window]}: KS D = ${fmt.num(row.ks_statistic, 3)}, p = ${pValue(row.ks_p_value)} (com Bonferroni: ${row.ks_drift_bonferroni ? "drift" : "sem drift"}); PSI ${fmt.num(row.psi, 3)} (${LEVEL[row.psi_level]}); Wasserstein normalizado ${fmt.num(row.wasserstein_norm, 3)}; média ${fmt.num(row.mean_ref, 3)} → ${fmt.num(row.mean_cur, 3)}. ${verdict}`;
  }

  function setupHeatmap() {
    const head = el("tr", {}, [el("th", { text: "Feature" }), ...WINDOWS.map((key) => el("th", { class: "num", text: SHORT[key] }))]);
    const rows = [];
    report.windows[WINDOWS[0]].drift.numeric.forEach((_, i) => {
      rows.push(el("tr", {}, [el("th", { class: "row", scope: "row", text: report.windows[WINDOWS[0]].drift.numeric[i].feature }), ...WINDOWS.map((key) => cell(key, report.windows[key].drift.numeric[i], false))]));
    });
    report.windows[WINDOWS[0]].drift.categorical.forEach((_, i) => {
      rows.push(el("tr", {}, [el("th", { class: "row", scope: "row", text: report.windows[WINDOWS[0]].drift.categorical[i].feature }), ...WINDOWS.map((key) => cell(key, report.windows[key].drift.categorical[i], true))]));
    });
    heat.replaceChildren(el("thead", {}, [head]), el("tbody", {}, rows));
  }

  /* ---------- 4. Concept drift ---------- */
  const edges = payload.band_curve.edges;
  const active = new Set(KEYS);
  const bandLabels = edges.slice(0, -1).map((low, i) => (i === 0 ? `< ${edges[1]}` : i === edges.length - 2 ? `≥ ${low}` : `${low}–${edges[i + 1] - 1}`));
  const MID_BANDS = edges.slice(0, -1).map((low, i) => (low >= 550 && edges[i + 1] <= 700 ? i : -1)).filter((i) => i >= 0);

  function midRate(key) {
    const counts = payload.band_curve.counts[key];
    const rates = payload.band_curve.default_rate[key];
    const total = sum(MID_BANDS.map((i) => counts[i]));
    return sum(MID_BANDS.map((i) => counts[i] * (rates[i] || 0))) / total;
  }

  function setupBands() {
    const toggles = document.getElementById("band-toggles");
    let bandChart;
    KEYS.forEach((key) => {
      const input = el("input", { type: "checkbox", id: "band-" + key });
      input.checked = true;
      input.addEventListener("change", () => { if (input.checked) active.add(key); else active.delete(key); bandChart.update(); });
      toggles.append(el("label", { class: "badge", for: "band-" + key }, [input, el("span", { class: "swatch", style: `border-color:${color(key)}` }), SHORT[key]]));
    });
    const ref = midRate("reference");
    const crisis = midRate("mes_3_crise");
    document.getElementById("band-caption").textContent = `Faixa de 550 a 700 pontos: ${fmt.pct(ref, 1)} na referência e ${fmt.pct(crisis, 1)} no Mês 3, ${fmt.num(crisis / ref, 1)} vezes mais. Fora dela, as linhas quase se sobrepõem.`;
    bandChart = chart({
      canvasId: "band-chart",
      tableId: "band-table",
      table: () => ({
        headers: [{ label: "Faixa de score" }, ...KEYS.map((key) => ({ label: SHORT[key], num: true }))],
        rows: bandLabels.map((label, i) => [label, ...KEYS.map((key) => fmt.pct(payload.band_curve.default_rate[key][i], 1))]),
      }),
      config: () => ({
        type: "line",
        data: {
          labels: bandLabels,
          datasets: KEYS.filter((key) => active.has(key)).map((key) => ({
            label: LABEL[key], data: payload.band_curve.default_rate[key].map((r) => (r === null ? null : 100 * r)),
            borderColor: color(key), backgroundColor: color(key), borderWidth: 2.5, borderDash: key === "reference" ? [6, 4] : [],
            pointStyle: POINTS[key], pointRadius: 5, pointHoverRadius: 7, spanGaps: true,
          })),
        },
        options: {
          interaction: { mode: "index", intersect: false },
          plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt.num(c.parsed.y, 1)} %` } } },
          scales: { x: { grid: { display: false }, title: { display: true, text: "Faixa de score_bureau" } }, y: { title: { display: true, text: "Inadimplência (%)" }, beginAtZero: true } },
        },
      }),
    });
  }

  /* ---------- 5. Degradação ---------- */
  const metricSelect = document.getElementById("perf-metric");
  const METRICS = {
    roc_auc: { label: "AUC", digits: 3, get: (w) => w.performance.roc_auc },
    f1: { label: "F1 no limiar operacional", digits: 3, get: (w) => w.performance.f1 },
    brier: { label: "Brier (calibração; menor é melhor)", digits: 3, get: (w) => w.performance.brier },
    base_rate: { label: "Taxa de inadimplência (%)", digits: 1, get: (w) => 100 * w.performance.base_rate },
    approval: { label: "Taxa de aprovação (%)", digits: 1, get: (w) => 100 * w.predictions.approval_rate },
  };

  function perfRows() {
    const metric = METRICS[metricSelect.value];
    const rows = WINDOWS.map((key) => ({ key, label: SHORT[key], value: metric.get(report.windows[key]) }));
    if (metricSelect.value === "roc_auc") rows.unshift({ key: "reference", label: "Validação", value: report.reference.roc_auc_validation });
    return rows;
  }

  function setupDegradation() {
    Object.entries(METRICS).forEach(([key, metric]) => metricSelect.append(el("option", { value: key, text: metric.label })));
    const perf = chart({
      canvasId: "perf-chart",
      tableId: "perf-table",
      table: () => ({ headers: [{ label: "Janela" }, { label: METRICS[metricSelect.value].label, num: true }], rows: perfRows().map((r) => [r.label, fmt.num(r.value, METRICS[metricSelect.value].digits)]) }),
      config: () => {
        const rows = perfRows();
        const metric = METRICS[metricSelect.value];
        return {
          type: "bar",
          data: { labels: rows.map((r) => r.label), datasets: [{ label: metric.label, data: rows.map((r) => r.value), backgroundColor: rows.map((r) => color(r.key)), borderRadius: 4, borderSkipped: "start", maxBarThickness: 56 }] },
          options: {
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${metric.label}: ${fmt.num(c.parsed.y, metric.digits)}` } } },
            scales: { x: { grid: { display: false } }, y: { title: { display: true, text: metric.label }, beginAtZero: metricSelect.value !== "roc_auc", min: metricSelect.value === "roc_auc" ? 0.6 : undefined } },
          },
        };
      },
    });
    const caption = document.getElementById("perf-caption");
    const describe = () => {
      const rows = perfRows();
      caption.textContent = `${METRICS[metricSelect.value].label}: ${rows.map((r) => `${r.label} ${fmt.num(r.value, METRICS[metricSelect.value].digits)}`).join(" · ")}.`;
    };
    metricSelect.addEventListener("change", () => { perf.update(); describe(); });
    describe();
    const quantiles = [["p50", "P50", "--series-1"], ["p90", "P90", "--series-2"], ["p99", "P99", "--critical"]];
    chart({
      canvasId: "pct-chart",
      tableId: "pct-table",
      table: () => ({ headers: [{ label: "Janela" }, ...quantiles.map((q) => ({ label: q[1], num: true }))], rows: WINDOWS.map((key) => [SHORT[key], ...quantiles.map((q) => fmt.num(report.windows[key].predictions[q[0]], 3))]) }),
      config: () => ({
        type: "line",
        data: { labels: WINDOWS.map((key) => SHORT[key]), datasets: quantiles.map((q, i) => ({ label: q[1], data: WINDOWS.map((key) => report.windows[key].predictions[q[0]]), borderColor: cssVar(q[2]), backgroundColor: cssVar(q[2]), borderWidth: 2.5, pointStyle: ["circle", "triangle", "rectRot"][i], pointRadius: 5, tension: 0.2 })) },
        options: {
          interaction: { mode: "index", intersect: false },
          plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt.num(c.parsed.y, 3)}` } } },
          scales: { x: { grid: { display: false } }, y: { title: { display: true, text: "Score previsto" }, beginAtZero: true } },
        },
      }),
    });
  }

  setupTimeline();
  setupPsi();
  setupHeatmap();
  setupBands();
  setupDegradation();
})();
