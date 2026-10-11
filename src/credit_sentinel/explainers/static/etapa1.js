/* Etapa 1 — validador, circuit breaker, distribuições e limiar do modelo. */
(function () {
  "use strict";
  const { payload, fmt, cssVar, el, chart, sleep } = window.CS;

  const FIELD_LABELS = {
    idade: "Idade (anos)",
    renda_mensal: "Renda mensal (R$)",
    score_bureau: "Score de bureau",
    taxa_endividamento: "Taxa de endividamento (0–1)",
    utilizacao_credito: "Utilização de crédito (0–1)",
    valor_parcela: "Valor da parcela (R$)",
    tempo_emprego_meses: "Tempo de emprego (meses)",
    situacao_emprego: "Situação de emprego",
  };
  const FEATURE_LABELS = {
    renda_mensal: "Renda mensal",
    score_bureau: "Score de bureau",
    taxa_endividamento: "Taxa de endividamento",
    utilizacao_credito: "Utilização de crédito",
  };
  const CROSS = {
    R10: (v) => v.valor_parcela === null || v.renda_mensal === null || v.valor_parcela <= v.renda_mensal,
    R11: (v) => v.situacao_emprego !== "desempregado" || v.tempo_emprego_meses === 0,
    R12: (v) => v.idade === null || v.idade < 18 || v.idade > 100 || v.tempo_emprego_meses === null
      || v.tempo_emprego_meses <= (v.idade - 16) * 12,
  };

  /* ---------- 1. Validador interativo ---------- */
  const form = document.getElementById("validator-form");
  const enums = payload.rules.find((r) => r.id === "R08").allowed;

  function buildForm() {
    payload.validator_fields.forEach((name) => {
      let input;
      if (name === "situacao_emprego") {
        const options = [...enums.situacao_emprego, "freelancer", "CLT "];
        input = el("select", { id: "f-" + name, name }, options.map((o) =>
          el("option", { value: o, text: enums.situacao_emprego.includes(o) ? o : `"${o}" (fora do domínio)` })));
      } else {
        input = el("input", { id: "f-" + name, name, type: "number", step: "any", inputmode: "decimal" });
      }
      form.append(el("label", { class: "field", for: "f-" + name }, [FIELD_LABELS[name], input]));
    });
    form.addEventListener("input", evaluateForm);
  }

  function readValues() {
    const values = {};
    payload.validator_fields.forEach((name) => {
      const raw = form.elements[name].value;
      if (name === "situacao_emprego") values[name] = raw;
      else values[name] = raw === "" ? null : Number(raw);
    });
    return values;
  }

  function fillValues(values) {
    payload.validator_fields.forEach((name) => {
      const value = values[name];
      if (name === "situacao_emprego" && value !== null && ![...form.elements[name].options].some((o) => o.value === value)) {
        form.elements[name].append(el("option", { value, text: `"${value}" (fora do domínio)` }));
      }
      form.elements[name].value = value === null || value === undefined ? "" : value;
    });
    evaluateForm();
  }

  function inBound(value, bound) {
    const lowOk = bound.include_min === false ? value > bound.min : value >= bound.min;
    return lowOk && (bound.max === null || bound.max === undefined || value <= bound.max);
  }

  function evaluateRule(rule, v) {
    const present = (cols) => cols.filter((c) => Object.prototype.hasOwnProperty.call(v, c));
    if (rule.kind === "schema" || rule.kind === "unique") return { status: "na", note: "avaliada no lote inteiro" };
    if (rule.kind === "notnull") {
      const cols = present(rule.columns);
      return { status: cols.every((c) => v[c] !== null) ? "ok" : "fail" };
    }
    if (rule.kind === "range") {
      const cols = present(Object.keys(rule.bounds)).filter((c) => v[c] !== null);
      if (!cols.length) return { status: "na", note: "campos fora do formulário" };
      return { status: cols.every((c) => inBound(v[c], rule.bounds[c])) ? "ok" : "fail" };
    }
    if (rule.kind === "isin") {
      const cols = present(Object.keys(rule.allowed));
      if (!cols.length) return { status: "na" };
      return { status: cols.every((c) => rule.allowed[c].includes(v[c])) ? "ok" : "fail" };
    }
    return { status: CROSS[rule.id](v) ? "ok" : "fail" };
  }

  function evaluateForm() {
    const values = readValues();
    const list = document.getElementById("rule-list");
    const failed = [];
    list.replaceChildren(...payload.rules.map((rule) => {
      const result = evaluateRule(rule, values);
      if (result.status === "fail") failed.push(rule.id);
      const label = { ok: "passa", fail: "viola", na: "n/a" }[result.status];
      const detail = el("div", {}, [el("div", { text: rule.description }), el("div", { class: "small muted", text: `${rule.dimension} · ${rule.severity}${result.note ? " · " + result.note : ""}` })]);
      return el("li", {}, [el("span", { class: "rule-id", text: rule.id }), detail, el("span", { class: "status " + result.status, text: label })]);
    }));
    const verdict = document.getElementById("validator-verdict");
    verdict.replaceChildren(failed.length
      ? el("span", {}, [el("span", { class: "status fail", text: "Linha rejeitada" }), ` vai para a quarentena por ${failed.join(", ")} (se o lote tiver até 1 % de linhas assim).`])
      : el("span", {}, [el("span", { class: "status ok", text: "Linha aprovada" }), " em todas as regras de linha avaliáveis."]));
  }

  function buildExamples() {
    const box = document.getElementById("example-buttons");
    payload.validator_examples.forEach((example, index) => {
      const button = el("button", { class: "btn" + (index === 0 ? " primary" : ""), type: "button", text: example.label });
      button.addEventListener("click", () => fillValues(example.values));
      box.append(button);
    });
  }

  /* ---------- 2. Circuit breaker ---------- */
  let running = false;

  function setProgress(ratio, state) {
    const bar = document.getElementById("breaker-progress");
    bar.className = "progress" + (state ? " " + state : "");
    bar.firstElementChild.style.width = Math.min(100, ratio * 100) + "%";
    bar.setAttribute("aria-valuenow", String(Math.round(ratio * 100)));
  }

  async function ingest(report) {
    if (running) return;
    running = true;
    const log = document.getElementById("breaker-log");
    const verdict = document.getElementById("breaker-verdict");
    const caption = document.getElementById("breaker-caption");
    log.replaceChildren();
    verdict.hidden = true;
    setProgress(0);
    caption.textContent = `Validando ${fmt.int(report.total_records)} linhas de "${report.batch_name}" contra ${payload.rules.length} regras…`;
    await sleep(400);
    for (const v of report.violations) {
      const structural = v.rule_id === "R01" || v.rule_id === "R02";
      log.append(el("li", { class: structural ? "structural" : "", text: `${v.rule_id} · ${v.column || "regra cruzada"} · ${fmt.int(v.n_failures)} falha(s) · ${v.check.replace(/^R\d{2} · /, "")}` }));
      await sleep(260);
    }
    const blocked = report.decision === "BLOCKED";
    setProgress(blocked ? Math.max(report.rejected_ratio, 0.04) : report.valid_records / report.total_records, blocked ? "fail" : "ok");
    caption.textContent = `${fmt.int(report.rejected_records)} linha(s) rejeitada(s) (${fmt.pct(report.rejected_ratio, 2)}) · ${fmt.int(report.valid_records)} válida(s).`;
    verdict.replaceChildren(blocked
      ? el("span", {}, [el("span", { class: "status fail", text: "BLOCKED" }), " " + report.blocked_reason + ". Nada chega ao modelo; exit code 2."])
      : el("span", {}, [el("span", { class: "status ok", text: "SUCCESS" }), " lote aprovado sem violações; segue para o modelo."]));
    verdict.hidden = false;
    running = false;
  }

  /* ---------- 3. Distribuições da referência ---------- */
  const select = document.getElementById("feature-select");

  function binLabel(feature, low, high) {
    const f = (x) => (feature === "renda_mensal" ? fmt.compact(x) : feature === "score_bureau" ? fmt.int(x) : fmt.num(x, 2));
    return `${f(low)}–${f(high)}`;
  }

  function histogramRows() {
    const feature = select.value;
    const h = payload.histograms[feature];
    return h.counts.map((count, i) => ({ label: binLabel(feature, h.edges[i], h.edges[i + 1]), count, rate: h.default_rate[i] }));
  }

  function setupDistributions() {
    Object.keys(payload.histograms).forEach((name) => select.append(el("option", { value: name, text: FEATURE_LABELS[name] || name })));
    const axisTitle = (text) => ({ display: true, text, color: cssVar("--ink-2") });
    const hist = chart({
      canvasId: "hist-chart",
      tableId: "hist-table",
      table: () => ({ headers: [{ label: "Faixa" }, { label: "Clientes", num: true }], rows: histogramRows().map((r) => [r.label, fmt.int(r.count)]) }),
      config: () => {
        const rows = histogramRows();
        return {
          type: "bar",
          data: { labels: rows.map((r) => r.label), datasets: [{ label: "Clientes", data: rows.map((r) => r.count), backgroundColor: cssVar("--series-1"), borderRadius: 4, borderSkipped: "start", maxBarThickness: 28 }] },
          options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${fmt.int(c.parsed.y)} clientes` } } }, scales: { x: { grid: { display: false }, ticks: { maxRotation: 60, autoSkip: true } }, y: { title: axisTitle("Clientes"), beginAtZero: true } } },
        };
      },
    });
    const rate = chart({
      canvasId: "rate-chart",
      tableId: "rate-table",
      table: () => ({ headers: [{ label: "Faixa" }, { label: "Inadimplência", num: true }], rows: histogramRows().map((r) => [r.label, fmt.pct(r.rate, 1)]) }),
      config: () => {
        const rows = histogramRows();
        return {
          type: "line",
          data: { labels: rows.map((r) => r.label), datasets: [{ label: "Taxa de inadimplência", data: rows.map((r) => (r.rate === null ? null : r.rate * 100)), borderColor: cssVar("--series-2"), backgroundColor: cssVar("--series-2"), borderWidth: 2, pointRadius: 4, pointHoverRadius: 6, spanGaps: true }] },
          options: { interaction: { mode: "index", intersect: false }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `inadimplência ${fmt.num(c.parsed.y, 1)} %` } } }, scales: { x: { grid: { display: false }, ticks: { maxRotation: 60 } }, y: { title: axisTitle("Inadimplência (%)"), beginAtZero: true } } },
        };
      },
    });
    select.addEventListener("change", () => { hist.update(); rate.update(); });
  }

  /* ---------- 4. Limiar do modelo ---------- */
  const H = payload.thresholds.score_histogram;
  const slider = document.getElementById("threshold");
  const totalPos = H.positives.reduce((a, b) => a + b, 0);
  const totalNeg = H.negatives.reduce((a, b) => a + b, 0);
  let scoreChart;

  function confusionAt(index) {
    const tp = H.positives.slice(index).reduce((a, b) => a + b, 0);
    const fp = H.negatives.slice(index).reduce((a, b) => a + b, 0);
    const fn = totalPos - tp;
    const tn = totalNeg - fp;
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = totalPos ? tp / totalPos : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { tp, fp, fn, tn, precision, recall, f1, approval: (tn + fn) / (totalPos + totalNeg), cost: 5 * fn + fp };
  }

  function renderThreshold() {
    const index = Number(slider.value);
    const c = confusionAt(index);
    document.getElementById("threshold-value").textContent = fmt.num(H.edges[index], 2);
    const kpi = (label, value) => el("div", { class: "panel kpi" }, [el("div", { class: "label", text: label }), el("div", { class: "value num", text: value })]);
    document.getElementById("threshold-kpis").replaceChildren(
      kpi("Precisão", fmt.num(c.precision, 3)), kpi("Recall", fmt.num(c.recall, 3)),
      kpi("F1", fmt.num(c.f1, 3)), kpi("Aprovação", fmt.pct(c.approval, 1)));
    const cell = (title, value, note) => el("div", {}, [title, el("strong", { text: fmt.int(value) }), el("span", { class: "small muted", text: note })]);
    document.getElementById("threshold-matrix").replaceChildren(
      el("div", { class: "head" }), el("div", { class: "head", text: "Previsto: bom (aprova)" }), el("div", { class: "head", text: "Previsto: inadimplente (nega)" }),
      el("div", { class: "head", text: "Real: bom" }), cell("VN", c.tn, "aprovado corretamente"), cell("FP", c.fp, "bom pagador negado"),
      el("div", { class: "head", text: "Real: inadimplente" }), cell("FN", c.fn, "inadimplente aprovado"), cell("VP", c.tp, "negado corretamente"));
    if (scoreChart) scoreChart.update();
  }

  function setupThreshold() {
    const nearest = (t) => Math.max(0, Math.min(49, Math.floor(t / 0.02 + 1e-9)));
    slider.value = String(nearest(payload.thresholds.f1_threshold));
    slider.addEventListener("input", renderThreshold);
    document.getElementById("goto-f1").addEventListener("click", () => { slider.value = String(nearest(payload.thresholds.f1_threshold)); renderThreshold(); });
    document.getElementById("goto-cost").addEventListener("click", () => { slider.value = String(nearest(payload.thresholds.cost_threshold)); renderThreshold(); });
    const share = (counts, total) => counts.map((n) => (100 * n) / total);
    const fade = (color, i) => (i < Number(slider.value) ? color + "55" : color);
    scoreChart = chart({
      canvasId: "score-chart",
      tableId: "score-table",
      table: () => ({
        headers: [{ label: "Faixa de score" }, { label: "Bons (%)", num: true }, { label: "Inadimplentes (%)", num: true }],
        rows: H.positives.map((p, i) => [`${fmt.num(H.edges[i], 2)}–${fmt.num(H.edges[i + 1], 2)}`, fmt.num((100 * H.negatives[i]) / totalNeg, 1), fmt.num((100 * p) / totalPos, 1)]),
      }),
      config: () => {
        const c1 = cssVar("--series-1");
        const c2 = cssVar("--series-2");
        return {
          type: "bar",
          data: {
            labels: H.edges.slice(0, -1).map((e) => fmt.num(e, 2)),
            datasets: [
              { label: "Bons pagadores", data: share(H.negatives, totalNeg), backgroundColor: H.negatives.map((_, i) => fade(c1, i)), borderRadius: 3, borderSkipped: "start" },
              { label: "Inadimplentes", data: share(H.positives, totalPos), backgroundColor: H.positives.map((_, i) => fade(c2, i)), borderRadius: 3, borderSkipped: "start" },
            ],
          },
          options: {
            interaction: { mode: "index", intersect: false },
            plugins: { tooltip: { callbacks: { title: (items) => `score ${items[0].label}`, label: (c) => `${c.dataset.label}: ${fmt.num(c.parsed.y, 1)} % da classe` } } },
            scales: { x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 11 } }, y: { title: { display: true, text: "% da classe" }, beginAtZero: true } },
          },
        };
      },
    });
    renderThreshold();
  }

  /* ---------- Importância das features ---------- */
  function setupImportance() {
    const top = payload.importance.slice(0, 10);
    chart({
      canvasId: "importance-chart",
      tableId: "importance-table",
      table: () => ({ headers: [{ label: "Feature" }, { label: "Queda de AUC", num: true }, { label: "Desvio", num: true }], rows: payload.importance.map((i) => [i.feature, fmt.num(i.importance_mean, 4), fmt.num(i.importance_std, 4)]) }),
      config: () => ({
        type: "bar",
        data: { labels: top.map((i) => i.feature), datasets: [{ label: "Queda média de ROC-AUC", data: top.map((i) => i.importance_mean), backgroundColor: cssVar("--series-1"), borderRadius: 4, borderSkipped: "start", maxBarThickness: 22 }] },
        options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `queda de AUC ${fmt.num(c.parsed.x, 4)} ± ${fmt.num(top[c.dataIndex].importance_std, 4)}` } } }, scales: { x: { beginAtZero: true, title: { display: true, text: "Queda de ROC-AUC" } }, y: { grid: { display: false } } } },
      }),
    });
  }

  buildForm();
  buildExamples();
  fillValues(payload.validator_examples[0].values);
  document.getElementById("ingest-bad").addEventListener("click", () => ingest(payload.bad_batch));
  document.getElementById("ingest-ref").addEventListener("click", () => ingest(payload.reference_report));
  setupDistributions();
  setupThreshold();
  setupImportance();
})();
