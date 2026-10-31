/* Etapa 3 — replay do pipeline, gauges de error budget, feed de alertas, registry e mascaramento de PII. */
(function () {
  "use strict";
  const { payload, fmt, el, sleep, reducedMotion } = window.CS;
  const audit = payload.audit;
  const STAGE_LABEL = {
    ingestion: "Ingestão", contract_validation: "Contrato", inference: "Inferência",
    drift_detection: "Drift", slo_evaluation: "SLOs", tracking: "Tracking",
  };
  const STATE_LABEL = { pending: "pendente", started: "rodando", success: "sucesso", failed: "falhou", blocked: "bloqueado", skipped: "pulado" };
  const STATUS_TEXT = { OK: "dentro do limite", WARN: "atenção", BREACHED: "estourado" };
  const window_ = (name) => audit.windows.find((w) => w.window === name);
  const offline = document.getElementById("offline-note");
  if (offline) offline.hidden = true; /* esta página não depende do Chart.js */

  /* ---------- 1. Replay do pipeline ---------- */
  const selectReplay = document.getElementById("replay-window");
  const stagesBox = document.getElementById("replay-stages");
  const consoleBox = document.getElementById("replay-console");
  const verdict = document.getElementById("replay-verdict");
  const runButtons = ["replay-run", "replay-all", "replay-skip"].map((id) => document.getElementById(id));
  let running = false;

  function resetStages() {
    stagesBox.replaceChildren(...payload.stages.map((stage) => el("div", { class: "stage st-pending", "data-stage": stage }, [
      el("span", { class: "name", text: STAGE_LABEL[stage] }),
      el("span", { class: "state", text: STATE_LABEL.pending }),
      el("span", { class: "ms", text: "—" }),
    ])));
  }

  function paintStage(event) {
    const node = stagesBox.querySelector(`[data-stage="${event.stage}"]`);
    if (!node) return;
    const state = event.status.toLowerCase();
    node.className = `stage st-${state}`;
    node.querySelector(".state").textContent = STATE_LABEL[state] || state;
    node.querySelector(".ms").textContent = state === "started" ? "…" : state === "skipped" ? "—" : `${fmt.int(event.latency_ms)} ms`;
  }

  function logLine(event) {
    const detail = event.message ? ` · ${event.message}` : "";
    const text = `${event.stage} · ${event.status} · ${fmt.int(event.records_processed)} registros${event.rejected_records ? `, ${fmt.int(event.rejected_records)} rejeitados` : ""}${detail}`;
    const line = el("li", { class: `st-${event.status.toLowerCase()}`, title: JSON.stringify(event) }, [text]);
    consoleBox.append(line);
    consoleBox.scrollTop = consoleBox.scrollHeight;
  }

  function showVerdict(name) {
    const w = window_(name);
    const alerts = w.slo.alerts.map((a) => a.severity + (a.slo_id ? ` (${a.slo_id})` : "")).join(", ");
    verdict.className = `verdict ${w.decision === "BLOCKED" ? "blocked" : "ok"}`;
    verdict.textContent = w.decision === "BLOCKED"
      ? `${name}: BLOCKED. O contrato bloqueou o lote; inferência e drift foram pulados. Alerta: ${alerts}.`
      : `${name}: SUCCESS. Alertas: ${alerts}. Saldo do orçamento: ${fmt.pct(w.slo.remaining_budget, 0)}.`;
  }

  async function replay(name, instant) {
    resetStages();
    consoleBox.replaceChildren();
    verdict.className = "verdict";
    verdict.textContent = `Executando ${name}…`;
    for (const event of payload.events.filter((e) => e.window === name)) {
      paintStage(event);
      logLine(event);
      if (!instant && event.status !== "SKIPPED") await sleep(160 + Math.min(event.latency_ms, 900) / 3);
    }
    showVerdict(name);
  }

  async function guarded(task) {
    if (running) return;
    running = true;
    runButtons.forEach((b) => { b.disabled = true; });
    try { await task(); } finally {
      running = false;
      runButtons.forEach((b) => { b.disabled = false; });
    }
  }

  document.getElementById("replay-run").addEventListener("click", () => guarded(() => replay(selectReplay.value, false)));
  document.getElementById("replay-skip").addEventListener("click", () => guarded(() => replay(selectReplay.value, true)));
  document.getElementById("replay-all").addEventListener("click", () => guarded(async () => {
    for (const w of audit.windows) {
      selectReplay.value = w.window;
      await replay(w.window, false);
      await sleep(400);
    }
  }));
  resetStages();

  /* ---------- 2. Gauges de error budget ---------- */
  const gaugeSelect = document.getElementById("gauge-window");
  const gaugeBox = document.getElementById("gauges");
  const SCALE = 2;
  const limitText = (spec) => `${spec.objective.op} ${fmt.num(spec.objective.value, spec.objective.value < 10 ? 2 : 0)}`;

  function gaugeCard(spec, result) {
    const card = el("div", { class: "gauge-card" }, [
      el("h4", { text: spec.title || spec.id }),
      el("span", { class: "limit-text", text: `${spec.id} · limite ${limitText(spec)}` }),
    ]);
    if (!result) {
      card.append(el("p", { class: "small muted", text: "não avaliado: depende de dados que o lote bloqueado não produziu." }));
      return card;
    }
    const fill = el("span", { class: "fill" });
    fill.style.width = `${Math.min(result.consumption / SCALE, 1) * 100}%`;
    const bar = el("div", { class: `gauge ${result.status.toLowerCase()}`, role: "img",
      "aria-label": `${spec.title}: ${fmt.int(100 * result.consumption)} % do orçamento, ${STATUS_TEXT[result.status]}` }, [fill, el("span", { class: "mark" })]);
    card.append(bar, el("span", { class: "gauge-text", text: `${fmt.int(100 * result.consumption)} % do orçamento · ${STATUS_TEXT[result.status]} · observado ${fmt.num(result.observed, result.observed < 10 ? 3 : 0)}` }));
    return card;
  }

  function renderGauges() {
    const w = window_(gaugeSelect.value);
    gaugeBox.replaceChildren(...payload.slo.slos.map((spec) => gaugeCard(spec, w.slo.results.find((r) => r.slo_id === spec.id))));
    document.getElementById("gauge-balance").textContent = fmt.pct(w.slo.remaining_budget, 0);
  }
  gaugeSelect.addEventListener("change", renderGauges);
  renderGauges();

  /* ---------- 3. Feed de alertas ---------- */
  const feed = document.getElementById("alert-feed");
  const filters = document.getElementById("alert-filters");
  const SEV_TEXT = { P1: "P1 · lote bloqueado", P2: "P2 · drift crítico", P3: "P3 · atenção", P4: "P4 · informativo" };
  let severity = "todos";

  function renderFeed() {
    const shown = audit.alerts.filter((a) => severity === "todos" || a.severity === severity);
    feed.replaceChildren(...shown.map((a) => el("li", { "data-severity": a.severity }, [
      el("div", { class: "head" }, [
        el("span", { class: `sev ${a.severity}`, text: SEV_TEXT[a.severity] }),
        el("span", { class: "where", text: a.window + (a.slo_id ? ` · ${a.slo_id}` : "") }),
      ]),
      el("div", { text: a.message }),
      el("div", { class: "action" }, [`${a.recommended_action} `, el("a", { href: `${feed.dataset.runbook}#${a.runbook_anchor}`, text: "Runbook" })]),
    ])));
    filters.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.severity === severity)));
  }
  ["todos", "P1", "P2", "P3", "P4"].forEach((value) => {
    const count = value === "todos" ? audit.alerts.length : audit.alerts.filter((a) => a.severity === value).length;
    const label = value === "todos" ? "Todos" : SEV_TEXT[value];
    const button = el("button", { class: "btn", type: "button", "data-severity": value, "aria-pressed": String(value === severity), text: `${label} (${count})` });
    button.addEventListener("click", () => { severity = value; renderFeed(); });
    filters.append(button);
  });
  renderFeed();

  /* ---------- 4. Registry (antes e depois) e galeria ---------- */
  const toggle = document.getElementById("alias-toggle");
  const aliasTable = document.getElementById("alias-table");
  let state = 0;

  function renderRegistry() {
    const current = payload.registry[state];
    aliasTable.replaceChildren(...current.versions.map((v) => el("tr", {}, [
      el("td", { class: "num", text: `v${v.version}` }),
      el("td", { text: v.model }),
      el("td", {}, v.aliases.length ? v.aliases.map((a) => el("span", { class: `chip ${a}`, text: `@${a}` })) : [el("span", { class: "muted", text: "sem alias" })]),
      el("td", {}, v.tag ? [el("code", { text: v.tag })] : [el("span", { class: "muted", text: "—" })]),
    ])));
    document.getElementById("alias-note").textContent = current.note;
    toggle.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(i === state)));
  }
  payload.registry.forEach((entry, index) => {
    const button = el("button", { class: "btn", type: "button", "aria-pressed": String(index === state), text: entry.label });
    button.addEventListener("click", () => { state = index; renderRegistry(); });
    toggle.append(button);
  });
  renderRegistry();

  document.querySelectorAll("#gallery .shot").forEach((figure) => {
    const button = figure.querySelector(".shot-btn");
    button.addEventListener("click", () => {
      const zoomed = figure.classList.toggle("zoomed");
      button.setAttribute("aria-expanded", String(zoomed));
    });
  });

  /* ---------- 5. Mascaramento de PII ---------- */
  const input = document.getElementById("pii-input");
  const output = document.getElementById("pii-output");
  const count = document.getElementById("pii-count");
  const rules = payload.pii.patterns.map((p) => new RegExp(p.source, "g"));
  const example = payload.pii.fragments.join("");

  function redact() {
    let total = 0;
    let text = input.value;
    rules.forEach((rule) => { text = text.replace(rule, () => { total += 1; return "***"; }); });
    output.textContent = text;
    count.textContent = total ? `${total} trecho(s) mascarado(s) com ***` : "Nenhum dado pessoal encontrado.";
  }
  input.value = example;
  input.addEventListener("input", redact);
  document.getElementById("pii-reset").addEventListener("click", () => { input.value = example; redact(); });
  redact();
  void reducedMotion;
})();
