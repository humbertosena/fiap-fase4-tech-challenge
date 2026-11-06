/* Etapa 4 — fluxo LGPD, inventário, fairness por grupo, grafo causal, intervenções, retenção e roteiro STAR. */
(function () {
  "use strict";

  /* ---------- Funções puras (testáveis em Node) ---------- */
  const AXIS = [0.6, 1.4];
  const BAND = [0.8, 1.25];
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const ratioPos = (ratio) => 100 * clamp((ratio - AXIS[0]) / (AXIS[1] - AXIS[0]), 0, 1);
  const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
  const brNum = (v, d) => v.toFixed(d).replace(".", ",");

  function blockAt(blocks, seconds) {
    const index = blocks.findIndex((b) => seconds >= b.start && seconds < b.end);
    if (index >= 0) return index;
    return seconds >= blocks[blocks.length - 1].end ? blocks.length - 1 : -1;
  }

  function useKind(row) {
    if (row.use === "feature") return "feature";
    return row.use.startsWith("fora") ? "fora" : "outras";
  }

  function retentionState(item, elapsedDays, items) {
    if (item.key === "salt") {
      const kept = items.some((i) => i.personal && i.days !== null && elapsedDays < i.days);
      return { kept, remaining: null };
    }
    if (item.days === null) return { kept: true, remaining: null };
    return { kept: elapsedDays < item.days, remaining: Math.max(0, item.days - elapsedDays) };
  }

  function daysLabel(days) {
    if (days === 0) return "agora (dia 0)";
    const years = days / 365;
    return days < 365 ? `${days} dias` : `${days} dias (cerca de ${brNum(years, 1)} anos)`;
  }

  function edgePath(a, b, size) {
    const [ax, ay] = [a.x + size.w, a.y + size.h / 2];
    if (a.column === b.column) {
      const bend = 46;
      return `M ${ax} ${ay} C ${ax + bend} ${ay} ${ax + bend} ${b.y + size.h / 2} ${ax} ${b.y + size.h / 2}`;
    }
    const [bx, by] = [b.x, b.y + size.h / 2];
    const mid = (ax + bx) / 2;
    return `M ${ax} ${ay} C ${mid} ${ay} ${mid} ${by} ${bx} ${by}`;
  }

  if (typeof window === "undefined") {
    module.exports = { ratioPos, clock, blockAt, useKind, retentionState, daysLabel, edgePath, BAND };
    return;
  }

  /* ---------- Página ---------- */
  const { payload, fmt, el, cssVar } = window.CS;
  const repo = (document.getElementById("steps-box") || {}).dataset ? document.getElementById("steps-box").dataset.repo : "";
  const offline = document.getElementById("offline-note");
  const store = {
    copy(text, button) {
      const done = () => {
        const original = button.textContent;
        button.textContent = "Copiado ✓";
        setTimeout(() => { button.textContent = original; }, 1400);
      };
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => {});
    },
  };

  /* 1. Fluxo LGPD */
  function mountSteps() {
    const box = document.getElementById("stepper");
    const detail = document.getElementById("step-detail");
    const steps = payload.steps;
    const buttons = steps.map((s, i) => el("button", { type: "button", class: `step${s.status === "política" ? " policy" : ""}`, "data-i": String(i) }, [
      el("span", { class: "n", text: String(i + 1) }),
      el("span", { class: "t", text: s.title }),
    ]));
    box.replaceChildren(...buttons);

    function show(i) {
      const s = steps[i];
      buttons.forEach((b, j) => { if (j === i) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current"); });
      const verify = el("code", { text: s.verify });
      const copy = el("button", { type: "button", class: "btn", "aria-label": "Copiar o comando" }, ["Copiar"]);
      copy.addEventListener("click", () => store.copy(s.verify, copy));
      const link = el("a", { href: `${repo}/blob/main/${s.artifact}` }, [el("code", { text: s.artifact })]);
      detail.replaceChildren(
        el("h3", {}, [`${i + 1} · ${s.title} `, el("span", { class: `chip-status ${s.status === "política" ? "politica" : "implementado"}`, text: s.status })]),
        el("p", { text: s.measure }),
        el("dl", { class: "kv" }, [
          el("dt", { text: "Artefato" }), el("dd", {}, [link]),
          el("dt", { text: "Como verificar" }), el("dd", {}, [verify, " ", s.verify.startsWith("uv ") ? copy : ""]),
        ]),
      );
    }

    box.addEventListener("click", (event) => {
      const button = event.target.closest(".step");
      if (button) show(Number(button.dataset.i));
    });
    box.addEventListener("keydown", (event) => {
      const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
      if (!(event.key in keys)) return;
      const current = buttons.findIndex((b) => b.getAttribute("aria-current") === "step");
      const next = clamp(current + keys[event.key], 0, steps.length - 1);
      show(next);
      buttons[next].focus();
      event.preventDefault();
    });
    show(0);
  }

  /* 2. Inventário */
  function mountInventory() {
    const filters = document.getElementById("inventory-filters");
    const body = document.getElementById("inventory-body");
    const note = document.getElementById("inventory-note");
    const options = [["todas", "Todas"], ["feature", "Features do modelo"], ["fora", "Fora do modelo"], ["outras", "Outras"]];
    const buttons = options.map(([key, label]) => el("button", { type: "button", class: "btn", "data-key": key, "aria-pressed": "false", text: label }));
    filters.replaceChildren(...buttons);

    function render(key) {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key)));
      const rows = payload.inventory.filter((r) => key === "todas" || useKind(r) === key);
      body.replaceChildren(...rows.map((r) => el("tr", {}, [
        el("td", {}, [el("code", { text: r.column })]), el("td", { text: r.role }), el("td", { text: r.purpose }), el("td", { text: r.use }),
      ])));
      note.textContent = `${rows.length} de ${payload.inventory.length} colunas.`;
    }
    filters.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (button) render(button.dataset.key);
    });
    render("todas");
  }

  /* 3. Fairness */
  function mountFairness() {
    const audit = payload.fairness;
    const labels = {
      genero: "Gênero", regiao: "Região", faixa_etaria: "Faixa etária",
      champion: "Champion", challenger: "Challenger",
      referencia_validacao: "Validação da referência", mes_3_crise_holdout: "Holdout da crise (dias 16 a 30)",
    };
    const label = (key) => labels[key] || key;
    const unique = (key) => [...new Set(audit.reports.map((r) => r[key]))];
    const selects = {
      dataset: document.getElementById("fair-dataset"),
      role: document.getElementById("fair-role"),
      attribute: document.getElementById("fair-attr"),
    };
    Object.entries(selects).forEach(([key, select]) => {
      select.replaceChildren(...unique(key).map((value) => el("option", { value, text: label(value) })));
      select.addEventListener("change", render);
    });
    selects.dataset.value = "mes_3_crise_holdout";
    selects.attribute.value = "regiao";

    function render() {
      const chosen = Object.fromEntries(Object.entries(selects).map(([k, s]) => [k, s.value]));
      const report = audit.reports.find((r) => r.dataset === chosen.dataset && r.role === chosen.role && r.attribute === chosen.attribute);
      document.getElementById("fair-dataset-note").textContent = `${audit.datasets[chosen.dataset].replace(/`/g, "")}. Limiar de decisão: ${fmt.num(report.threshold, 4)}; n = ${fmt.int(report.n)}.`;
      document.getElementById("fair-list").replaceChildren(...report.groups.map((g) => {
        const isRef = g.group === report.reference_group;
        const out = !g.within_four_fifths;
        const track = el("div", { class: "fair-track" }, [
          el("div", { class: "fair-band", style: `left:${ratioPos(BAND[0])}%;width:${ratioPos(BAND[1]) - ratioPos(BAND[0])}%` }),
          el("div", { class: "fair-one", style: `left:${ratioPos(1)}%` }),
          el("div", { class: "fair-dot", style: `left:${ratioPos(g.disparate_impact)}%` }),
        ]);
        return el("div", { class: `fair-row${out ? " out" : ""}` }, [
          el("span", { class: "label", text: isRef ? `${g.group} (referência)` : g.group }),
          track,
          el("span", { class: "value", text: `${fmt.num(g.disparate_impact, 3)} · aprova ${fmt.pct(g.approval_rate)}` }),
        ]);
      }));
      const worst = report.groups.reduce((a, b) => (Math.abs(b.disparate_impact - 1) > Math.abs(a.disparate_impact - 1) ? b : a));
      const verdict = document.getElementById("fair-verdict");
      verdict.className = `verdict-note ${report.passed ? "ok" : "fail"}`;
      verdict.textContent = report.passed
        ? `${label(chosen.attribute)}: todos os grupos dentro de 4/5. Mais distante de 1,00: ${worst.group} (${fmt.num(worst.disparate_impact, 3)}).`
        : `${label(chosen.attribute)}: há grupo fora de 4/5. Mais distante de 1,00: ${worst.group} (${fmt.num(worst.disparate_impact, 3)}).`;
      document.getElementById("fair-body").replaceChildren(...report.groups.map((g) => el("tr", {}, [
        el("td", {}, [el("code", { text: g.group })]),
        el("td", { class: "num", text: fmt.int(g.n) }), el("td", { class: "num", text: fmt.pct(g.approval_rate) }),
        el("td", { class: "num", text: fmt.num(g.disparate_impact, 3) }),
        el("td", { class: "num", text: fmt.pct(g.tpr) }), el("td", { class: "num", text: fmt.pct(g.fpr) }),
        el("td", { class: "num", text: fmt.pct(g.observed_default_rate) }),
        el("td", { class: "num", text: g.equal_opportunity_diff === null ? "—" : (g.equal_opportunity_diff >= 0 ? "+" : "") + fmt.num(g.equal_opportunity_diff, 3) }),
      ])));
    }
    render();
  }

  /* 4. Grafo causal */
  function mountDag() {
    const dag = payload.dag;
    const SIZE = { w: 210, h: 38 };
    const COLS = [10, 290, 570, 850];
    const heads = ["Narrativa", "Mecanismos do gerador", "Tipo de drift", "O que a operação vê"];
    const kinds = { narrativa: "Narrativa econômica", mecanismo: "Mecanismo do gerador", efeito: "Tipo de drift", consequencia: "O que a operação vê" };
    const windowLabel = { mes_1: "Mês 1 (controle)", mes_2: "Mês 2", mes_3_crise: "Mês 3 (crise)" };
    const svg = document.getElementById("dag");
    const ns = "http://www.w3.org/2000/svg";
    const make = (tag, attrs) => {
      const node = document.createElementNS(ns, tag);
      Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
      return node;
    };
    const place = Object.fromEntries(dag.nodes.map((n) => [n.id, { ...n, x: COLS[n.column], y: 34 + n.row * 58 }]));
    let selected = "CON";
    let windowKey = "mes_3_crise";

    const defs = make("defs");
    const marker = make("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" });
    marker.append(make("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: cssVar("--axis") }));
    defs.append(marker);
    svg.append(defs);
    heads.forEach((text, i) => { const h = make("text", { x: COLS[i], y: 16, class: "dag-head" }); h.textContent = text; svg.append(h); });

    const edgeNodes = dag.edges.map((e) => {
      const path = make("path", { d: edgePath(place[e.source], place[e.target], SIZE), class: `dag-edge${e.dashed ? " dashed" : ""}`, "marker-end": "url(#arrow)" });
      svg.append(path);
      return { edge: e, path };
    });
    const groups = dag.nodes.map((n) => {
      const p = place[n.id];
      const g = make("g", { class: `dag-node ${n.kind}`, tabindex: "0", role: "button", "aria-label": `${kinds[n.kind]}: ${n.label}`, "data-id": n.id });
      g.append(make("rect", { x: p.x, y: p.y, width: SIZE.w, height: SIZE.h, rx: "8" }));
      const text = make("text", { x: p.x + 10, y: p.y + SIZE.h / 2 + 4.5 });
      text.textContent = n.label;
      g.append(text);
      svg.append(g);
      return g;
    });

    const select = document.getElementById("dag-select");
    select.replaceChildren(...Object.keys(kinds).map((kind) => {
      const group = el("optgroup", { label: kinds[kind] });
      dag.nodes.filter((n) => n.kind === kind).forEach((n) => group.append(el("option", { value: n.id, text: n.label })));
      return group;
    }));
    const windowBox = document.getElementById("dag-windows");
    windowBox.replaceChildren(...dag.windows.map((w) => el("button", { type: "button", class: "btn", "data-w": w, "aria-pressed": "false", text: windowLabel[w] || w })));

    function render() {
      const node = place[selected];
      groups.forEach((g) => g.classList.toggle("selected", g.dataset.id === selected));
      edgeNodes.forEach(({ edge, path }) => path.classList.toggle("hot", edge.source === selected || edge.target === selected));
      select.value = selected;
      windowBox.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.w === windowKey)));
      const rows = (node.by_window[windowKey] || []).map((e) => el("li", {}, [el("span", { text: e.label }), el("span", { class: "v", text: e.text })]));
      const fixed = node.static.map((e) => el("li", {}, [el("span", { text: e.label }), el("span", { class: "v", text: e.text })]));
      const parts = [el("span", { class: "kind-chip", text: kinds[node.kind] }), el("h3", { text: node.label }), el("p", { text: node.text })];
      if (rows.length) parts.push(el("h4", { text: `Evidências em ${windowLabel[windowKey] || windowKey}` }), el("ul", { class: "evidence-list" }, rows));
      if (fixed.length) parts.push(el("h4", { text: "Nos dados da intervenção e da faixa 550-700" }), el("ul", { class: "evidence-list" }, fixed));
      if (!rows.length && !fixed.length) parts.push(el("p", { class: "small muted", text: "Sem número direto: é parte da narrativa." }));
      document.getElementById("dag-detail").replaceChildren(...parts);
    }
    svg.addEventListener("click", (event) => {
      const g = event.target.closest(".dag-node");
      if (g) { selected = g.dataset.id; render(); }
    });
    svg.addEventListener("keydown", (event) => {
      const g = event.target.closest(".dag-node");
      if (g && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); selected = g.dataset.id; render(); }
    });
    select.addEventListener("change", () => { selected = select.value; render(); });
    windowBox.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (button) { windowKey = button.dataset.w; render(); }
    });
    render();
  }

  /* 5. Intervenções (barras com tabela equivalente) */
  function mountInterventions() {
    const rows = payload.causal.scenarios.filter((s) => s.key !== "controle");
    const signed = (v) => (v >= 0 ? "+" : "−") + fmt.num(Math.abs(v), 3);
    window.CS.chart({
      canvasId: "interv-canvas",
      tableId: "interv-table",
      config: () => ({
        type: "bar",
        data: {
          labels: rows.map((r) => r.label),
          datasets: [{
            label: "Variação do AUC contra o controle",
            data: rows.map((r) => r.delta_auc),
            backgroundColor: rows.map((r) => (Math.abs(r.delta_auc) >= 0.02 ? cssVar("--critical") : cssVar("--series-1"))),
          }],
        },
        options: {
          indexAxis: "y",
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (ctx) => `${signed(rows[ctx.dataIndex].delta_auc)} ± ${fmt.num(rows[ctx.dataIndex].delta_auc_std, 3)} · taxa base ${fmt.pct(rows[ctx.dataIndex].base_rate, 2)}` } },
          },
          scales: { x: { title: { display: true, text: "Δ AUC (média das sementes)" } } },
        },
      }),
      table: () => ({
        headers: [{ label: "Cenário" }, { label: "Taxa base", num: true }, { label: "AUC", num: true }, { label: "Δ AUC", num: true }],
        rows: rows.map((r) => [r.label, fmt.pct(r.base_rate, 2), fmt.num(r.auc, 3), `${signed(r.delta_auc)} ± ${fmt.num(r.delta_auc_std, 3)}`]),
      }),
    });
  }

  /* 6. Retenção */
  function mountRetention() {
    const items = payload.retention;
    const slider = document.getElementById("ret-slider");
    const grid = document.getElementById("ret-grid");
    const presets = document.getElementById("ret-presets");
    const options = [[0, "Hoje"], [30, "30 dias"], [90, "90 dias"], [1826, "5 anos"], [2190, "6 anos"]];
    presets.replaceChildren(...options.map(([days, text]) => el("button", { type: "button", class: "btn", "data-days": String(days), text })));

    function render() {
      const days = Number(slider.value);
      document.getElementById("ret-label").textContent = daysLabel(days);
      const states = items.map((item) => [item, retentionState(item, days, items)]);
      grid.replaceChildren(...states.map(([item, state]) => el("article", { class: `ret-card ${state.kept ? "kept" : "gone"}` }, [
        el("h3", { text: item.label }),
        el("span", { class: "where", text: item.where }),
        el("div", {}, [el("span", { class: "badge-state", text: state.kept ? "retido" : `descartado: ${item.after}` })]),
        el("p", { text: `Prazo-alvo: ${item.until}.` }),
        el("p", { text: state.kept && state.remaining !== null ? `Faltam ${fmt.int(state.remaining)} dias. Base: ${item.basis}.` : `Base: ${item.basis}.` }),
      ])));
      const personal = states.filter(([item]) => item.personal);
      const kept = personal.filter(([, state]) => state.kept).length;
      document.getElementById("ret-summary").textContent = `${kept} de ${personal.length} tipos de dado pessoal ainda retidos. ${kept === 0 ? "O salt também já pode ser destruído." : ""}`;
    }
    slider.addEventListener("input", render);
    presets.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (button) { slider.value = button.dataset.days; render(); }
    });
    render();
  }

  /* 7. Roteiro STAR com cronômetro */
  function mountStar() {
    const star = payload.star;
    const host = document.getElementById("star-blocks");
    const clockNode = document.getElementById("star-clock");
    const bar = document.getElementById("star-progress");
    const nowNode = document.getElementById("star-now");
    const startButton = document.getElementById("star-start");
    const nodes = star.blocks.map((b) => {
      const details = el("details", { class: "star-block" }, [
        el("summary", {}, [
          el("span", { class: "time", text: `${clock(b.start)}–${clock(b.end)}` }),
          el("span", { class: "name", text: b.name }),
          el("span", { class: "small muted", text: `${b.duration} s` }),
        ]),
        el("div", { class: "body" }, [
          el("div", {}, [el("strong", { text: "Tela: " }), b.screen]),
          el("div", {}, [el("strong", { text: "Artefato: " }), ...b.artifacts.flatMap((a, i) => [i ? ", " : "", el("code", { text: a })])]),
          el("blockquote", { class: "speech", text: b.speech || "—" }),
        ]),
      ]);
      host.append(details);
      return details;
    });
    let elapsed = 0;
    let startedAt = null;
    let timer = null;

    function paint() {
      clockNode.textContent = clock(Math.min(elapsed, star.total_seconds));
      bar.style.width = `${clamp(100 * elapsed / star.total_seconds, 0, 100)}%`;
      const index = blockAt(star.blocks, elapsed);
      nodes.forEach((n, i) => {
        n.classList.toggle("now", i === index && elapsed < star.total_seconds);
        n.classList.toggle("done", index >= 0 && i < index);
      });
      if (index >= 0 && elapsed < star.total_seconds) {
        nodes[index].open = true;
        nowNode.textContent = `Agora: ${star.blocks[index].name}`;
      } else {
        nowNode.textContent = elapsed >= star.total_seconds ? "Fim dos 5 minutos." : "";
      }
    }
    function stop() { clearInterval(timer); timer = null; startedAt = null; startButton.textContent = elapsed > 0 && elapsed < star.total_seconds ? "Continuar" : "Iniciar"; }
    function tick() {
      elapsed = (Date.now() - startedAt) / 1000;
      if (elapsed >= star.total_seconds) { elapsed = star.total_seconds; stop(); startButton.textContent = "Iniciar"; }
      paint();
    }
    startButton.addEventListener("click", () => {
      if (timer) { stop(); return; }
      if (elapsed >= star.total_seconds) elapsed = 0;
      startedAt = Date.now() - elapsed * 1000;
      timer = setInterval(tick, 250);
      startButton.textContent = "Pausar";
    });
    document.getElementById("star-reset").addEventListener("click", () => { stop(); elapsed = 0; startButton.textContent = "Iniciar"; paint(); });
    paint();
  }

  if (offline) offline.hidden = !!window.Chart;
  mountSteps();
  mountInventory();
  mountFairness();
  mountDag();
  mountInterventions();
  mountRetention();
  mountStar();
})();
