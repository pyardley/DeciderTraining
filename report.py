"""Write a standalone HTML exhibit from results/compare.json.

compare.py calls this after it scores both models. To rebuild the page
from a results file that is already on disk:

    python report.py
"""

import argparse
import json
from pathlib import Path

RESULTS_PATH = Path("results") / "compare.json"
HTML_PATH = Path("results") / "compare.html"


def _series(block: dict, label: str) -> dict:
    overall = block["overall"]
    timing = block.get("timing") or {}
    return {
        "label": label,
        "accuracy": overall["accuracy"],
        "ece": overall["ece"],
        "record_accuracy": overall["record_accuracy"],
        "n": overall["n"],
        "median_seconds": timing.get("median_seconds"),
        "mean_seconds": timing.get("mean_seconds"),
        "by_type": {
            name: {"accuracy": row["accuracy"], "n": row["n"]}
            for name, row in block.get("by_type", {}).items()
        },
        "by_workflow": {
            name: {"accuracy": row["accuracy"], "n": row["n"]}
            for name, row in block.get("by_workflow", {}).items()
        },
        "decisions": block.get("decisions") or [],
    }


def report_gaps(results: dict) -> list:
    """Fields the standalone page reads. An empty list means the page can be filled in."""
    gaps = []
    catalogs = []
    for name in ("untrained", "trained"):
        block = results.get(name)
        if not isinstance(block, dict):
            gaps.append(f"{name} scores are missing")
            continue
        overall = block.get("overall") or {}
        for key in ("accuracy", "ece", "record_accuracy", "n"):
            if key not in overall:
                gaps.append(f"{name} overall has no {key}")
        timing = block.get("timing") or {}
        for key in ("median_seconds", "mean_seconds"):
            if timing.get(key) is None:
                gaps.append(f"{name} timing has no {key}")
        for group in ("by_type", "by_workflow"):
            rows = block.get(group) or {}
            if not rows:
                gaps.append(f"{name} has no {group}")
            for label, row in rows.items():
                if "accuracy" not in row or "n" not in row:
                    gaps.append(f"{name} {group} {label} is missing accuracy or n")
        decisions = block.get("decisions") or []
        if not decisions:
            gaps.append(f"{name} has no per-question decisions")
        needed = ("question", "pick", "gold", "confidence", "correct", "qtype", "workflow")
        missing = sum(1 for point in decisions if any(point.get(key) is None for key in needed))
        if missing:
            gaps.append(f"{name} is missing the chosen option, confidence, or question type on {missing} decisions")
        catalog = block.get("catalog") or []
        if not catalog:
            gaps.append(f"{name} has no question catalog for the radial chart")
        else:
            index = {(row["workflow"], row["question"]): row for row in catalog}
            for point in decisions:
                if any(point.get(key) is None for key in ("question", "pick", "gold", "workflow")):
                    continue
                spec = index.get((point.get("workflow"), point.get("question")))
                if spec is None:
                    gaps.append(f"{name} catalog has no options for {point.get('workflow')} {point.get('question')}")
                    break
                count = len(spec.get("options") or [])
                pick, gold = point.get("pick"), point.get("gold")
                if not (isinstance(pick, int) and isinstance(gold, int) and 0 <= pick < count and 0 <= gold < count):
                    gaps.append(f"{name} chose an option outside the list for {point.get('question')}")
                    break
            catalogs.append({
                (row["workflow"], row["question"]): tuple(row["options"])
                for row in catalog
            })
    if len(catalogs) == 2 and catalogs[0] != catalogs[1]:
        gaps.append("the untrained and trained question catalogs do not match")
    if not results.get("cases"):
        gaps.append("the case count is missing")
    return gaps


def write_report(results: dict, path: Path = HTML_PATH) -> Path:
    payload = {
        "dataset": results.get("dataset", ""),
        "split": results.get("split", "test"),
        "cases": results.get("cases"),
        "untrained": _series(results["untrained"], "Untrained"),
        "trained": _series(results["trained"], "Trained"),
        "questions": (
            results["untrained"].get("catalog")
            or results["trained"].get("catalog")
            or []
        ),
    }
    data = json.dumps(payload, separators = (",", ":")).replace("<", "\\u003c")
    path = Path(path)
    path.parent.mkdir(exist_ok = True)
    path.write_text(_PAGE.replace("/*__DATA__*/", data), encoding = "utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description = "Build the standalone results page.")
    parser.add_argument("results", nargs = "?", default = str(RESULTS_PATH))
    parser.add_argument("-o", "--output", default = str(HTML_PATH))
    args = parser.parse_args()
    source = Path(args.results)
    if not source.is_file():
        raise SystemExit(f"{source} is not written yet. compare.py creates it after both models are scored.")
    destination = write_report(json.loads(source.read_text(encoding = "utf-8")), Path(args.output))
    print(f"Wrote {destination}")


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decider 0.8B, trained against untrained</title>
<style>
  :root {
    --bg: #101412;
    --panel: #1a211c;
    --ink: #e7f0e8;
    --muted: #93a498;
    --faint: #6e7f74;
    --line: #2a3830;
    --teal: #3ddec4;
    --red: #f07167;
    --untrained: #8aa0b4;
    --trained: #e4c16a;
    --gate: #f3d36b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font: 16px/1.5 "Segoe UI", sans-serif;
  }
  main { max-width: 980px; margin: 0 auto; padding: 40px 24px 72px; }
  h1, h2 { font-family: Georgia, "Palatino Linotype", serif; font-weight: 500; }
  h1 { font-size: 34px; line-height: 1.15; margin: 0 0 8px; }
  h2 { font-size: 22px; margin: 0 0 8px; }
  p { margin: 0 0 14px; }
  .kicker { color: var(--trained); letter-spacing: 0.08em; text-transform: uppercase; font-size: 12px; margin-bottom: 10px; }
  .sub { color: var(--muted); max-width: 68ch; }
  section { margin-top: 36px; }
  .hero { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 28px; }
  .stat {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 18px 20px 16px;
  }
  .stat .who { color: var(--muted); font-size: 13px; }
  .stat .num { font-family: Georgia, serif; font-size: 48px; line-height: 1.1; margin: 4px 0; }
  .untrained .num { color: var(--untrained); }
  .trained .num { color: var(--trained); }
  .stat .detail { color: var(--muted); font-size: 14px; }
  .delta { grid-column: 1 / -1; margin: 0; color: var(--ink); }
  .filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }
  button {
    background: transparent;
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 6px 12px;
    font: inherit;
    cursor: pointer;
  }
  button[aria-pressed="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink); }
  .legend { display: flex; gap: 18px; color: var(--muted); font-size: 14px; margin-bottom: 8px; }
  .swatch { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
  .swatch.ok { background: var(--teal); }
  .swatch.bad { background: var(--red); }
  .strip {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 12px 12px 4px;
    margin-bottom: 12px;
    position: relative;
  }
  .strip h3 { margin: 0 8px 4px; font-size: 15px; font-weight: 600; }
  canvas { width: 100%; height: 168px; display: block; cursor: crosshair; }
  .tip {
    position: fixed;
    z-index: 5;
    pointer-events: none;
    background: #0c0f0d;
    border: 1px solid var(--line);
    color: var(--ink);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 13px;
    display: none;
  }
  .gate-row { display: flex; align-items: center; gap: 12px; margin: 8px 0 14px; }
  input[type="range"] { flex: 1; accent-color: var(--gate); }
  output { min-width: 3.2em; color: var(--gate); font-variant-numeric: tabular-nums; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; font-size: 13px; }
  .bar-row { display: grid; grid-template-columns: 190px 1fr; gap: 12px; align-items: center; margin: 8px 0; }
  .bar-label { color: var(--muted); font-size: 14px; }
  .tracks { display: grid; gap: 4px; }
  .track { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; }
  .fill { height: 8px; border-radius: 99px; background: var(--line); }
  .fill > span { display: block; height: 100%; border-radius: 99px; }
  .fill.u > span { background: var(--untrained); }
  .fill.t > span { background: var(--trained); }
  .pct { font-size: 13px; color: var(--muted); min-width: 3.4em; }
  .reliability { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 12px; }
  .reliability svg { width: 100%; height: auto; }
  .ring-stage {
    background: radial-gradient(circle at 50% 42%, #15203a 0%, #070b16 62%);
    border: 1px solid #1c2744;
    border-radius: 18px;
    padding: 16px 16px 12px;
  }
  .ring-controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .ring-controls select {
    background: #0d1426;
    color: #e7eef8;
    border: 1px solid #2a3a5c;
    border-radius: 999px;
    padding: 6px 12px;
    font: inherit;
  }
  .ring-frame { position: relative; max-width: 720px; margin: 8px auto 0; }
  .ring-frame svg { width: 100%; height: auto; display: block; }
  .ring-center {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -52%);
    width: 32%;
    text-align: center;
    pointer-events: none;
    color: #f4f7fb;
  }
  .ring-score { font-family: Georgia, "Palatino Linotype", serif; font-size: clamp(52px, 8vw, 88px); line-height: 0.85; letter-spacing: -0.03em; }
  .ring-score .frac { color: #e4b15a; font-size: 0.42em; letter-spacing: 0; }
  .ring-caption { margin-top: 10px; color: #c5d0dc; font-size: 14px; }
  .ring-legend { color: #9aabc0; font-size: 13px; margin: 4px 8px 0; }
  .note { color: var(--faint); font-size: 13px; max-width: 78ch; }
  footer { margin-top: 40px; color: var(--faint); font-size: 13px; max-width: 78ch; }
  @media (max-width: 700px) {
    .hero, .bar-row { grid-template-columns: 1fr; }
    h1 { font-size: 28px; }
  }
</style>
</head>
<body>
<main>
  <p class="kicker">Decider 0.8B</p>
  <h1>Trained against the same model before training</h1>
  <p class="sub" id="intro"></p>
  <div class="hero" id="hero"></div>

  <section>
    <h2>Where the answers land</h2>
    <p class="sub">Each dot is one test case, drawn at the option the model chose. Teal means that option is the gold label. Red means it chose something else. A model that has not learned the question piles onto one or two options.</p>
    <div class="ring-stage" id="ring-stage">
      <div class="ring-controls">
        <button type="button" id="ring-untrained" aria-pressed="true">Untrained</button>
        <button type="button" id="ring-trained" aria-pressed="false">Trained</button>
        <select id="ring-workflow" aria-label="Workflow"></select>
        <select id="ring-question" aria-label="Question"></select>
      </div>
      <div class="ring-frame">
        <svg id="ring" viewBox="0 0 760 760" role="img" aria-label="Test cases arranged by the option the model chose"></svg>
        <div class="ring-center" id="ring-center"></div>
      </div>
      <p class="ring-legend"><i class="swatch ok"></i>The option the gold label names <i class="swatch bad"></i>Another option</p>
    </div>
  </section>

  <section>
    <h2>Each question, by confidence</h2>
    <p class="sub">One dot is one test question. Teal means the top option matched the gold label. Red means it did not. Dots sit further right as the model assigns more probability to its top option. The line is the gate: at or above it, an application would act on its own.</p>
    <div class="filters" id="filters"></div>
    <div class="legend"><span><i class="swatch ok"></i>Matched gold</span><span><i class="swatch bad"></i>Did not</span></div>
    <div id="strips"></div>
    <div class="gate-row">
      <label for="gate">Gate</label>
      <input id="gate" type="range" min="50" max="99" value="90">
      <output id="gate-out" for="gate">90%</output>
    </div>
    <table id="gate-table"></table>
    <p class="note" id="gate-note"></p>
  </section>

  <section>
    <h2>Accuracy</h2>
    <p class="sub">Share of questions whose top option matches the gold label. These bars are the full test split. The filter above changes the dots and the gate, not these bars.</p>
    <div id="bars"></div>
  </section>

  <section>
    <h2>Does the confidence match the hit rate?</h2>
    <p class="sub">Each point is a band of stated confidence. It should land on the diagonal: questions the model calls at 80% ought to be right about 80% of the time. Points below the line are overconfident.</p>
    <div class="reliability" id="reliability"></div>
  </section>

  <section>
    <h2>Time for one case</h2>
    <p class="sub" id="timing"></p>
  </section>
  <footer id="foot"></footer>
</main>
<div class="tip" id="tip"></div>
<script>const DATA = /*__DATA__*/;</script>
<script>
const untrained = DATA.untrained;
const trained = DATA.trained;
const filters = ["all", "choice", "score", "noul"];
let active = "all";

const pct = (value) => `${(value * 100).toFixed(1)}%`;
const pretty = (name) => name.replaceAll("_", " ");

function selected(model) {
  if (active === "all") return model.decisions;
  return model.decisions.filter((row) => row.qtype === active);
}

function renderIntro() {
  const n = untrained.n.toLocaleString();
  document.getElementById("intro").textContent =
    `${DATA.cases.toLocaleString()} held-out cases from ${DATA.dataset}, ${n} questions. ` +
    `The trained model is the same checkpoint after this fine-tune.`;
  const change = trained.accuracy - untrained.accuracy;
  const sign = change >= 0 ? "+" : "";
  document.getElementById("hero").innerHTML = [untrained, trained].map((model) => `
    <article class="stat ${model.label.toLowerCase()}">
      <div class="who">${model.label}</div>
      <div class="num">${pct(model.accuracy)}</div>
      <div class="detail">${model.n.toLocaleString()} questions · calibration error ${model.ece.toFixed(3)} · whole case ${pct(model.record_accuracy)}</div>
    </article>`).join("") + `<p class="delta">${sign}${(change * 100).toFixed(1)} points of accuracy on the full test split.</p>`;
}

function renderFilters() {
  document.getElementById("filters").innerHTML = filters.map((name) =>
    `<button type="button" data-filter="${name}" aria-pressed="${name === active}">${name === "all" ? "All questions" : name}</button>`
  ).join("");
}

function gateOf(decisions, gate) {
  let review = 0, automatic = 0, wrong = 0;
  for (const row of decisions) {
    if (row.confidence + 1e-12 >= gate) {
      automatic += 1;
      if (!row.correct) wrong += 1;
    } else {
      review += 1;
    }
  }
  return { review, automatic, wrong };
}

function renderGate() {
  const gate = Number(document.getElementById("gate").value) / 100;
  document.getElementById("gate-out").textContent = `${Math.round(gate * 100)}%`;
  const rows = [untrained, trained].map((model) => ({ model, ...gateOf(selected(model), gate) }));
  const cells = (pick) => rows.map((row) => `<td>${pick(row).toLocaleString()}</td>`).join("");
  const share = (row) => row.automatic ? `${((row.wrong / row.automatic) * 100).toFixed(1)}%` : "—";
  document.getElementById("gate-table").innerHTML = `
    <thead><tr><th></th><th>Untrained</th><th>Trained</th></tr></thead>
    <tbody>
      <tr><td>Sent for review</td>${cells((row) => row.review)}</tr>
      <tr><td>Automatic</td>${cells((row) => row.automatic)}</tr>
      <tr><td>Incorrect among the automatic</td>${cells((row) => row.wrong)}</tr>
      <tr><td>Incorrect share of the automatic</td><td>${share(rows[0])}</td><td>${share(rows[1])}</td></tr>
    </tbody>`;
  const scope = active === "all" ? "all questions" : `${active} questions`;
  document.getElementById("gate-note").textContent =
    `At a ${Math.round(gate * 100)}% gate, on ${scope}. The wrong counts are wrong among the decisions let through, not among every question. ` +
    `Training changes which option wins as well as the probability, so this is not a pure temperature comparison.`;
  drawAll(gate);
}

function jitter(index) {
  const raw = Math.sin(index * 12.9898) * 43758.5453;
  return (raw - Math.floor(raw)) * 2 - 1;
}

const hitTargets = new Map();

function drawStrip(canvas, model, gate) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 640;
  const height = 168;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const padL = 36, padR = 16, padT = 16, padB = 28;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  ctx.strokeStyle = "#2a3830";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.fillStyle = "#93a498";
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  for (const tick of [0, 0.5, 1]) {
    const x = padL + tick * plotW;
    ctx.fillText(`${Math.round(tick * 100)}%`, x, height - 8);
  }
  const gx = padL + gate * plotW;
  ctx.strokeStyle = "#f3d36b";
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(gx, padT);
  ctx.lineTo(gx, padT + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  const decisions = selected(model);
  const hits = [];
  const order = decisions.map((row, index) => ({ row, index }))
    .sort((a, b) => Number(a.row.correct) - Number(b.row.correct));
  for (const item of order) {
    const x = padL + Math.min(Math.max(item.row.confidence, 0), 1) * plotW;
    const y = padT + plotH / 2 + jitter(item.index) * (plotH * 0.42);
    ctx.beginPath();
    ctx.fillStyle = item.row.correct ? "#3ddec4" : "#f07167";
    ctx.globalAlpha = 0.85;
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fill();
    hits.push({ x, y, row: item.row });
  }
  ctx.globalAlpha = 1;
  hitTargets.set(canvas, hits);
}

function drawAll(gate) {
  for (const canvas of document.querySelectorAll("canvas[data-model]")) {
    const model = canvas.dataset.model === "trained" ? trained : untrained;
    drawStrip(canvas, model, gate);
  }
  drawReliability();
}

function mountStrips() {
  document.getElementById("strips").innerHTML = [untrained, trained].map((model) => `
    <article class="strip">
      <h3>${model.label}</h3>
      <canvas data-model="${model.label.toLowerCase()}"></canvas>
    </article>`).join("");
}

function renderBars() {
  const names = ["overall", ...Object.keys(untrained.by_type), ...Object.keys(untrained.by_workflow)];
  const value = (model, name) => {
    if (name === "overall") return model.accuracy;
    return (model.by_type[name] || model.by_workflow[name] || {}).accuracy ?? 0;
  };
  document.getElementById("bars").innerHTML = names.map((name) => {
    const left = value(untrained, name);
    const right = value(trained, name);
    const bar = (klass, amount) => `<div class="track"><div class="fill ${klass}"><span style="width:${Math.max(0, Math.min(1, amount)) * 100}%"></span></div><span class="pct">${pct(amount)}</span></div>`;
    return `<div class="bar-row"><div class="bar-label">${pretty(name)}</div><div class="tracks">${bar("u", left)}${bar("t", right)}</div></div>`;
  }).join("");
}

function binsFor(model) {
  const buckets = Array.from({ length: 10 }, () => ({ n: 0, conf: 0, hit: 0 }));
  for (const row of selected(model)) {
    const index = Math.min(9, Math.floor(row.confidence * 10));
    const bucket = buckets[index];
    bucket.n += 1;
    bucket.conf += row.confidence;
    bucket.hit += row.correct ? 1 : 0;
  }
  return buckets.filter((bucket) => bucket.n > 0).map((bucket) => ({
    n: bucket.n,
    conf: bucket.conf / bucket.n,
    hit: bucket.hit / bucket.n,
  }));
}

function drawReliability() {
  const w = 640, h = 360, padL = 56, padB = 48, padT = 20, padR = 18;
  const xOf = (value) => padL + value * (w - padL - padR);
  const yOf = (value) => (h - padB) - value * (h - padB - padT);
  const series = [
    [untrained, "#8aa0b4"],
    [trained, "#e4c16a"],
  ];
  let marks = "";
  for (const [model, color] of series) {
    const points = binsFor(model).map((bin) => `${xOf(bin.conf).toFixed(1)},${yOf(bin.hit).toFixed(1)}`);
    if (points.length > 1) {
      marks += `<polyline fill="none" stroke="${color}" stroke-width="2" points="${points.join(" ")}"></polyline>`;
    }
    for (const bin of binsFor(model)) {
      const radius = 3 + Math.sqrt(bin.n) * 0.45;
      marks += `<circle cx="${xOf(bin.conf).toFixed(1)}" cy="${yOf(bin.hit).toFixed(1)}" r="${radius.toFixed(1)}" fill="${color}"></circle>`;
    }
  }
  const ticks = [0, 0.5, 1].map((tick) => `
    <text x="${xOf(tick)}" y="${h - 18}" fill="#93a498" font-size="12" text-anchor="middle">${Math.round(tick * 100)}%</text>
    <text x="${padL - 8}" y="${yOf(tick) + 4}" fill="#93a498" font-size="12" text-anchor="end">${Math.round(tick * 100)}%</text>`).join("");
  document.getElementById("reliability").innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Confidence against how often that confidence was right">
      <line x1="${padL}" y1="${yOf(0)}" x2="${xOf(1)}" y2="${yOf(0)}" stroke="#2a3830"/>
      <line x1="${padL}" y1="${yOf(0)}" x2="${padL}" y2="${yOf(1)}" stroke="#2a3830"/>
      <line x1="${padL}" y1="${yOf(0)}" x2="${xOf(1)}" y2="${yOf(1)}" stroke="#3ddec4" stroke-dasharray="4 4" opacity="0.7"/>
      ${ticks}
      ${marks}
      <text x="${(padL + xOf(1)) / 2}" y="${h - 4}" fill="#6e7f74" font-size="12" text-anchor="middle">Stated confidence</text>
      <text transform="translate(14 ${(padT + yOf(0)) / 2}) rotate(-90)" fill="#6e7f74" font-size="12" text-anchor="middle">Hit rate</text>
    </svg>
    <p class="note">Grey-blue is untrained. Gold is trained. A larger dot is more questions in that band. The dashed line is a confidence that matches the hit rate. Expected calibration error on the full test split is ${untrained.ece.toFixed(3)} untrained and ${trained.ece.toFixed(3)} trained.</p>`;
}

function renderTiming() {
  const line = (model) => model.median_seconds == null
    ? `${model.label}: not recorded`
    : `${model.label}: median ${model.median_seconds.toFixed(3)} seconds, mean ${model.mean_seconds.toFixed(3)} seconds`;
  document.getElementById("timing").textContent =
    `${line(untrained)}. ${line(trained)}. ` +
    `This is GPU time for one case after the first case is left out as warmup. Tokenizing and loading the model are not included.`;
}

function renderFoot() {
  document.getElementById("foot").textContent =
    `Gold on ${DATA.dataset} is agreement with that dataset's teacher, not a fact about the world. ` +
    `Trained probabilities use the temperatures fitted on the held-out training rows. ` +
    `Untrained probabilities use the head's default temperature. ` +
    `This file stands alone; it does not call a model.`;
}

function nearest(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let best = null;
  let bestDistance = 10;
  for (const hit of hitTargets.get(canvas) || []) {
    const distance = Math.hypot(hit.x - x, hit.y - y);
    if (distance < bestDistance) {
      best = hit;
      bestDistance = distance;
    }
  }
  return best;
}

function bind() {
  document.getElementById("filters").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    active = button.dataset.filter;
    renderFilters();
    renderGate();
  });
  document.getElementById("gate").addEventListener("input", renderGate);
  const tip = document.getElementById("tip");
  document.getElementById("strips").addEventListener("mousemove", (event) => {
    const canvas = event.target.closest("canvas");
    if (!canvas) {
      tip.style.display = "none";
      return;
    }
    const hit = nearest(canvas, event);
    if (!hit) {
      tip.style.display = "none";
      return;
    }
    const row = hit.row;
    tip.style.display = "block";
    tip.style.left = `${event.clientX + 14}px`;
    tip.style.top = `${event.clientY + 14}px`;
    tip.innerHTML = `${pretty(row.workflow)} · ${row.qtype}<br>${pct(row.confidence)} · ${row.correct ? "matched gold" : "did not match"}`;
  });
  document.getElementById("strips").addEventListener("mouseleave", () => {
    tip.style.display = "none";
  });
  window.addEventListener("resize", renderGate);
}

let ringModel = "untrained";
let ringWorkflow = "";
let ringQuestion = "";

function ringRows(model) {
  return model.decisions.filter((row) => row.question === ringQuestion && row.workflow === ringWorkflow && row.pick != null);
}

function ringSpec() {
  return (DATA.questions || []).find((item) => item.workflow === ringWorkflow && item.question === ringQuestion);
}

function preferredQuestion(workflow) {
  const rows = (DATA.questions || []).filter((item) => item.workflow === workflow);
  const choices = rows.filter((item) => item.qtype === "choice");
  const pool = choices.length ? choices : rows;
  return pool.slice().sort((a, b) => b.options.length - a.options.length || a.question.localeCompare(b.question))[0];
}

function polar(cx, cy, radius, degrees) {
  const radians = (degrees - 90) * Math.PI / 180;
  return [cx + radius * Math.cos(radians), cy + radius * Math.sin(radians)];
}

function arcPath(cx, cy, radius, sweep) {
  const start = polar(cx, cy, radius, 0);
  const end = polar(cx, cy, radius, Math.min(sweep, 359.9));
  const large = sweep > 180 ? 1 : 0;
  return `M ${start[0].toFixed(1)} ${start[1].toFixed(1)} A ${radius} ${radius} 0 ${large} 1 ${end[0].toFixed(1)} ${end[1].toFixed(1)}`;
}

function renderRingControls() {
  const questions = DATA.questions || [];
  const workflows = [...new Set(questions.map((item) => item.workflow))];
  if (!ringWorkflow || !workflows.includes(ringWorkflow)) {
    ringWorkflow = workflows.includes("customer_service") ? "customer_service" : workflows[0] || "";
  }
  const inWorkflow = questions.filter((item) => item.workflow === ringWorkflow);
  if (!inWorkflow.some((item) => item.question === ringQuestion)) {
    ringQuestion = (preferredQuestion(ringWorkflow) || {}).question || "";
  }
  document.getElementById("ring-workflow").innerHTML = workflows.map((name) =>
    `<option value="${esc(name)}" ${name === ringWorkflow ? "selected" : ""}>${esc(pretty(name))}</option>`
  ).join("");
  document.getElementById("ring-question").innerHTML = inWorkflow.map((item) =>
    `<option value="${esc(item.question)}" ${item.question === ringQuestion ? "selected" : ""}>${esc(pretty(item.question))}</option>`
  ).join("");
  document.getElementById("ring-untrained").setAttribute("aria-pressed", String(ringModel === "untrained"));
  document.getElementById("ring-trained").setAttribute("aria-pressed", String(ringModel === "trained"));
}

function esc(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function renderRing() {
  const spec = ringSpec();
  const svg = document.getElementById("ring");
  const center = document.getElementById("ring-center");
  if (!spec) {
    svg.innerHTML = "";
    center.innerHTML = `<div class="ring-caption">The ring is drawn once compare.py has recorded which option each model chose.</div>`;
    return;
  }
  const model = ringModel === "trained" ? trained : untrained;
  const rows = ringRows(model);
  const n = rows.length;
  const hits = n ? rows.filter((row) => row.pick === row.gold).length : 0;
  const accuracy = n ? hits / n : 0;
  const shown = (accuracy * 100).toFixed(1).split(".");
  center.innerHTML = `
    <div class="ring-score">${shown[0]}<span class="frac">.${shown[1]}%</span></div>
    <div class="ring-caption">right on ${n.toLocaleString()} unseen test cases</div>`;
  const cx = 380, cy = 380;
  const count = spec.options.length;
  const labels = spec.options.map((option, index) => {
    const degrees = count ? (360 * index) / count : 0;
    const [x, y] = polar(cx, cy, 332, degrees);
    let rotation = degrees;
    if (degrees > 90 && degrees < 270) rotation += 180;
    const anchor = "middle";
    return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="#e8eef8" font-size="13" font-family="Segoe UI, sans-serif" letter-spacing="1.4" text-anchor="${anchor}" dominant-baseline="middle" transform="rotate(${rotation.toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)})">${esc(option.toUpperCase())}</text>`;
  }).join("");
  const markers = spec.options.map((_, index) => {
    const [x, y] = polar(cx, cy, 286, (360 * index) / count);
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5" fill="#3ee0c5"/>`;
  }).join("");
  const spread = count > 1 ? (360 / count) * 0.22 : 12;
  const dots = rows.map((row, index) => {
    const spoke = (360 * row.pick) / count;
    const angle = spoke + (jitter(index + 3) * 2 - 1) * spread;
    const radius = 196 + (jitter(index + 11) * 2 - 1) * 26;
    const [x, y] = polar(cx, cy, radius, angle);
    const color = row.pick === row.gold ? "#3ee0c5" : "#ff5d73";
    const chosen = spec.options[row.pick] || "";
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4.2" fill="${color}" opacity="0.9"><title>${esc(chosen)} · ${pct(row.confidence)}</title></circle>`;
  });
  const wrong = dots.filter((_, index) => rows[index].pick !== rows[index].gold).join("");
  const right = dots.filter((_, index) => rows[index].pick === rows[index].gold).join("");
  const sweep = accuracy * 360;
  const arc = sweep > 1 ? `<path d="${arcPath(cx, cy, 132, sweep)}" fill="none" stroke="#e4b15a" stroke-width="7" stroke-linecap="round"/>` : "";
  svg.innerHTML = `
    <circle cx="${cx}" cy="${cy}" r="286" fill="none" stroke="#1c3148" stroke-width="1"/>
    ${markers}
    ${wrong}${right}
    <circle cx="${cx}" cy="${cy}" r="112" fill="#070b16"/>
    ${arc}
    ${labels}`;
}

function bindRing() {
  if (!(DATA.questions || []).length) return;
  document.getElementById("ring-untrained").addEventListener("click", () => {
    ringModel = "untrained";
    renderRingControls();
    renderRing();
  });
  document.getElementById("ring-trained").addEventListener("click", () => {
    ringModel = "trained";
    renderRingControls();
    renderRing();
  });
  document.getElementById("ring-workflow").addEventListener("change", (event) => {
    ringWorkflow = event.target.value;
    ringQuestion = "";
    renderRingControls();
    renderRing();
  });
  document.getElementById("ring-question").addEventListener("change", (event) => {
    ringQuestion = event.target.value;
    renderRingControls();
    renderRing();
  });
}

renderIntro();
renderFilters();
mountStrips();
renderBars();
renderTiming();
renderFoot();
bind();
bindRing();
renderRingControls();
renderRing();
renderGate();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
