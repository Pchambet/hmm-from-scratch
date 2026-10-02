"""Figures (docs/figures/*.png) and the static report page (site/index.html).

Everything is rendered from results/*.json, so the page and the README figures can
never drift from the numbers the experiments produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .data import ROOT
from .markov import geometric_survival

INK, TEAL, AMBER, SLATE, GRID = "#0f172a", "#0d9488", "#d97706", "#64748b", "#e2e8f0"
FIGURES = ROOT / "docs" / "figures"
SITE = ROOT / "site"
FEATURE_LABEL = {"spectrum": "Log spectrum", "filterbank": "Log mel filterbank", "mfcc": "MFCC"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SLATE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "font.size": 10,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f} %"


def best_loso(words: dict, norm: str) -> dict:
    runs = [
        r for r in words["fsdd"]["leave_one_speaker_out"].values() if r["normalization"] == norm
    ]
    return max(runs, key=lambda r: (r["mean"], -r["states"]))


def key_numbers(rain: dict, words: dict) -> dict:
    """The handful of numbers quoted in the README and the report page."""
    obs = rain["observed"]["test"]
    m1, h3, h4 = (rain["models"][k] for k in ("markov_1", "hmm_3", "hmm_4"))
    chains = [m for key, m in rain["models"].items() if key.startswith("markov")]
    best_chain = max(chains, key=lambda m: m["test_gain_nats_per_day"])
    fr = words["fruits"]
    best = fr["grid"][fr["best"]]
    loso = words["fsdd"]["leave_one_speaker_out"]
    k = best["states"]
    noise = words["noise"]
    i20 = noise["snr_db"].index(20)
    return {
        "obs_p72": obs["p_dry_gt_72h"],
        "m1_p72": m1["p_dry_gt_72h_exact"],
        "h3_p72": h3["simulated"]["p_dry_gt_72h"],
        "m1_ks": m1["simulated"]["ks_dry"],
        "h3_ks": h3["simulated"]["ks_dry"],
        "h4_ks_wet": h4["simulated"]["ks_wet"],
        "m1_ks_wet": m1["simulated"]["ks_wet"],
        "h3_gain": h3["test_gain_nats_per_day"],
        "h4_gain": h4["test_gain_nats_per_day"],
        "chain_gain": best_chain["test_gain_nats_per_day"],
        "chain_label": best_chain["label"],
        "chain_params": best_chain["n_params"],
        "fruits_best": f"{FEATURE_LABEL[best['feature']]}, {k} states",
        "fruits_best_acc": best["mean"],
        "fruits_1state_acc": fr["grid"][f"{best['feature']}/1"]["mean"],
        "official_acc": words["fsdd"]["official_split"]["results"][str(k)]["accuracy"],
        "loso_raw": loso[f"none/{k}"]["mean"],
        "loso_cmn": loso[f"cmn/{k}"]["mean"],
        "loso_cmn_1": loso["cmn/1"]["mean"],
        "loso_raw_1": loso["none/1"]["mean"],
        "loso_cmn_min": loso[f"cmn/{k}"]["min"],
        "noise20": {name: c["by_snr"][i20] for name, c in noise["curves"].items()},
        "k": k,
        "feature": best["feature"],
        "restart_spread": max(
            max(m["restart_final_logliks"]) - min(m["restart_final_logliks"])
            for key, m in rain["models"].items()
            if key.startswith("hmm")
        ),
    }


# --------------------------------------------------------------------------- figures


def chain_dry_survival(rain: dict) -> np.ndarray:
    """Exact P(dry spell > x) of the first-order chain on the survival grid.

    Its dry spells are geometric, so the closed form replaces a simulated estimate that
    runs out of spells in the tail.
    """
    steps = np.round(np.array(rain["survival_hours"]) * 60 / 5)
    p_leave = rain["models"]["markov_1"]["p_wet_given_context"][0]
    return geometric_survival(p_leave, steps)


def _survival_axis(ax, rain: dict) -> None:
    hours = np.array(rain["survival_hours"])
    series = [
        ("Observed, held-out", rain["observed"]["test"]["dry_survival"], INK, "o", "-"),
        ("Markov chain", chain_dry_survival(rain), SLATE, None, "--"),
        ("3-state HMM", rain["models"]["hmm_3"]["simulated"]["dry_survival"], TEAL, None, "-"),
    ]
    for label, values, color, marker, ls in series:
        y = np.array(values)
        keep = y > 0
        ax.plot(
            hours[keep],
            y[keep],
            color=color,
            ls=ls,
            lw=2 if marker is None else 0,
            marker=marker,
            ms=3.5,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1 / 12, 1, 6, 24, 72, 168], ["5 min", "1 h", "6 h", "1 d", "3 d", "7 d"])
    ax.set_xlabel("Dry-spell length")
    ax.set_ylabel("P(dry spell longer than x)")


def fig_hero(rain: dict, words: dict, n: dict, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"wspace": 0.28})
    _survival_axis(a1, rain)
    a1.set_ylim(5e-4, 1.3)
    a1.set_xlim(1 / 12, 400)
    a1.annotate(
        f"> 3 days: observed {pct(n['obs_p72'])}\nMarkov chain {pct(n['m1_p72'])}"
        f" · HMM {pct(n['h3_p72'])}",
        xy=(72, n["obs_p72"]),
        xytext=(0.9, 2e-3),
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": SLATE, "lw": 0.8},
    )
    a1.text(30, 0.5, "Markov chain\n(geometric)", color=SLATE, fontsize=9)
    a1.text(0.12, 0.12, "observed,\nheld-out days", color=INK, fontsize=9)
    a1.text(200, 4e-3, "3-state\nHMM", color=TEAL, fontsize=9, va="center")
    a1.set_title("Rain: hidden states reproduce the multi-day dry spells\na Markov chain cannot")

    loso = words["fsdd"]["leave_one_speaker_out"]
    states = words["protocol"]["states_grid"]
    for norm, color, label in (("none", SLATE, "raw MFCC"), ("cmn", TEAL, "MFCC + mean norm.")):
        acc = [100 * loso[f"{norm}/{k}"]["mean"] for k in states]
        a2.plot(states, acc, color=color, lw=2, marker="o", ms=5)
        a2.annotate(
            label,
            (states[-1], acc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
        )
    a2.axhline(10, color=SLATE, lw=1, ls=":")
    a2.text(states[0], 12, "chance", color=SLATE, fontsize=8.5)
    a2.set_xticks(states)
    a2.set_xlim(0.5, states[-1] + 2.6)
    a2.set_ylim(0, 105)
    a2.set_xlabel("Hidden states per word model (1 = no time structure)")
    a2.set_ylabel("Digit accuracy, speaker left out of training (%)")
    a2.set_title(
        f"Speech: digits from an unseen speaker are recognised {pct(n['loso_cmn'], 0)}"
        f" of the time\nwith {n['k']} states + mean normalisation"
        f" ({pct(n['loso_cmn_1'], 0)} with 1 state)"
    )
    fig.savefig(path)
    plt.close(fig)


def gain_order(rain: dict) -> list[str]:
    """Every model but the first-order chain, which is the zero of the gain scale."""
    return [k for k in rain["models"] if k != "markov_1"]


def fig_rain_models(rain: dict, path: Path) -> None:
    order = gain_order(rain)
    models = rain["models"]
    gains = [models[k]["test_gain_nats_per_day"] for k in order]
    labels = [f"{models[k]['label']} ({models[k]['n_params']} params)" for k in order]
    colors = [SLATE if k.startswith("markov") else TEAL for k in order]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_axisbelow(True)
    ax.barh(labels, gains, color=colors, height=0.6)
    for i, g in enumerate(gains):
        ax.text(g + 0.04, i, f"+{g:.2f}", va="center", fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Held-out log-likelihood gain over a first-order Markov chain (nats per day)")
    chains = [k for k in order if k.startswith("markov")]
    big = max(chains, key=lambda k: models[k]["n_params"])
    gain = {k: models[k]["test_gain_nats_per_day"] for k in order}
    beating = [k for k in order if k.startswith("hmm") and gain[k] > gain[big]]
    if beating:
        small = min(beating, key=lambda k: models[k]["n_params"])
        title = (
            f"a {small.split('_')[1]}-state HMM ({models[small]['n_params']} params) beats\n"
            f"a chain of order {big.split('_')[1]} ({models[big]['n_params']} params)"
        )
    else:
        title = f"no HMM beats the chain of order {big.split('_')[1]}"
    ax.set_title(f"Held-out {rain['test_days']:.0f} days: {title}")
    fig.savefig(path)
    plt.close(fig)


def fig_rain_week(rain: dict, path: Path) -> None:
    week = rain["decoded_week"]
    obs, states = np.array(week["obs"]), np.array(week["states"])
    days = np.arange(len(obs)) / 288
    p_wet = rain["models"]["hmm_3"]["p_wet_given_state"]
    fig, (a1, a2) = plt.subplots(
        2,
        1,
        figsize=(10, 3.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.15},
    )
    a1.bar(days, obs, width=1 / 288, color=INK)
    a1.set_yticks([])
    a1.set_ylabel("wet", rotation=0, ha="right", va="center")
    a1.grid(False)
    a1.set_title("Viterbi decoding of the wettest held-out week (3-state HMM)")
    palette = [GRID, AMBER, TEAL]
    for s in range(3):
        a2.fill_between(
            days, s - 0.4, s + 0.4, where=states == s, color=palette[s], step="mid", lw=0
        )
    a2.set_yticks(range(3), [f"state {s}: P(wet) = {p:.3f}" for s, p in enumerate(p_wet)])
    a2.set_ylim(-0.6, 2.6)
    a2.grid(False)
    a2.set_xlabel("Day of the week shown")
    fig.savefig(path)
    plt.close(fig)


def fig_words_grid(words: dict, path: Path) -> None:
    grid = words["fruits"]["grid"]
    feats = list(FEATURE_LABEL)
    states = words["protocol"]["states_grid"]
    acc = np.array([[100 * grid[f"{f}/{k}"]["mean"] for k in states] for f in feats])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    cmap = LinearSegmentedColormap.from_list("teal", ["#f0fdfa", TEAL])
    ax.imshow(acc, cmap=cmap, vmin=90, vmax=100, aspect="auto")
    for i in range(len(feats)):
        for j in range(len(states)):
            ax.text(
                j,
                i,
                f"{acc[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if acc[i, j] > 97 else INK,
            )
    ax.set_xticks(range(len(states)), [str(k) for k in states])
    ax.set_yticks(range(len(feats)), [FEATURE_LABEL[f] for f in feats])
    ax.set_xlabel("Hidden states per word")
    ax.grid(False)
    ax.set_title("Fruit words, 5-fold CV x3: accuracy (%) saturates for MFCC")
    fig.savefig(path)
    plt.close(fig)


def fig_noise(words: dict, path: Path) -> None:
    noise = words["noise"]
    snr = ["clean", *[f"{s} dB" for s in noise["snr_db"]]]
    styles = {
        "spectrum": (AMBER, "-"),
        "filterbank": (SLATE, "-"),
        "mfcc": (TEAL, "-"),
        "mfcc_1state": (TEAL, ":"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(snr))
    for name, curve in noise["curves"].items():
        color, ls = styles[name]
        y = 100 * np.array([curve["clean"], *curve["by_snr"]])
        label = f"{FEATURE_LABEL[curve['feature']]}, {curve['states']} state" + (
            "s" if curve["states"] > 1 else ""
        )
        ax.plot(x, y, color=color, ls=ls, lw=2, marker="o", ms=4, label=label)
    ax.legend(loc="upper right", fontsize=9)
    n_words = len(words["fruits"]["words"])
    ax.axhline(100 / n_words, color=SLATE, lw=1, ls=":")
    ax.text(-0.2, 100 / n_words + 2, f"chance (1/{n_words})", color=SLATE, fontsize=8.5)
    ax.set_xticks(x, snr)
    ax.set_xlim(-0.3, len(snr) - 0.7)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Test-time signal-to-noise ratio (white noise; models trained on clean audio)")
    ax.set_ylabel("Fruit-word accuracy (%)")
    ax.set_title(
        "Noise (fruit words, one speaker, 5-fold CV): 3-state MFCC models hold\n"
        "to 20 dB; below that, the time-blind 1-state model degrades more gracefully"
    )
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- page

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Hidden Markov Models, Measured</title>
<meta name="description" content="Rain spells and spoken words with hidden Markov models written
from scratch in NumPy, validated on held-out data.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
 rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root { --bg:#ffffff; --fg:#0f172a; --muted:#64748b; --rule:#e2e8f0; --card:#f8fafc;
  --teal:#0d9488; --amber:#d97706; --slate:#64748b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#0b1120; --fg:#e2e8f0; --muted:#94a3b8; --rule:#1e293b; --card:#111827;
  --teal:#2dd4bf; --amber:#fbbf24; --slate:#94a3b8; } }
:root[data-theme="dark"] { --bg:#0b1120; --fg:#e2e8f0; --muted:#94a3b8; --rule:#1e293b;
  --card:#111827; --teal:#2dd4bf; --amber:#fbbf24; --slate:#94a3b8; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font: 16px/1.65 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 40px 16px 64px; }
h1 { font-size: 2rem; line-height:1.2; margin: 0 0 8px; letter-spacing:-0.01em; }
h2 { font-size: 1.3rem; margin: 48px 0 8px; padding-top: 16px; border-top: 1px solid var(--rule); }
p, li { color: var(--fg); }
.lede { color: var(--muted); font-size: 1.08rem; margin: 0 0 24px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px;
  margin: 24px 0; }
.kpi { background:var(--card); border:1px solid var(--rule); border-radius:10px; padding:14px; }
.kpi b { display:block; font-size:1.5rem; color:var(--teal); }
.kpi span { color:var(--muted); font-size:.88rem; }
.chart { width:100%; height:380px; margin: 8px 0 4px; }
.note { color:var(--muted); font-size:.9rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width:100%; font-size:.92rem; margin: 12px 0; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--rule); }
th { color: var(--muted); font-weight:600; }
td.num { text-align:right; font-variant-numeric: tabular-nums; }
a { color: var(--teal); }
code { font-size: .9em; }
footer { margin-top: 56px; color: var(--muted); font-size: .9rem; }
</style>
</head>
<body>
<main>
<h1>Hidden Markov models, measured</h1>
<p class="lede">Two questions, one model family written from scratch in NumPy and judged only
on data it never saw: do hidden states explain how long it stays dry, and how far does a
word recogniser built on them carry to a voice it has never heard?</p>
<div class="kpis">
 <div class="kpi"><b>__OBS72__ vs __M172__</b><span>dry spells over 3 days: observed vs
 first-order Markov chain (3-state HMM: __H372__)</span></div>
 <div class="kpi"><b>+__H4GAIN__ nats/day</b><span>held-out log-likelihood of the best HMM over
 the first-order chain (best chain, __CHAINLABEL__ with __CHAINPARAMS__ parameters:
 +__CHAINGAIN__)</span></div>
 <div class="kpi"><b>__LOSO__</b><span>digit accuracy on an unseen speaker, __K__-state HMMs with
 mean normalisation (raw features: __LOSORAW__)</span></div>
</div>

<h2>1. Rain: how long does it stay dry?</h2>
<p>The record holds __DAYS__ days of 5-minute rain-gauge data. Models are fitted on the first
__TRAIN__ days and judged on the last __TEST__. A first-order Markov chain implies geometric spell
lengths; the observed dry spells have a much heavier tail. Hidden states, each with its own
persistence, produce a mixture of geometric laws, which is what the data look like.</p>
<div id="survival" class="chart"></div>
<p class="note">Survival of dry-spell lengths, log-log. The first-order chain's curve is its
exact geometric law; the HMM curves come from simulating each fitted model for 3x the record
length. Kolmogorov-Smirnov distance to the held-out spells: Markov chain __M1KS__, 3-state HMM
__H3KS__.</p>
<div id="gain" class="chart" style="height:320px"></div>
<p class="note">Held-out log-likelihood gain over the first-order chain, per day of test data.
All models score exactly the same steps; p. = free parameters. Chains of order 5 and 6
have more parameters than any HMM here (32 and 64, against at most 19) and still gain less
than the 3-state HMM.</p>
__RAINTABLE__

<h2>2. Speech: which features, how many states, which speaker?</h2>
<p>One left-to-right Gaussian HMM per word; a recording is assigned to the word whose model
explains it best. The feature set and state count are chosen by cross-validation on a small
single-speaker corpus of 7 fruit names (__FRUITS__ recordings), where the selected model,
__BEST__, reaches __BESTACC__ (a 1-state bag of frames already gets __FR1__: this corpus is
easy). The choice is then tested on the Free Spoken Digit Dataset (6 speakers, 3,000
recordings): __OFFICIAL__ on its official split, where test speakers were also heard in
training, and __LOSORAW__ when the test speaker is left out entirely.</p>
<div id="loso" class="chart"></div>
<p class="note">Leave-one-speaker-out accuracy, averaged over the 6 speakers. Mean
normalisation removes each recording's average cepstrum: it erases a speaker's fixed
spectral colouring, and it also erases what a 1-state model relies on, hence its collapse.</p>
<div id="noise" class="chart"></div>
<p class="note">Fruit-word corpus (one speaker, 5-fold CV x3), models trained on clean audio
and tested with white noise added; chance is 1/7. At 20 dB: MFCC __N20MFCC__, filterbank
__N20FB__, spectrum __N20SP__.</p>

<h2>Limitations</h2>
<ul>
<li>Beyond about 3 days the 3-state HMM thins out faster than the record (P(dry &gt; 7 d):
__H3P7__ simulated vs __OBSP7__ observed): its slowest state still has a geometric tail.</li>
<li>One rain gauge, 761 days, location undocumented; the held-out period is a different season
mix from the training period.</li>
<li>HMM likelihoods are maximised by EM from 4 starts; the best and worst start end within
__SPREAD__ nats of each other, but a global optimum is not guaranteed.</li>
<li>The fruit corpus is tiny and saturated; the digit corpus has only 6 speakers, so the
unseen-speaker figure is an average of 6 numbers (worst speaker: __LOSOMIN__).</li>
<li>No language model, no deltas, no GMM emissions: the point is a clean baseline, not a
state-of-the-art recogniser.</li>
</ul>
<footer>Built by <a href="https://github.com/Pchambet">Pierre Chambet</a> — decision science
for operations under uncertainty. Code and data: <a
href="https://github.com/Pchambet/tp-hmm-markov">github.com/Pchambet/tp-hmm-markov</a>.</footer>
</main>
<script>
const DATA = __DATA__;
function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function layout(extra) {
  const fg = css('--fg'), muted = css('--muted'), rule = css('--rule');
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {family: 'Inter, system-ui, sans-serif', color: fg, size: 13},
    margin: {l: 64, r: 16, t: 16, b: 56},
    legend: {orientation: 'h', x: 0, xanchor: 'left', y: 1.02, yanchor: 'bottom',
      bgcolor: 'rgba(0,0,0,0)'},
  }, extra, {
    xaxis: Object.assign({gridcolor: rule, zerolinecolor: rule, linecolor: muted,
      automargin: true}, extra.xaxis),
    yaxis: Object.assign({gridcolor: rule, zerolinecolor: rule, linecolor: muted,
      automargin: true}, extra.yaxis),
  });
}
const cfg = {displayModeBar: false, responsive: true};
function draw() {
  const teal = css('--teal'), amber = css('--amber'), slate = css('--slate'), fg = css('--fg');
  const s = DATA.survival;
  const tr = (name, y, color, dash, mode) => ({x: s.hours.filter((_, i) => y[i] > 0),
    y: y.filter(v => v > 0), name, mode: mode || 'lines', line: {color, dash, width: 2.5},
    marker: {color, size: 6}, hovertemplate: '%{x:.2f} h: %{y:.4f}<extra>' + name + '</extra>'});
  Plotly.react('survival', [tr('Observed (held-out)', s.observed, fg, 'solid', 'markers'),
    tr('Markov chain', s.markov, slate, 'dash'), tr('3-state HMM', s.hmm3, teal, 'solid'),
    tr('4-state HMM', s.hmm4, amber, 'dot')], layout({
    xaxis: {type: 'log', title: {text: 'dry-spell length'},
      tickvals: [1 / 12, 1, 6, 24, 72, 168],
      ticktext: ['5 min', '1 h', '6 h', '1 d', '3 d', '7 d'], minor: {showgrid: false}},
    yaxis: {type: 'log', range: [-3.3, 0.05], title: {text: 'P(dry spell longer than x)'},
      tickvals: [0.001, 0.01, 0.1, 1], ticktext: ['0.1 %', '1 %', '10 %', '100 %']}}), cfg);
  const g = DATA.gain;
  Plotly.react('gain', [{type: 'bar', orientation: 'h', x: g.values, y: g.labels,
    marker: {color: g.labels.map(l => l.startsWith('HMM') ? teal : slate)},
    text: g.values.map(v => '+' + v.toFixed(2)), textposition: 'outside', cliponaxis: false,
    hovertemplate: '%{y}: %{x:.2f} nats/day<extra></extra>'}], layout({
    margin: {l: 8, r: 56, t: 8, b: 48}, showlegend: false,
    xaxis: {title: {text: 'nats per day vs first-order chain'},
      range: [0, 1.15 * Math.max(...g.values)]},
    yaxis: {autorange: 'reversed'}}), cfg);
  const l = DATA.loso;
  Plotly.react('loso', [
    {x: l.states, y: l.none, name: 'raw MFCC', mode: 'lines+markers', line: {color: slate}},
    {x: l.states, y: l.cmn, name: 'MFCC + mean normalisation', mode: 'lines+markers',
     line: {color: teal, width: 3}}], layout({
    xaxis: {title: {text: 'hidden states per word model'}, tickvals: l.states},
    yaxis: {range: [0, 100], title: {text: 'unseen-speaker digit accuracy (%)'}},
    shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 10, y1: 10,
      line: {color: slate, dash: 'dot', width: 1}}],
    annotations: [{xref: 'paper', x: 1, xanchor: 'right', y: 10, yanchor: 'bottom',
      text: 'chance (1/10)', showarrow: false, font: {color: slate, size: 12}}]}), cfg);
  const n = DATA.noise;
  const colors = {spectrum: amber, filterbank: slate, mfcc: teal, mfcc_1state: teal};
  Plotly.react('noise', Object.entries(n.curves).map(([k, c]) => ({x: n.x, y: c.y, name: c.label,
    mode: 'lines+markers', line: {color: colors[k], dash: k.endsWith('1state') ? 'dot' : 'solid'}})),
    layout({yaxis: {range: [0, 102], title: {text: 'fruit-word accuracy (%)'}},
      xaxis: {title: {text: 'test-time signal-to-noise ratio'}},
      shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: n.chance, y1: n.chance,
        line: {color: slate, dash: 'dot', width: 1}}],
      annotations: [{xref: 'paper', x: 0, xanchor: 'left', y: n.chance, yanchor: 'bottom',
        text: 'chance (1/' + n.n_words + ')', showarrow: false,
        font: {color: slate, size: 12}}]}),
    cfg);
}
draw();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script>
</body>
</html>
"""


def rain_table(rain: dict) -> str:
    rows = []
    for m in rain["models"].values():
        sim = m["simulated"]
        tail = m.get("p_dry_gt_72h_exact", sim["p_dry_gt_72h"])
        rows.append(
            f"<tr><td>{m['label']}</td><td class=num>{m['n_params']}</td>"
            f"<td class=num>{m['test_bits_per_step'] * 1000:.1f}</td>"
            f"<td class=num>{sim['ks_dry']:.3f}</td><td class=num>{sim['ks_wet']:.3f}</td>"
            f"<td class=num>{pct(tail, 2 if tail < 0.01 else 1)}</td></tr>"
        )
    obs = rain["observed"]["test"]
    rows.append(
        f"<tr><td><b>Observed, held-out</b></td><td></td><td></td><td></td><td></td>"
        f"<td class=num>{pct(obs['p_dry_gt_72h'])}</td></tr>"
    )
    head = (
        "<tr><th>Model</th><th>params</th><th>held-out log-loss (millibits/step)</th>"
        "<th>KS dry</th><th>KS wet</th><th>P(dry &gt; 3 d)</th></tr>"
    )
    return f"<div class=scroll><table>{head}{''.join(rows)}</table></div>"


def short_label(key: str, n_params: int) -> str:
    """Bar label short enough for a phone-width chart, e.g. 'HMM, 3 states · 11 p.'."""
    family, size = key.split("_")
    name = f"chain, order {size}" if family == "markov" else f"HMM, {size} states"
    return f"{name} · {n_params} p."


def page_data(rain: dict, words: dict) -> dict:
    models = rain["models"]
    order = gain_order(rain)
    loso = words["fsdd"]["leave_one_speaker_out"]
    states = words["protocol"]["states_grid"]
    noise = words["noise"]
    curves = {}
    for name, c in noise["curves"].items():
        plural = "s" if c["states"] > 1 else ""
        curves[name] = {
            "label": f"{FEATURE_LABEL[c['feature']]}, {c['states']} state{plural}",
            "y": [round(100 * v, 2) for v in [c["clean"], *c["by_snr"]]],
        }
    return {
        "survival": {
            "hours": rain["survival_hours"],
            "observed": rain["observed"]["test"]["dry_survival"],
            "markov": chain_dry_survival(rain).tolist(),
            "hmm3": models["hmm_3"]["simulated"]["dry_survival"],
            "hmm4": models["hmm_4"]["simulated"]["dry_survival"],
        },
        "gain": {
            "labels": [short_label(k, models[k]["n_params"]) for k in order],
            "values": [round(models[k]["test_gain_nats_per_day"], 2) for k in order],
        },
        "loso": {
            "states": states,
            "none": [round(100 * loso[f"none/{k}"]["mean"], 2) for k in states],
            "cmn": [round(100 * loso[f"cmn/{k}"]["mean"], 2) for k in states],
        },
        "noise": {
            "x": ["clean", *[f"{s} dB" for s in noise["snr_db"]]],
            "curves": curves,
            "n_words": len(words["fruits"]["words"]),
            "chance": round(100 / len(words["fruits"]["words"]), 2),
        },
    }


def render_page(rain: dict, words: dict, n: dict) -> str:
    subs = {
        "__OBS72__": pct(n["obs_p72"]),
        "__M172__": pct(n["m1_p72"]),
        "__H372__": pct(n["h3_p72"]),
        "__H4GAIN__": f"{n['h4_gain']:.1f}",
        "__CHAINGAIN__": f"{n['chain_gain']:.1f}",
        "__CHAINLABEL__": n["chain_label"].lower().removeprefix("markov chain, "),
        "__CHAINPARAMS__": str(n["chain_params"]),
        "__LOSO__": pct(n["loso_cmn"]),
        "__LOSORAW__": pct(n["loso_raw"]),
        "__LOSOMIN__": pct(n["loso_cmn_min"]),
        "__K__": str(n["k"]),
        "__DAYS__": f"{rain['n_days']:.0f}",
        "__TRAIN__": f"{rain['train_days']:.0f}",
        "__TEST__": f"{rain['test_days']:.0f}",
        "__M1KS__": f"{n['m1_ks']:.2f}",
        "__H3KS__": f"{n['h3_ks']:.2f}",
        "__FRUITS__": str(words["fruits"]["n_utterances"]),
        "__BEST__": n["fruits_best"],
        "__BESTACC__": pct(n["fruits_best_acc"]),
        "__FR1__": pct(n["fruits_1state_acc"]),
        "__OFFICIAL__": pct(n["official_acc"]),
        "__N20MFCC__": pct(n["noise20"]["mfcc"], 0),
        "__N20FB__": pct(n["noise20"]["filterbank"], 0),
        "__N20SP__": pct(n["noise20"]["spectrum"], 0),
        "__H3P7__": pct(rain["models"]["hmm_3"]["simulated"]["dry_survival"][-1]),
        "__OBSP7__": pct(rain["observed"]["test"]["dry_survival"][-1]),
        "__SPREAD__": f"{n['restart_spread']:.2f}",
        "__RAINTABLE__": rain_table(rain),
        "__DATA__": json.dumps(page_data(rain, words)),
    }
    html = PAGE
    for key, value in subs.items():
        html = html.replace(key, value)
    return html


def build(results: Path) -> list[Path]:
    rain = json.loads((results / "rain.json").read_text())
    words = json.loads((results / "words.json").read_text())
    n = key_numbers(rain, words)
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(exist_ok=True)
    out = []
    for name, fn in (
        ("hero.png", lambda p: fig_hero(rain, words, n, p)),
        ("rain_models.png", lambda p: fig_rain_models(rain, p)),
        ("rain_week.png", lambda p: fig_rain_week(rain, p)),
        ("words_grid.png", lambda p: fig_words_grid(words, p)),
        ("words_noise.png", lambda p: fig_noise(words, p)),
    ):
        fn(FIGURES / name)
        out.append(FIGURES / name)
    page = SITE / "index.html"
    page.write_text(render_page(rain, words, n))
    out.append(page)
    return out
