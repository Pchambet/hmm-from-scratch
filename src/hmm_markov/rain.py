"""Experiment 1: do hidden states explain rain spells better than an observable chain?

Protocol
- 761 days of 5-minute wet/dry data, split chronologically: first 70 % for fitting,
  last 30 % held out. The split is aligned on whole weeks.
- Both splits are cut into week-long chunks (2016 steps). Every model scores the same
  symbols: steps 4..2016 of each held-out week, conditioned on the steps before them
  within that week, so k-th order chains (k <= 3) and HMMs are compared like for like.
- Candidates: observable Markov chains of order 1-3, and HMMs with K = 2, 3, 4 hidden
  states and Bernoulli emissions, each fitted by Baum-Welch from several starts.
- Spell realism: simulate each fitted model for 3x the record length and compare the
  dry- and wet-spell length distributions with the held-out record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import STEPS_PER_DAY, load_rain
from .hmm import HMM, Categorical, pad
from .markov import OrderKChain, spell_lengths, stationary_distribution

CHUNK = 7 * STEPS_PER_DAY
CONTEXT = 3  # symbols per chunk used only as context, so all models score the same steps
TRAIN_FRACTION = 0.7
HMM_STATES = (2, 3, 4)
N_RESTARTS = 4
SIM_FACTOR = 3
# Spell-length grid for survival curves: 5 min to 7 days, log-spaced, in 5-min steps.
SURVIVAL_STEPS = np.unique(np.round(np.geomspace(1, 7 * STEPS_PER_DAY, 48)).astype(int))

# The hand-set 3-state model from the course (clear / cloudy / very cloudy sky), used as
# one of the Baum-Welch starting points for K = 3.
COURSE_INIT = (
    np.array([0.8, 0.15, 0.05]),
    np.array([[0.90, 0.09, 0.01], [0.20, 0.60, 0.20], [0.05, 0.25, 0.70]]),
    np.array([[0.999, 0.001], [0.80, 0.20], [0.20, 0.80]]),
)


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray

    def chunks(self, which: str) -> list[np.ndarray]:
        x = getattr(self, which)
        return [
            x[i : i + CHUNK] for i in range(0, len(x), CHUNK) if len(x[i : i + CHUNK]) > CONTEXT
        ]


def chronological_split(x: np.ndarray, fraction: float = TRAIN_FRACTION) -> Split:
    cut = int(fraction * len(x)) // CHUNK * CHUNK
    return Split(train=x[:cut], test=x[cut:])


def random_hmm(k: int, rng: np.random.Generator) -> HMM:
    """Sticky random start: persistence is what rain spells are about."""
    A = rng.dirichlet(np.ones(k), size=k) * 0.1 + np.eye(k) * 0.9
    probs = np.sort(rng.uniform(0.001, 0.9, size=k))
    emission = Categorical(np.column_stack([1 - probs, probs]))
    return HMM(pi=np.full(k, 1.0 / k), A=A, emission=emission)


def fit_hmm(chunks: list[np.ndarray], k: int, seed: int) -> tuple[HMM, list[float]]:
    """Best of several Baum-Welch runs (by training log-likelihood)."""
    X, lengths = pad(chunks)
    rng = np.random.default_rng(seed)
    starts = [random_hmm(k, rng) for _ in range(N_RESTARTS)]
    if k == 3:
        pi, A, B = COURSE_INIT
        starts[0] = HMM(pi.copy(), A.copy(), Categorical(B.copy()))
    finals = []
    for model in starts:
        model.fit(X, lengths, max_iter=500, tol=1e-7)
        finals.append(model.history[-1])
    best = starts[int(np.argmax(finals))]
    return order_states(best), finals


def order_states(model: HMM) -> HMM:
    """Relabel states by increasing wet probability so outputs are comparable."""
    order = np.argsort(model.emission.probs[:, 1])
    return HMM(
        pi=model.pi[order],
        A=model.A[np.ix_(order, order)],
        emission=Categorical(model.emission.probs[order]),
        history=model.history,
    )


def hmm_conditional_loglik(model: HMM, chunks: list[np.ndarray]) -> float:
    """log p(x_{CONTEXT+1:T} | x_{1:CONTEXT}) summed over chunks."""
    X, lengths = pad(chunks)
    full = model.score(X, lengths)
    prefix = model.score(X[:, :CONTEXT], np.full(len(chunks), CONTEXT))
    return float(np.sum(full - prefix))


def chain_conditional_loglik(chain: OrderKChain, chunks: list[np.ndarray]) -> float:
    """Same scored steps as the HMMs: drop the symbols before CONTEXT."""
    trimmed = [c[CONTEXT - chain.order :] for c in chunks]
    total, _ = chain.log_likelihood(trimmed)
    return total


def survival(lengths: np.ndarray, thresholds_steps: np.ndarray) -> np.ndarray:
    """Empirical P(L > l) for each threshold."""
    lengths = np.sort(lengths)
    return 1.0 - np.searchsorted(lengths, thresholds_steps, side="right") / len(lengths)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic on integer spell lengths."""
    grid = np.union1d(a, b)
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def spell_summary(seq: np.ndarray, reference: np.ndarray | None = None) -> dict:
    dry, wet = spell_lengths(seq, 0), spell_lengths(seq, 1)
    steps = SURVIVAL_STEPS
    out = {
        "wet_fraction": float(seq.mean()),
        "n_dry_spells": len(dry),
        "mean_dry_hours": float(dry.mean() * 5 / 60),
        "mean_wet_minutes": float(wet.mean() * 5),
        "p_dry_gt_24h": float(np.mean(dry > STEPS_PER_DAY)),
        "p_dry_gt_72h": float(np.mean(dry > 3 * STEPS_PER_DAY)),
        "p_wet_gt_1h": float(np.mean(wet > 12)),
        "dry_survival": survival(dry, steps).tolist(),
        "wet_survival": survival(wet, steps).tolist(),
    }
    if reference is not None:
        out["ks_dry"] = ks_distance(dry, spell_lengths(reference, 0))
        out["ks_wet"] = ks_distance(wet, spell_lengths(reference, 1))
    return out


def run(seed: int = 0) -> dict:
    x = load_rain()
    split = chronological_split(x)
    train_chunks, test_chunks = split.chunks("train"), split.chunks("test")
    n_scored = sum(len(c) - CONTEXT for c in test_chunks)
    test_days = n_scored / STEPS_PER_DAY
    rng = np.random.default_rng(seed)
    n_sim = SIM_FACTOR * len(x)

    models: dict[str, dict] = {}
    fitted: dict[str, HMM] = {}
    for order in (1, 2, 3):
        chain = OrderKChain(order).fit(train_chunks)
        sim = chain.sample(n_sim, rng)
        models[f"markov_{order}"] = {
            "label": f"Markov chain, order {order}",
            "n_params": chain.n_params,
            "train_loglik": chain_conditional_loglik(chain, train_chunks),
            "test_loglik": chain_conditional_loglik(chain, test_chunks),
            "p_wet_given_context": chain.probs.tolist(),
            "simulated": spell_summary(sim, reference=split.test),
        }
    for k in HMM_STATES:
        model, finals = fit_hmm(train_chunks, k, seed=seed + k)
        fitted[f"hmm_{k}"] = model
        _, sim = model.sample(n_sim, rng)
        models[f"hmm_{k}"] = {
            "label": f"HMM, {k} hidden states",
            "n_params": model.n_free_params(),
            "train_loglik": hmm_conditional_loglik(model, train_chunks),
            "test_loglik": hmm_conditional_loglik(model, test_chunks),
            "restart_final_logliks": finals,
            "em_iterations": len(model.history),
            "pi": model.pi.tolist(),
            "A": model.A.tolist(),
            "p_wet_given_state": model.emission.probs[:, 1].tolist(),
            "stationary_wet_fraction": stationary_wet(model),
            "simulated": spell_summary(sim, reference=split.test),
        }

    base = models["markov_1"]["test_loglik"]
    for m in models.values():
        m["test_gain_nats_per_day"] = (m["test_loglik"] - base) / test_days
        m["test_bits_per_step"] = -m["test_loglik"] / n_scored / np.log(2)

    week = decode_week(fitted["hmm_3"], split.test)
    return {
        "n_steps": len(x),
        "n_days": len(x) / STEPS_PER_DAY,
        "train_days": len(split.train) / STEPS_PER_DAY,
        "test_days": len(split.test) / STEPS_PER_DAY,
        "scored_test_steps": n_scored,
        "survival_hours": (SURVIVAL_STEPS * 5 / 60).tolist(),
        "observed": {
            "full": spell_summary(x),
            "train": spell_summary(split.train),
            "test": spell_summary(split.test),
        },
        "models": models,
        "decoded_week": week,
    }


def stationary_wet(model: HMM) -> float:
    return float(stationary_distribution(model.A) @ model.emission.probs[:, 1])


def decode_week(model: HMM, test: np.ndarray) -> dict:
    """Viterbi path over the held-out week with the most wet steps (for the figure)."""
    weeks = [test[i : i + CHUNK] for i in range(0, len(test) - CHUNK + 1, CHUNK)]
    idx = int(np.argmax([w.sum() for w in weeks]))
    obs = weeks[idx]
    path, _ = model.viterbi(obs)
    return {"week_index": idx, "obs": obs.tolist(), "states": path.tolist()}
