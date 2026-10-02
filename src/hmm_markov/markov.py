"""Observable Markov chains on discrete sequences.

These are the baselines the hidden Markov models have to beat: if a k-th order chain
already explains the rain series, the hidden layer is not worth its parameters.
"""

from __future__ import annotations

import numpy as np


def fit_transition_matrix(seq: np.ndarray, n_states: int) -> np.ndarray:
    """Maximum-likelihood transition matrix: normalised counts of consecutive pairs."""
    seq = np.asarray(seq, dtype=int)
    counts = np.zeros((n_states, n_states))
    np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
    rows = counts.sum(axis=1, keepdims=True)
    if np.any(rows == 0):
        raise ValueError("every state must be visited at least once before the last step")
    return counts / rows


def stationary_distribution(A: np.ndarray) -> np.ndarray:
    """Solve pi = pi A with sum(pi) = 1 (assumes a single recurrent class)."""
    n = A.shape[0]
    system = np.vstack([A.T - np.eye(n), np.ones(n)])
    rhs = np.zeros(n + 1)
    rhs[-1] = 1.0
    pi, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    return pi


def simulate_chain(A: np.ndarray, pi0: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a path of length n by inverse-CDF sampling (one uniform per step)."""
    cum = np.cumsum(A, axis=1)
    u = rng.random(n)
    out = np.empty(n, dtype=np.int64)
    state = int(np.searchsorted(np.cumsum(pi0), u[0], side="right"))
    out[0] = state
    for t in range(1, n):
        state = int(np.searchsorted(cum[state], u[t], side="right"))
        out[t] = state
    return np.minimum(out, A.shape[0] - 1)


def spell_lengths(seq: np.ndarray, value: int) -> np.ndarray:
    """Lengths of maximal runs equal to ``value``.

    The first and last runs are dropped: they are cut by the observation window, so
    their true length is unknown (right/left censoring) and would bias the tail down.
    """
    seq = np.asarray(seq)
    change = np.flatnonzero(np.diff(seq)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(seq)]])
    lengths = ends - starts
    vals = seq[starts]
    keep = np.ones(len(starts), dtype=bool)
    keep[[0, -1]] = False
    return lengths[keep & (vals == value)]


def geometric_survival(leave_prob: float, lengths: np.ndarray) -> np.ndarray:
    """P(L > l) for the geometric spell length implied by a first-order chain."""
    return (1.0 - leave_prob) ** np.asarray(lengths, dtype=float)


class OrderKChain:
    """k-th order Markov chain on a binary alphabet, fitted by counting.

    The state is the last k symbols. Add-one smoothing keeps unseen contexts finite on
    held-out data.
    """

    def __init__(self, order: int, alpha: float = 1.0) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.alpha = alpha
        self.probs: np.ndarray | None = None  # P(x_t = 1 | context), indexed by context

    def _contexts(self, seq: np.ndarray) -> np.ndarray:
        k = self.order
        ctx = np.zeros(len(seq) - k, dtype=np.int64)
        for lag in range(k, 0, -1):  # oldest symbol first, so the latest is the low bit
            ctx = 2 * ctx + seq[k - lag : len(seq) - lag]
        return ctx

    def fit(self, chunks: list[np.ndarray]) -> OrderKChain:
        ones = np.zeros(2**self.order)
        totals = np.zeros(2**self.order)
        for seq in chunks:
            seq = np.asarray(seq, dtype=np.int64)
            ctx = self._contexts(seq)
            np.add.at(totals, ctx, 1.0)
            np.add.at(ones, ctx, seq[self.order :])
        self.probs = (ones + self.alpha) / (totals + 2 * self.alpha)
        return self

    @property
    def n_params(self) -> int:
        return 2**self.order

    def log_likelihood(self, chunks: list[np.ndarray]) -> tuple[float, int]:
        """Total log-likelihood (nats) of the predicted symbols and how many were scored.

        The first k symbols of each chunk only serve as context and are not scored.
        """
        if self.probs is None:
            raise RuntimeError("fit the chain first")
        total, n = 0.0, 0
        for seq in chunks:
            seq = np.asarray(seq, dtype=np.int64)
            p1 = self.probs[self._contexts(seq)]
            y = seq[self.order :]
            total += float(np.sum(np.where(y == 1, np.log(p1), np.log1p(-p1))))
            n += len(y)
        return total, n

    def sample(self, n: int, rng: np.random.Generator, burn_in: int = 10_000) -> np.ndarray:
        if self.probs is None:
            raise RuntimeError("fit the chain first")
        mask = 2**self.order - 1
        u = rng.random(n + burn_in)
        out = np.empty(n + burn_in, dtype=np.int64)
        ctx = 0
        for t in range(n + burn_in):
            x = int(u[t] < self.probs[ctx])
            out[t] = x
            ctx = ((ctx << 1) | x) & mask
        return out[burn_in:]
