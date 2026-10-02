"""Hidden Markov models: forward-backward, Baum-Welch and Viterbi in NumPy.

Written from scratch rather than wrapping a library so that every number reported in
the README traces back to a few dozen lines of tested code. Sequences of unequal length
are padded and processed as one batch: a padded step gets a neutral emission (b = 1),
which leaves the recursions unchanged, so no special casing is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class Categorical:
    """Discrete emissions: probs[k, v] = P(observation v | state k)."""

    probs: np.ndarray

    @property
    def n_states(self) -> int:
        return self.probs.shape[0]

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            logp = np.log(self.probs)
        return np.moveaxis(logp[:, X], 0, -1)

    def update(self, X: np.ndarray, gamma: np.ndarray) -> None:
        n_symbols = self.probs.shape[1]
        counts = np.stack([gamma[np.equal(X, v)].sum(axis=0) for v in range(n_symbols)], axis=1)
        self.probs = counts / counts.sum(axis=1, keepdims=True)

    def sample(self, state: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.probs.shape[1], p=self.probs[state]))


@dataclass
class DiagGaussian:
    """Gaussian emissions with diagonal covariance and a variance floor.

    The floor matters with few training utterances: without it a state that sees a
    near-constant feature collapses to a spike and dominates the likelihood.
    """

    means: np.ndarray  # (K, D)
    variances: np.ndarray  # (K, D)
    var_floor: float | np.ndarray = 1e-3

    @property
    def n_states(self) -> int:
        return self.means.shape[0]

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        inv = 1.0 / self.variances  # (K, D)
        # Expand the quadratic form to avoid a (B, T, K, D) temporary.
        quad = (
            np.einsum("btd,kd->btk", X**2, inv)
            - 2.0 * np.einsum("btd,kd->btk", X, self.means * inv)
            + np.sum(self.means**2 * inv, axis=1)
        )
        log_det = np.sum(np.log(self.variances), axis=1) + X.shape[-1] * LOG_2PI
        return -0.5 * (quad + log_det)

    def update(self, X: np.ndarray, gamma: np.ndarray) -> None:
        w = gamma.sum(axis=(0, 1))[:, None]  # (K, 1)
        w = np.maximum(w, 1e-12)
        means = np.einsum("btk,btd->kd", gamma, X) / w
        second = np.einsum("btk,btd->kd", gamma, X**2) / w
        self.means = means
        self.variances = np.maximum(second - means**2, self.var_floor)

    def sample(self, state: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.means[state], np.sqrt(self.variances[state]))


def pad(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack variable-length sequences into a zero-padded batch plus their lengths."""
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    first = np.asarray(sequences[0])
    X = np.zeros((len(sequences), int(lengths.max()), *first.shape[1:]), dtype=first.dtype)
    for i, s in enumerate(sequences):
        X[i, : len(s)] = s
    return X, lengths


@dataclass
class Posteriors:
    gamma: np.ndarray  # (B, T, K) state posteriors, zero on padded steps
    xi_sum: np.ndarray  # (K, K) expected transition counts
    log_likelihood: np.ndarray  # (B,) log p(x_b) in nats


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(x, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return np.log(np.sum(np.exp(x - mx), axis=axis)) + np.squeeze(mx, axis=axis)


def forward_backward(
    log_b: np.ndarray, lengths: np.ndarray, pi: np.ndarray, A: np.ndarray
) -> Posteriors:
    """Forward-backward on a padded batch, robust to extreme likelihood ratios.

    The forward pass keeps normalised alpha in probability space but rescales each
    step in the log domain *after* weighting by the predicted state distribution, so a
    state that is unreachable (left-to-right models) cannot hide the reachable ones by
    having a huge emission density. The backward pass runs in the log domain.
    """
    B, T, K = log_b.shape
    valid = np.arange(T)[None, :] < lengths[:, None]
    # Time-major, contiguous; padded steps get a neutral emission (log b = 0).
    lb = np.ascontiguousarray(np.where(valid[..., None], log_b, 0.0).transpose(1, 0, 2))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_A = np.log(A)
        alpha = np.empty((T, B, K))
        log_c = np.empty((T, B))
        pred = np.broadcast_to(pi, (B, K))
        for t in range(T):
            if t > 0:
                pred = alpha[t - 1] @ A
            w = np.log(pred) + lb[t]
            s = w.max(axis=1, keepdims=True)
            if not np.all(np.isfinite(s)):
                raise FloatingPointError(f"observation at step {t} is impossible under the model")
            e = np.exp(w - s)
            tot = e.sum(axis=1, keepdims=True)
            alpha[t] = e / tot
            log_c[t] = (s + np.log(tot))[:, 0]

        log_beta = np.zeros((T, B, K))
        keep = (np.arange(T)[:, None] < (lengths - 1)[None, :])[..., None]
        for t in range(T - 2, -1, -1):
            v = lb[t + 1] + log_beta[t + 1] - log_c[t + 1][:, None]
            nxt = _logsumexp(log_A[None, :, :] + v[:, None, :], axis=2)
            log_beta[t] = np.where(keep[t], nxt, 0.0)

        log_alpha = np.log(alpha)
        gamma = np.exp(log_alpha + log_beta)
        gamma /= gamma.sum(axis=2, keepdims=True)
        gamma = gamma.transpose(1, 0, 2) * valid[..., None]

        v = lb[1:] + log_beta[1:] - log_c[1:, :, None]
        xi = np.exp(log_alpha[:-1, :, :, None] + log_A + v[:, :, None, :])
        xi_sum = np.einsum("tbij,tb->ij", xi, valid.T[1:].astype(float))

    log_lik = np.sum(np.where(valid.T, log_c, 0.0), axis=0)
    return Posteriors(gamma=gamma, xi_sum=xi_sum, log_likelihood=log_lik)


@dataclass
class HMM:
    """First-order HMM: start distribution pi, transitions A, emission model."""

    pi: np.ndarray
    A: np.ndarray
    emission: Categorical | DiagGaussian
    history: list[float] = field(default_factory=list)

    @property
    def n_states(self) -> int:
        return len(self.pi)

    def n_free_params(self) -> int:
        """Free parameters, counting only non-structural-zero transitions."""
        k = self.n_states
        trans = int(np.sum(self.A > 0)) - k
        start = int(np.sum(self.pi > 0)) - 1
        if isinstance(self.emission, Categorical):
            emis = k * (self.emission.probs.shape[1] - 1)
        else:
            emis = 2 * self.emission.means.size
        return trans + start + emis

    def posteriors(self, X: np.ndarray, lengths: np.ndarray) -> Posteriors:
        return forward_backward(self.emission.log_likelihood(X), lengths, self.pi, self.A)

    def score(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Log-likelihood of each sequence in the batch (nats)."""
        return self.posteriors(X, lengths).log_likelihood

    def fit(
        self, X: np.ndarray, lengths: np.ndarray, max_iter: int = 200, tol: float = 1e-6
    ) -> HMM:
        """Baum-Welch (EM). Stops when the relative log-likelihood gain falls below tol.

        Structural zeros in pi and A stay zero, so a left-to-right topology is kept.
        """
        self.history = []
        for _ in range(max_iter):
            post = self.posteriors(X, lengths)
            total = float(post.log_likelihood.sum())
            self.history.append(total)
            first = post.gamma[:, 0].sum(axis=0)
            self.pi = first / first.sum()
            rows = post.xi_sum.sum(axis=1, keepdims=True)
            self.A = np.where(rows > 0, post.xi_sum / np.maximum(rows, 1e-300), self.A)
            self.emission.update(X, post.gamma)
            if len(self.history) > 1:
                gain = self.history[-1] - self.history[-2]
                if gain < tol * abs(self.history[-1]):
                    break
        return self

    def viterbi(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Most likely state path for one sequence and its joint log-probability."""
        log_b = self.emission.log_likelihood(np.asarray(x)[None])[0]
        with np.errstate(divide="ignore"):
            log_a, log_pi = np.log(self.A), np.log(self.pi)
        T, K = log_b.shape
        delta = log_pi + log_b[0]
        back = np.zeros((T, K), dtype=np.int64)
        for t in range(1, T):
            cand = delta[:, None] + log_a
            back[t] = np.argmax(cand, axis=0)
            delta = cand[back[t], np.arange(K)] + log_b[t]
        path = np.empty(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path, float(delta[path[-1]])

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw (states, observations) of length n."""
        states = np.empty(n, dtype=np.int64)
        cum = np.cumsum(self.A, axis=1)
        u = rng.random(n)
        s = min(int(np.searchsorted(np.cumsum(self.pi), u[0], side="right")), self.n_states - 1)
        for t in range(n):
            if t > 0:
                s = min(int(np.searchsorted(cum[s], u[t], side="right")), self.n_states - 1)
            states[t] = s
        if isinstance(self.emission, Categorical):
            cum_b = np.cumsum(self.emission.probs, axis=1)
            v = rng.random(n)
            obs = (v[:, None] > cum_b[states]).sum(axis=1)
            obs = np.minimum(obs, self.emission.probs.shape[1] - 1)
        else:
            obs = np.stack([self.emission.sample(int(s), rng) for s in states])
        return states, obs
