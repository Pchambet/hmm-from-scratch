import itertools

import numpy as np
import pytest

from hmm_markov.hmm import HMM, Categorical, DiagGaussian, forward_backward, pad


def random_categorical_hmm(rng, k=3, v=2):
    return HMM(
        pi=rng.dirichlet(np.ones(k)),
        A=rng.dirichlet(np.ones(k), size=k),
        emission=Categorical(rng.dirichlet(np.ones(v), size=k)),
    )


def brute_force(model, x):
    """Enumerate every state path: p(x), state posteriors and expected transitions."""
    k, T = model.n_states, len(x)
    B = model.emission.probs
    px, gamma, xi = 0.0, np.zeros((T, k)), np.zeros((k, k))
    best, best_path = -1.0, None
    for path in itertools.product(range(k), repeat=T):
        p = model.pi[path[0]] * B[path[0], x[0]]
        for t in range(1, T):
            p *= model.A[path[t - 1], path[t]] * B[path[t], x[t]]
        px += p
        for t in range(T):
            gamma[t, path[t]] += p
        for t in range(1, T):
            xi[path[t - 1], path[t]] += p
        if p > best:
            best, best_path = p, path
    return px, gamma / px, xi / px, np.array(best_path), np.log(best)


def test_forward_backward_matches_enumeration_on_padded_batch():
    rng = np.random.default_rng(1)
    model = random_categorical_hmm(rng)
    seqs = [rng.integers(0, 2, size=n) for n in (6, 4, 5)]
    X, lengths = pad(seqs)
    post = model.posteriors(X, lengths)
    xi_total = np.zeros((3, 3))
    for i, x in enumerate(seqs):
        px, gamma, xi, _, _ = brute_force(model, x)
        assert post.log_likelihood[i] == pytest.approx(np.log(px), abs=1e-10)
        np.testing.assert_allclose(post.gamma[i, : len(x)], gamma, atol=1e-10)
        assert np.all(post.gamma[i, len(x) :] == 0)
        xi_total += xi
    np.testing.assert_allclose(post.xi_sum, xi_total, atol=1e-10)


def test_viterbi_matches_enumeration():
    rng = np.random.default_rng(2)
    model = random_categorical_hmm(rng, k=3, v=3)
    x = rng.integers(0, 3, size=6)
    _, _, _, best_path, best_logp = brute_force(model, x)
    path, logp = model.viterbi(x)
    np.testing.assert_array_equal(path, best_path)
    assert logp == pytest.approx(best_logp)


def test_baum_welch_never_decreases_likelihood():
    rng = np.random.default_rng(3)
    truth = random_categorical_hmm(rng, k=2, v=3)
    seqs = [truth.sample(200, rng)[1] for _ in range(5)]
    model = random_categorical_hmm(np.random.default_rng(99), k=2, v=3)
    model.fit(*pad(seqs), max_iter=50, tol=0)
    assert np.all(np.diff(model.history) >= -1e-8)


def test_baum_welch_recovers_known_gaussian_hmm():
    rng = np.random.default_rng(4)
    truth = HMM(
        pi=np.array([0.5, 0.5]),
        A=np.array([[0.95, 0.05], [0.10, 0.90]]),
        emission=DiagGaussian(np.array([[-2.0, 0.0], [2.0, 1.0]]), np.full((2, 2), 0.5)),
    )
    seqs = [truth.sample(500, rng)[1] for _ in range(8)]
    start = HMM(
        pi=np.array([0.5, 0.5]),
        A=np.array([[0.7, 0.3], [0.3, 0.7]]),
        emission=DiagGaussian(np.array([[-0.5, 0.0], [0.5, 0.0]]), np.ones((2, 2))),
    )
    start.fit(*pad(seqs), max_iter=200, tol=1e-9)
    np.testing.assert_allclose(start.emission.means, truth.emission.means, atol=0.1)
    np.testing.assert_allclose(start.emission.variances, truth.emission.variances, atol=0.1)
    np.testing.assert_allclose(start.A, truth.A, atol=0.03)


def test_left_to_right_zeros_survive_training():
    rng = np.random.default_rng(5)
    A = np.array([[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]])
    model = HMM(
        pi=np.array([1.0, 0.0, 0.0]),
        A=A,
        emission=DiagGaussian(rng.normal(size=(3, 2)), np.ones((3, 2))),
    )
    seqs = [rng.normal(size=(n, 2)) for n in (20, 25, 30)]
    model.fit(*pad(seqs), max_iter=10)
    assert np.all(model.A[A == 0] == 0)
    assert model.pi.tolist() == [1.0, 0.0, 0.0]


def test_unreachable_state_with_huge_density_does_not_underflow():
    # State 2 explains the first frame far better than anything else but cannot be
    # reached at t = 0; a naive per-step max rescaling divides by zero here.
    model = HMM(
        pi=np.array([1.0, 0.0, 0.0]),
        A=np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]),
        emission=DiagGaussian(
            np.array([[0.0] * 50, [1.0] * 50, [30.0] * 50]), np.full((3, 50), 0.01)
        ),
    )
    x = np.full((4, 50), 30.0)
    post = model.posteriors(*pad([x]))
    assert np.isfinite(post.log_likelihood[0])
    np.testing.assert_allclose(post.gamma[0].sum(axis=1), 1.0)


def test_impossible_observation_raises():
    model = HMM(
        pi=np.array([1.0, 0.0]),
        A=np.eye(2),
        emission=Categorical(np.array([[1.0, 0.0], [0.0, 1.0]])),
    )
    with pytest.raises(FloatingPointError):
        forward_backward(
            model.emission.log_likelihood(np.array([[0, 1]])), np.array([2]), model.pi, model.A
        )
