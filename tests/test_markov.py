import numpy as np
import pytest

from hmm_markov.markov import (
    OrderKChain,
    fit_transition_matrix,
    geometric_survival,
    simulate_chain,
    spell_lengths,
    stationary_distribution,
)


def test_transition_counts_by_hand():
    seq = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    # from 0: 0->0 three times, 0->1 twice; from 1: 1->1 once, 1->0 once
    np.testing.assert_allclose(fit_transition_matrix(seq, 2), [[3 / 5, 2 / 5], [1 / 2, 1 / 2]])


def test_stationary_distribution_two_state_closed_form():
    p, q = 0.1, 0.5  # P(0->1), P(1->0); pi_1 = p / (p + q)
    A = np.array([[1 - p, p], [q, 1 - q]])
    np.testing.assert_allclose(stationary_distribution(A), [5 / 6, 1 / 6])


def test_spell_lengths_drop_censored_edges():
    seq = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0])
    assert spell_lengths(seq, 1).tolist() == [3, 1]
    assert spell_lengths(seq, 0).tolist() == [1]


def test_simulated_spells_match_geometric_law():
    p_leave_dry = 0.05
    A = np.array([[1 - p_leave_dry, p_leave_dry], [0.3, 0.7]])
    seq = simulate_chain(A, np.array([1.0, 0.0]), 200_000, np.random.default_rng(0))
    dry = spell_lengths(seq, 0)
    assert dry.mean() == pytest.approx(1 / p_leave_dry, rel=0.05)
    assert np.mean(dry > 30) == pytest.approx(geometric_survival(p_leave_dry, 30), abs=0.02)


def test_order_one_chain_equals_transition_mle_without_smoothing():
    seq = simulate_chain(
        np.array([[0.9, 0.1], [0.4, 0.6]]), np.array([0.5, 0.5]), 5000, np.random.default_rng(1)
    )
    chain = OrderKChain(1, alpha=0.0).fit([seq])
    np.testing.assert_allclose(chain.probs, fit_transition_matrix(seq, 2)[:, 1])


def test_order_two_chain_recovers_its_own_parameters():
    truth = OrderKChain(2)
    truth.probs = np.array([0.05, 0.6, 0.3, 0.9])  # context = 2 * x[t-2] + x[t-1]
    seq = truth.sample(100_000, np.random.default_rng(2))
    refit = OrderKChain(2).fit([seq])
    np.testing.assert_allclose(refit.probs, truth.probs, atol=0.02)


def test_chain_log_likelihood_by_hand():
    chain = OrderKChain(1, alpha=0.0)
    chain.probs = np.array([0.2, 0.7])
    total, n = chain.log_likelihood([np.array([0, 1, 1, 0])])
    assert n == 3
    assert total == pytest.approx(np.log(0.2) + np.log(0.7) + np.log(0.3))
