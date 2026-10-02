import numpy as np
import pytest

from hmm_markov.data import load_rain
from hmm_markov.hmm import HMM, Categorical
from hmm_markov.markov import OrderKChain
from hmm_markov.rain import (
    CHUNK,
    chain_conditional_loglik,
    chronological_split,
    hmm_conditional_loglik,
    ks_distance,
    survival,
)


def test_hmm_with_identity_emissions_scores_like_the_markov_chain():
    # An HMM whose states are observed directly *is* a first-order chain, so both
    # scoring paths must agree on the same steps: this ties the two model families
    # to one log-likelihood scale.
    chain = OrderKChain(1, alpha=0.0)
    chain.probs = np.array([0.02, 0.8])
    hmm = HMM(
        pi=np.array([0.9, 0.1]),
        A=np.array([[0.98, 0.02], [0.2, 0.8]]),
        emission=Categorical(np.eye(2)),
    )
    rng = np.random.default_rng(0)
    chunks = [chain.sample(500, rng) for _ in range(3)]
    assert hmm_conditional_loglik(hmm, chunks) == pytest.approx(
        chain_conditional_loglik(chain, chunks)
    )


def test_split_is_chronological_and_week_aligned():
    x = np.arange(10 * CHUNK + 17)
    split = chronological_split(x, 0.7)
    assert len(split.train) % CHUNK == 0
    assert split.train[-1] + 1 == split.test[0]


def test_survival_and_ks_by_hand():
    lengths = np.array([1, 2, 2, 5])
    np.testing.assert_allclose(survival(lengths, np.array([0, 2, 4, 5])), [1, 0.25, 0.25, 0])
    assert ks_distance(lengths, lengths) == 0
    assert ks_distance(np.array([1, 1]), np.array([3, 3])) == 1


def test_committed_rain_record_matches_documented_shape():
    wet = load_rain()
    assert len(wet) == 761 * 288
    assert wet.mean() == pytest.approx(0.0419, abs=1e-4)
