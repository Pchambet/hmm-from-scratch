import numpy as np

from hmm_markov.words import classify, stratified_folds, train_models


def test_stratified_folds_balance_every_class():
    labels = np.repeat(["a", "b", "c"], [10, 10, 5])
    fold = stratified_folds(labels, 5, np.random.default_rng(0))
    for f in range(5):
        counts = [np.sum((fold == f) & (labels == c)) for c in "abc"]
        assert counts == [2, 2, 1]


def synthetic_word(order: list[float], rng: np.random.Generator) -> np.ndarray:
    """Frames that visit a fixed sequence of means, each for a random duration."""
    parts = [rng.normal(m, 0.3, size=(rng.integers(8, 15), 2)) for m in order]
    return np.concatenate(parts)


def test_hidden_states_capture_order_that_a_bag_of_frames_cannot():
    # "up" and "down" use exactly the same frames in opposite order: a 1-state model
    # sees identical frame distributions, a 3-state left-to-right HMM sees the order.
    rng = np.random.default_rng(0)
    make = {"up": [-2.0, 0.0, 2.0], "down": [2.0, 0.0, -2.0]}
    train = [(w, synthetic_word(m, rng)) for w, m in make.items() for _ in range(10)]
    test = [(w, synthetic_word(m, rng)) for w, m in make.items() for _ in range(20)]
    labels = np.array([w for w, _ in train])
    truth = np.array([w for w, _ in test])
    seqs, test_seqs = [s for _, s in train], [s for _, s in test]

    acc = {k: np.mean(classify(train_models(seqs, labels, k), test_seqs) == truth) for k in (1, 3)}
    assert acc[3] == 1.0
    assert acc[1] < 0.8
