"""Experiment 2: isolated-word recognition with one Gaussian HMM per word.

Protocol
- Each utterance becomes a sequence of feature frames (spectrum, filterbank or MFCC).
- One left-to-right HMM with diagonal Gaussian emissions is trained per word; a test
  utterance is assigned to the word whose model gives the highest log-likelihood.
- A 1-state model is a single Gaussian over all frames of a word: it ignores time
  order entirely and is the baseline that shows what the hidden dynamics add.
- Model selection (feature set x number of states) uses repeated stratified 5-fold
  cross-validation on the single-speaker fruit corpus only; ties go to fewer states.
- The selected feature set is then tested on the Free Spoken Digit Dataset (6
  speakers), which played no part in the choice: on its official split (speakers
  seen in training) and leave-one-speaker-out (unseen speaker), with and without
  per-utterance cepstral mean normalisation.
- Robustness: models trained on clean folds are scored on test folds with white
  noise added at a fixed signal-to-noise ratio.
- Features are standardised with statistics of the training data only.
"""

from __future__ import annotations

import numpy as np

from .data import Utterance, load_corpus, load_fsdd
from .features import FeatureConfig, extract
from .hmm import HMM, DiagGaussian, pad

FEATURES = ("spectrum", "filterbank", "mfcc")
STATES = (1, 3, 5, 8)
N_SPLITS = 5
N_REPEATS = 3
SNR_DB = (30, 20, 10, 5, 0)
VAR_FLOOR = 1e-2  # on standardised features; fixed a priori, not tuned
MAX_ITER = 50


def stratified_folds(labels: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    """Fold id per item, each class spread as evenly as possible across folds."""
    fold = np.empty(len(labels), dtype=np.int64)
    offset = 0
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        fold[idx] = (np.arange(len(idx)) + offset) % n_splits
        offset += len(idx)
    return fold


def init_left_to_right(seqs: list[np.ndarray], k: int) -> HMM:
    """Flat start: cut every utterance into k equal segments, one per state.

    The self-loop probability is set so the expected stay per state matches the
    average segment length, which is what the uniform segmentation assumes.
    """
    chunks = [[] for _ in range(k)]
    for s in seqs:
        for j, part in enumerate(np.array_split(s, k)):
            chunks[j].append(part)
    stacked = [np.concatenate(c) for c in chunks]
    means = np.stack([c.mean(axis=0) for c in stacked])
    variances = np.maximum(np.stack([c.var(axis=0) for c in stacked]), VAR_FLOOR)
    mean_len = np.mean([len(s) for s in seqs]) / k
    stay = 1.0 - 1.0 / max(mean_len, 1.0) if k > 1 else 1.0
    A = np.eye(k) * stay + np.eye(k, k=1) * (1.0 - stay)
    A[-1, -1] = 1.0
    pi = np.zeros(k)
    pi[0] = 1.0
    return HMM(pi=pi, A=A, emission=DiagGaussian(means, variances, VAR_FLOOR))


def train_models(seqs: list[np.ndarray], labels: np.ndarray, k: int) -> dict[str, HMM]:
    models = {}
    for word in np.unique(labels):
        word_seqs = [s for s, lab in zip(seqs, labels, strict=True) if lab == word]
        model = init_left_to_right(word_seqs, k)
        X, lengths = pad(word_seqs)
        models[str(word)] = model.fit(X, lengths, max_iter=MAX_ITER, tol=1e-4)
    return models


def classify(models: dict[str, HMM], seqs: list[np.ndarray]) -> np.ndarray:
    X, lengths = pad(seqs)
    names = list(models)
    scores = np.stack([models[w].score(X, lengths) for w in names], axis=1)
    return np.array(names)[np.argmax(scores, axis=1)]


def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = np.mean((signal - signal.mean()) ** 2)
    return signal + rng.normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), size=signal.shape)


class Standardizer:
    def __init__(self, seqs: list[np.ndarray]) -> None:
        frames = np.concatenate(seqs)
        self.mean = frames.mean(axis=0)
        self.std = frames.std(axis=0) + 1e-8

    def __call__(self, seqs: list[np.ndarray]) -> list[np.ndarray]:
        return [(s - self.mean) / self.std for s in seqs]


def featurize(
    utts: list[Utterance], cfg: FeatureConfig, names: tuple[str, ...] = FEATURES
) -> dict[str, list[np.ndarray]]:
    feats = [extract(u.signal, u.rate, cfg) for u in utts]
    return {name: [f[name] for f in feats] for name in names}


def mean_normalize(seqs: list[np.ndarray]) -> list[np.ndarray]:
    """Per-utterance mean removal (CMN): cancels a fixed channel or speaker offset."""
    return [s - s.mean(axis=0) for s in seqs]


def evaluate(
    feats: list[np.ndarray], labels: np.ndarray, train: np.ndarray, test: np.ndarray, k: int
) -> np.ndarray:
    """Train word models on ``train`` indices and return predictions for ``test``."""
    scale = Standardizer([feats[i] for i in train])
    models = train_models(scale([feats[i] for i in train]), labels[train], k)
    return classify(models, scale([feats[i] for i in test]))


def confusion_counts(truth: np.ndarray, pred: np.ndarray, words: np.ndarray) -> np.ndarray:
    out = np.zeros((len(words), len(words)), dtype=np.int64)
    np.add.at(out, (np.searchsorted(words, truth), np.searchsorted(words, pred)), 1)
    return out


def cross_validate(
    feats: list[np.ndarray], labels: np.ndarray, k: int, seed: int
) -> tuple[list[float], np.ndarray]:
    """Accuracy per fold over N_REPEATS x N_SPLITS folds, plus summed confusion counts."""
    rng = np.random.default_rng(seed)
    words = np.unique(labels)
    confusion = np.zeros((len(words), len(words)), dtype=np.int64)
    accs = []
    for _ in range(N_REPEATS):
        fold = stratified_folds(labels, N_SPLITS, rng)
        for f in range(N_SPLITS):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            pred = evaluate(feats, labels, tr, te, k)
            accs.append(float(np.mean(pred == labels[te])))
            confusion += confusion_counts(labels[te], pred, words)
    return accs, confusion


def noise_curve(
    utts: list[Utterance], feature: str, k: int, cfg: FeatureConfig, seed: int
) -> list[float]:
    """Mean accuracy at each SNR: train on clean folds, test on noisy copies.

    Folds are drawn exactly as in ``cross_validate`` (same seed), so the clean
    accuracy of this curve equals the grid entry; noise uses its own generator.
    """
    fold_rng, noise_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    labels = np.array([u.label for u in utts])
    clean = featurize(utts, cfg)[feature]
    accs = np.zeros((N_REPEATS * N_SPLITS, len(SNR_DB)))
    row = 0
    for _ in range(N_REPEATS):
        fold = stratified_folds(labels, N_SPLITS, fold_rng)
        for f in range(N_SPLITS):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            scale = Standardizer([clean[i] for i in tr])
            models = train_models(scale([clean[i] for i in tr]), labels[tr], k)
            for j, snr in enumerate(SNR_DB):
                noisy = [
                    extract(add_noise(utts[i].signal, snr, noise_rng), utts[i].rate, cfg)[feature]
                    for i in te
                ]
                accs[row, j] = np.mean(classify(models, scale(noisy)) == labels[te])
            row += 1
    return accs.mean(axis=0).tolist()


def summarize(accs: list[float]) -> dict:
    a = np.asarray(accs)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "folds": a.tolist()}


def fsdd_experiments(feature: str, k_selected: int, cfg: FeatureConfig) -> dict:
    utts = load_fsdd()
    labels = np.array([u.label for u in utts])
    speakers = np.array([u.speaker for u in utts])
    takes = np.array([u.take for u in utts])
    words = np.unique(labels)
    raw = featurize(utts, cfg, names=(feature,))[feature]
    variants = {"none": raw, "cmn": mean_normalize(raw)}

    # Official FSDD split: takes 0-4 of every speaker are the test set.
    test = np.flatnonzero(takes < 5)
    train = np.flatnonzero(takes >= 5)
    official = {}
    for k in sorted({1, k_selected}):
        pred = evaluate(raw, labels, train, test, k)
        official[str(k)] = {
            "accuracy": float(np.mean(pred == labels[test])),
            "confusion": confusion_counts(labels[test], pred, words).tolist(),
        }

    loso: dict[str, dict] = {}
    for norm, feats in variants.items():
        for k in STATES:
            per_speaker = {}
            confusion = np.zeros((len(words), len(words)), dtype=np.int64)
            for spk in np.unique(speakers):
                tr, te = np.flatnonzero(speakers != spk), np.flatnonzero(speakers == spk)
                pred = evaluate(feats, labels, tr, te, k)
                per_speaker[str(spk)] = float(np.mean(pred == labels[te]))
                confusion += confusion_counts(labels[te], pred, words)
            accs = list(per_speaker.values())
            loso[f"{norm}/{k}"] = {
                "normalization": norm,
                "states": k,
                "mean": float(np.mean(accs)),
                "min": float(np.min(accs)),
                "per_speaker": per_speaker,
                "confusion": confusion.tolist(),
            }
    return {
        "n_utterances": len(utts),
        "speakers": sorted(set(speakers.tolist())),
        "words": words.tolist(),
        "feature": feature,
        "official_split": {"n_train": len(train), "n_test": len(test), "results": official},
        "leave_one_speaker_out": loso,
    }


def run(seed: int = 0) -> dict:
    cfg = FeatureConfig()
    fruits = load_corpus("fruits")
    labels = np.array([u.label for u in fruits])
    feats = featurize(fruits, cfg)

    grid = {}
    for name in FEATURES:
        for k in STATES:
            accs, _ = cross_validate(feats[name], labels, k, seed)
            grid[f"{name}/{k}"] = {"feature": name, "states": k, **summarize(accs)}
    best_key = max(grid, key=lambda key: (grid[key]["mean"], -grid[key]["states"]))
    best = grid[best_key]
    _, confusion = cross_validate(feats[best["feature"]], labels, best["states"], seed)

    noise = {}
    curves = [
        (name, max(STATES, key=lambda s, n=name: grid[f"{n}/{s}"]["mean"])) for name in FEATURES
    ]
    curves.append((best["feature"], 1))
    for name, k in curves:
        key = name if k > 1 or name != best["feature"] else f"{name}_1state"
        noise[key] = {
            "feature": name,
            "states": k,
            "clean": grid[f"{name}/{k}"]["mean"],
            "by_snr": noise_curve(fruits, name, k, cfg, seed),
        }

    return {
        "fruits": {
            "n_utterances": len(fruits),
            "words": sorted(set(labels.tolist())),
            "frames_per_utterance": float(np.mean([len(f) for f in feats["mfcc"]])),
            "dims": {name: int(feats[name][0].shape[1]) for name in FEATURES},
            "grid": grid,
            "best": best_key,
            "confusion": confusion.tolist(),
        },
        "fsdd": fsdd_experiments(best["feature"], best["states"], cfg),
        "noise": {"snr_db": list(SNR_DB), "curves": noise},
        "protocol": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "states_grid": list(STATES),
            "var_floor": VAR_FLOOR,
            "max_iter": MAX_ITER,
        },
    }
