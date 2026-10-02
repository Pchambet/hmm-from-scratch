import numpy as np
import pytest

from hmm_markov.features import (
    FeatureConfig,
    extract,
    frame_signal,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
)


def test_mel_scale_round_trip_and_anchor():
    hz = np.array([0.0, 440.0, 1000.0, 4000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-9)
    assert hz_to_mel(1000.0) == pytest.approx(1000.0, abs=0.5)  # mel scale is anchored at 1 kHz


def test_filterbank_triangles_are_ordered_and_bounded():
    bank = mel_filterbank(26, 256, 8000)
    assert bank.shape == (26, 129)
    assert bank.min() >= 0 and bank.max() <= 1
    peaks = bank.argmax(axis=1)
    assert np.all(np.diff(peaks) >= 0)


def test_framing_covers_every_sample():
    frames = frame_signal(np.arange(10.0), frame_len=4, step=3)
    assert frames.shape == (3, 4)
    assert frames[-1].tolist() == [6.0, 7.0, 8.0, 9.0]


def test_pure_tone_peaks_at_its_frequency():
    rate, f0 = 8000, 1000.0
    t = np.arange(int(0.3 * rate)) / rate
    feats = extract(np.sin(2 * np.pi * f0 * t), rate, FeatureConfig())
    peak_bin = int(np.median(feats["spectrum"].argmax(axis=1)))
    assert peak_bin == round(f0 * 256 / rate)  # bin 32
    assert feats["filterbank"].shape[1] == 26
    assert feats["mfcc"].shape == (feats["spectrum"].shape[0], 12)
