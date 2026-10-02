"""Frame-level acoustic features: log power spectrum, log mel filterbank, MFCC.

The three representations are compared in the word-recognition experiment. They differ
in dimension (129 / 26 / 12) and in how correlated their components are, which is
exactly what a diagonal-covariance Gaussian HMM is sensitive to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, rfft

EPS = 1e-10


@dataclass(frozen=True)
class FeatureConfig:
    """Defaults follow the course setup: 20 ms frames, 10 ms hop, 256-point FFT."""

    winlen: float = 0.02
    winstep: float = 0.01
    nfft: int = 256
    nfilt: int = 26
    numcep: int = 12
    preemph: float = 0.97
    ceplifter: int = 22


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(nfilt: int, nfft: int, rate: int) -> np.ndarray:
    """Triangular filters equally spaced on the mel scale, shape (nfilt, nfft // 2 + 1)."""
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(rate / 2), nfilt + 2)
    bins = np.floor((nfft + 1) * mel_to_hz(mels) / rate).astype(int)
    bank = np.zeros((nfilt, nfft // 2 + 1))
    for j in range(nfilt):
        lo, mid, hi = bins[j], bins[j + 1], bins[j + 2]
        for i in range(lo, mid):
            bank[j, i] = (i - lo) / max(mid - lo, 1)
        for i in range(mid, hi):
            bank[j, i] = (hi - i) / max(hi - mid, 1)
    return bank


def frame_signal(signal: np.ndarray, frame_len: int, step: int) -> np.ndarray:
    """Overlapping frames; the tail is zero-padded so every sample is covered."""
    n_frames = 1 + max(0, int(np.ceil((len(signal) - frame_len) / step)))
    padded = np.zeros((n_frames - 1) * step + frame_len)
    padded[: len(signal)] = signal
    idx = np.arange(frame_len)[None, :] + step * np.arange(n_frames)[:, None]
    return padded[idx]


def power_spectrum(signal: np.ndarray, rate: int, cfg: FeatureConfig) -> np.ndarray:
    emphasized = np.append(signal[0], signal[1:] - cfg.preemph * signal[:-1])
    frame_len = round(cfg.winlen * rate)
    step = round(cfg.winstep * rate)
    frames = frame_signal(emphasized, frame_len, step) * np.hamming(frame_len)
    return np.abs(rfft(frames, n=cfg.nfft)) ** 2 / cfg.nfft


def extract(signal: np.ndarray, rate: int, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    """All three feature sets for one utterance, each of shape (n_frames, dim)."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - signal.mean()
    pspec = power_spectrum(signal, rate, cfg)
    energies = pspec @ mel_filterbank(cfg.nfilt, cfg.nfft, rate).T
    log_fbank = np.log(np.maximum(energies, EPS))
    ceps = dct(log_fbank, type=2, axis=1, norm="ortho")[:, : cfg.numcep]
    lift = 1.0 + (cfg.ceplifter / 2.0) * np.sin(np.pi * np.arange(cfg.numcep) / cfg.ceplifter)
    return {
        "spectrum": 10.0 * np.log10(np.maximum(pspec, EPS)),
        "filterbank": 10.0 * np.log10(np.maximum(energies, EPS)),
        "mfcc": ceps * lift,
    }
