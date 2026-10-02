"""Dataset loaders.

The rain record and the fruit-word recordings are small and committed. The Free Spoken
Digit Dataset is downloaded once (pinned release, checksum verified) into data/raw/.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, wavfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
FSDD_URL = "https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/refs/tags/v1.0.10"
# Digest of the 3000 recordings (sorted file names and bytes), not of the zip archive,
# which GitHub regenerates on the fly and does not guarantee byte-stable.
FSDD_SHA256 = "45a9b976ba3397a2fd5c70a9f2e43c34f442a3a6be47392ecfc41174b678f37e"
DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
STEP_MINUTES = 5
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES
WET_THRESHOLD = 0.1  # accumulation per 5-min step above which the step counts as wet


def load_rain(path: Path = DATA / "RR5MN.mat") -> np.ndarray:
    """Binary wet (1) / dry (0) series at 5-minute resolution.

    The file stores the accumulation ``RR5MN`` and a 1/2 indicator ``Support``; the
    indicator is exactly ``RR5MN > 0.1``, which is re-derived here and checked.
    """
    mat = loadmat(path)
    amount = mat["RR5MN"].squeeze()
    wet = (amount > WET_THRESHOLD).astype(np.int64)
    if not np.array_equal(wet, mat["Support"].squeeze().astype(np.int64) - 1):
        raise ValueError("wet/dry indicator does not match the stored Support series")
    return wet


@dataclass(frozen=True)
class Utterance:
    label: str
    speaker: str
    take: int
    rate: int
    signal: np.ndarray


def load_corpus(name: str, root: Path = DATA) -> list[Utterance]:
    """Single-speaker word recordings stored as ``<root>/<name>/<word>/<word><take>.wav``."""
    folder = root / name
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    out = []
    for word_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for wav in sorted(word_dir.glob("*.wav")):
            rate, signal = wavfile.read(wav)
            take = int(wav.stem.removeprefix(word_dir.name))
            out.append(Utterance(word_dir.name, "unknown", take, int(rate), signal.astype(float)))
    return out


def recordings_digest(recordings: Path) -> str:
    h = hashlib.sha256()
    for wav in sorted(recordings.glob("*.wav")):
        h.update(wav.name.encode())
        h.update(wav.read_bytes())
    return h.hexdigest()


def fetch_fsdd(dest: Path = RAW / "fsdd") -> Path:
    """Download FSDD v1.0.10 (16 MB, CC BY-SA 4.0) once; return the recordings folder."""
    recordings = dest / "recordings"
    # Checked on every call (about a second), so a partial or corrupted earlier download
    # is fetched again instead of being used unverified.
    if recordings.is_dir() and recordings_digest(recordings) == FSDD_SHA256:
        return recordings
    with urllib.request.urlopen(FSDD_URL, timeout=120) as resp:
        payload = resp.read()
    recordings.mkdir(parents=True, exist_ok=True)
    for stale in recordings.glob("*.wav"):  # leftovers would break the digest for good
        stale.unlink()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.namelist():
            if "/recordings/" in member and member.endswith(".wav"):
                (recordings / Path(member).name).write_bytes(zf.read(member))
    digest = recordings_digest(recordings)
    if digest != FSDD_SHA256:
        raise ValueError(f"FSDD recordings differ from the pinned release (sha256 {digest})")
    return recordings


def load_fsdd(recordings: Path | None = None) -> list[Utterance]:
    """FSDD recordings named ``{digit}_{speaker}_{take}.wav``, labelled by digit word."""
    recordings = recordings or fetch_fsdd()
    out = []
    for wav in sorted(recordings.glob("*.wav")):
        digit, speaker, take = wav.stem.split("_")
        rate, signal = wavfile.read(wav)
        out.append(
            Utterance(DIGITS[int(digit)], speaker, int(take), int(rate), signal.astype(float))
        )
    return out
