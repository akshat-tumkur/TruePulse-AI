"""Read UBFC-rPPG ground truth.

DATASET_2 ships a `ground_truth.txt` next to each `vid.avi`. Secondary sources
describe it as three whitespace-separated rows: the PPG waveform, the heart rate
(bpm) and a time stamp in seconds, one value per sample. This layout has not been
confirmed against the official dataset description, so the reader validates it
and stops with a description of what it found if the file looks different.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class GroundTruthError(ValueError):
    """The ground-truth file does not have the expected layout."""


@dataclass
class UbfcGroundTruth:
    ppg: np.ndarray
    hr_bpm: np.ndarray
    time_s: np.ndarray

    @property
    def n_samples(self):
        return int(self.time_s.size)

    @property
    def sample_rate_hz(self):
        span = self.time_s[-1] - self.time_s[0]
        return (self.n_samples - 1) / span if span > 0 else float("nan")


def load_ubfc_ground_truth(path):
    """Load a DATASET_2 style ground_truth.txt (3 rows: PPG, HR, time in seconds)."""
    path = Path(path)
    if not path.is_file():
        raise GroundTruthError(f"Ground-truth file not found: {path.resolve()}")
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 3:
        raise GroundTruthError(
            f"{path}: expected 3 non-empty rows (PPG, heart rate, time), found {len(rows)}. "
            "Please send me the first 200 characters of each row."
        )
    try:
        ppg, hr, t = (np.asarray(row, dtype=float) for row in rows)
    except ValueError as exc:
        raise GroundTruthError(f"{path}: a value is not a number ({exc}).") from exc
    if not (ppg.size == hr.size == t.size):
        raise GroundTruthError(
            f"{path}: rows have different lengths (PPG {ppg.size}, HR {hr.size}, time {t.size})."
        )
    if t.size < 2:
        raise GroundTruthError(f"{path}: needs at least 2 samples, found {t.size}.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise GroundTruthError(
            f"{path}: the third row is not strictly increasing, so it is not a time axis "
            f"(first values: {t[:5].tolist()})."
        )
    return UbfcGroundTruth(ppg=ppg, hr_bpm=hr, time_s=t)
