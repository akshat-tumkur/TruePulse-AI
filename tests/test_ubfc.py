"""Tests for truepulse.ubfc (synthetic ground-truth files)."""

import numpy as np
import pytest

from truepulse.ubfc import GroundTruthError, load_ubfc_ground_truth


def _write(path, rows):
    path.write_text("\n".join("  ".join(str(v) for v in row) for row in rows) + "\n")
    return path


def test_loads_three_rows(tmp_path):
    t = np.arange(300) / 30.0
    gt = load_ubfc_ground_truth(_write(tmp_path / "gt.txt", [np.sin(t), np.full(300, 72.5), t]))
    assert gt.n_samples == 300
    assert gt.sample_rate_hz == pytest.approx(30.0)
    assert gt.hr_bpm[0] == 72.5


def test_scientific_notation_and_extra_whitespace(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("  1.0e+00   2.5e-01\n\n7.2e+01 7.3e+01  \n0.0e+00 3.3e-02\n")
    gt = load_ubfc_ground_truth(path)
    assert gt.ppg.tolist() == [1.0, 0.25]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[1, 2], [70, 70]], "expected 3 non-empty rows"),
        ([[1, 2, 3], [70, 70], [0, 1, 2]], "different lengths"),
        ([[1, "x"], [70, 70], [0, 1]], "not a number"),
        ([[1, 2, 3], [70, 70, 70], [0, 2, 1]], "not strictly increasing"),
        ([[1], [70], [0]], "at least 2 samples"),
    ],
)
def test_bad_layouts_raise_clear_errors(tmp_path, rows, message):
    with pytest.raises(GroundTruthError, match=message):
        load_ubfc_ground_truth(_write(tmp_path / "gt.txt", rows))


def test_missing_file(tmp_path):
    with pytest.raises(GroundTruthError, match="not found"):
        load_ubfc_ground_truth(tmp_path / "nope.txt")
