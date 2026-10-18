# TruePulse-AI

An experimental, learning-oriented **deepfake detector based on remote
photoplethysmography (rPPG)**. It recovers the faint pulse signal in the skin of a
face video. It then tests whether the pulse signals from different facial regions
behave the way a real person's do.

> **Research prototype, not a forensic tool.** A heart rate, or the lack of one,
> does not prove that a video is real or fake. High-quality deepfakes can inherit
> the pulse of the video they were made from (Seibold et al., 2025). Any accuracy
> we report applies only to the data it was measured on.

## Status

| Milestone | Status |
|---|---|
| 1. Specification, architecture, assumptions, project structure | ✅ Delivered, awaiting your review |
| 2. Python environment, dependencies, camera and video support | ⏳ Not started |
| 3. Load and inspect a video | ⏳ Not started |
| 4. Face landmarks and skin regions | ⏳ Not started |
| 5. Per-frame RGB signals to CSV | ⏳ Not started |
| 6. Filtering and FFT heart rate | ⏳ Not started |
| 7. Validation against a reference pulse | ⏳ Not started |
| 8. Multi-region agreement features | ⏳ Not started |
| 9. Labelled real/fake dataset | ⏳ Not started |
| 10. Baseline classifier with leakage-free splits | ⏳ Not started |
| 11. Evaluation and error analysis | ⏳ Not started |
| 12. Command-line inference with quality warnings | ⏳ Not started |
| 13. Reproducibility documentation | ⏳ Not started |

**Start here:** [docs/01_plan_and_architecture.md](docs/01_plan_and_architecture.md).
It covers the interpretation of the specification, architecture, risks,
assumptions, verified facts and the questions to answer before Milestone 2.

## What it will do when complete

1. Read a face video and detect the face.
2. Extract skin regions: forehead and both cheeks.
3. Average their colour in every frame to get time series.
4. Filter those series and estimate the heart rate.
5. Measure how consistently the regions' pulse signals agree over time.
6. Train a real-vs-fake classifier on those measurements.
7. Evaluate it on people and source videos never seen in training.

## Repository layout

```text
TruePulse-AI/
├── README.md              this file
├── prompt.txt             the project brief
├── Deepfake detection from your pulse_ how it works and how to build it.md
│                          the specification
├── docs/                  plan, dataset notes, results, reproducibility guide
├── src/truepulse/         library code (built up milestone by milestone)
├── scripts/               one runnable script per milestone (mNN_*.py)
├── tests/                 automated tests on synthetic signals
├── data/                  inputs and derived signals (ignored by git; see data/README.md)
├── models/                MediaPipe model file and trained classifiers (ignored by git)
└── outputs/               figures, reports and logs (ignored by git)
```

## Data, privacy and licences

- **Never commit videos, dataset files or model weights.** `.gitignore` blocks
  the contents of `data/`, `models/` and `outputs/`, and common video formats
  anywhere in the repository.
- **Faces and heart rates are biometric and health data.** Record only yourself,
  or people who have given informed consent.
- **Respect dataset licences.** Most deepfake and rPPG datasets require
  registration, are licensed for non-commercial research only and must not be
  redistributed. Each dataset's terms are documented before use (Milestone 9).

## Milestone 1 check

From the repository root:

```bash
# macOS / Linux
python3 scripts/m01_system_report.py

# Windows (PowerShell or Command Prompt)
py scripts\m01_system_report.py
```

The script uses only the Python standard library. It installs nothing, writes no
files and makes no network requests. Paste its full output into the chat; it
decides the exact setup steps for Milestone 2.

## Credits

- Specification: *Deepfake detection from your pulse: how it works and how to
  build it* by Hari (@harixganesh), included in this repository.
- Full references: section 9 of the
  [plan](docs/01_plan_and_architecture.md#9-references).
