# Milestone 1: specification analysis, architecture and plan

| | |
|---|---|
| **Project** | TruePulse-AI: rPPG-based deepfake detection, an experimental research prototype |
| **Specification** | [Deepfake detection from your pulse: how it works and how to build it](../Deepfake%20detection%20from%20your%20pulse_%20how%20it%20works%20and%20how%20to%20build%20it.md), by Hari (@harixganesh) |
| **Brief** | [prompt.txt](../prompt.txt) |
| **Status** | Delivered 2026-10-18; waiting for your review before Milestone 2 |

> **This is a learning and research prototype, not a forensic authentication tool.**
> A pulse (or its absence) does not prove that a video is real or fake, and every
> result we measure applies only to the data it was measured on.

Contents:
[1. Interpretation](#1-interpretation-of-the-specification) ·
[2. Architecture](#2-architecture-and-data-flow) ·
[3. Structure](#3-project-structure) ·
[4. Milestones](#4-milestone-plan) ·
[5. Risks](#5-key-technical-risks) ·
[6. Assumptions](#6-assumptions) ·
[7. Verified facts](#7-verified-facts-checked-2026-10-18) ·
[8. Questions for you](#8-information-needed-from-you) ·
[9. References](#9-references)

---

## 1. Interpretation of the specification

### 1.1 What the specification says

- **Physics.** Each heartbeat briefly increases the blood volume under the skin.
  Blood absorbs light, so the skin reflects slightly less, most visibly in the
  green channel. Measuring this with light is photoplethysmography (PPG). Doing
  it from ordinary video, without contact, is remote PPG (rPPG).
- **Heart-rate recipe for one clip.** Face landmarks → skin patches (forehead,
  cheeks) → one mean colour value per patch per frame → band-pass 0.7–4 Hz
  (42–240 bpm) → Fourier transform → strongest frequency × 60 = beats per
  minute. A 10 s clip at 30 fps gives 300 samples and a frequency step of
  0.1 Hz (6 bpm).
- **Detector idea.** The detector follows FakeCatcher (Ciftci, Demir & Yin, 2020), which Intel made a product in 2022.
  On a real face the pulse signals of different regions rise and fall together.
  On a synthesised face they are less consistent. FakeCatcher uses 32 regions,
  keeps 0.7–14 Hz, and trains a classifier. It reports about 96 % accuracy on an
  older dataset and about 91 % on its authors' harder one.
- **Caveats the specification itself raises:**
  - Good face-swaps can copy the pulse of their source video (2025).
  - A BBC test found false alarms on real videos.
  - Compression, motion, uneven light and short clips degrade results.
  - Accuracy can differ by skin tone.
  - Testing on the people you trained on inflates the score.
- **Build recipe.** OpenCV (frames, real frame rate) → MediaPipe Face Landmarker
  (478 landmarks) → forehead and cheek mean RGB → `detrend`, Butterworth
  band-pass (`sosfiltfilt`), `rfft` → check bpm against a pulse oximeter or
  UBFC-rPPG (mean absolute error) → pairwise `coherence` of 3–6 patches per
  window → logistic regression or SVM, then a majority vote over windows →
  `GroupKFold` by person → test on unseen people, ideally from an unseen dataset.

### 1.2 How I read it: two systems sharing one pipeline

1. **A heart-rate estimator.** This is a measurement problem with physical ground
   truth from a contact pulse sensor. Output: bpm plus a quality indicator.
   Metric: mean absolute error (MAE) in bpm.
2. **A real-vs-fake classifier** built on rPPG-derived features, mainly how well
   the regions' signals agree. This is a statistical learning problem. Metrics:
   accuracy, precision, recall, F1 and ROC-AUC on people and source videos
   excluded from training.

System 1 is a prerequisite and a sanity check for system 2. If we cannot recover
a pulse from clean, real videos, the multi-region features are mostly noise.
Any classifier "success" would then have to come from something else, such as
compression, resolution, blending artefacts or the dataset source. But system 1
working does **not** show that system 2 works. They are evaluated separately.

The central claim is a **hypothesis to test, not an assumption**: *in the data we
have, measurable agreement between regional pulse signals differs between real
and fake videos.* High-quality fakes can carry a plausible, even correct, heart
rate (Seibold et al., 2025). So the detector will rely on spatial and temporal
agreement between regions, not on "is there a pulse?". If the features do not
separate the classes, that is a valid result and we will report it as such.

Two expectations follow, and we will **test** them rather than build them in:

- rPPG features should say more about **face swaps**, where the facial skin
  pixels are synthesised, than about **expression re-enactment**, where most
  skin pixels come from the original video. We will report each manipulation
  type separately.
- Many face-swap methods replace only the inner face. The forehead may still
  show the original person's skin and pulse. Forehead-versus-cheek disagreement
  could therefore be informative, or the forehead could make a fake look
  "alive". In Milestone 9 we check which regions each manipulation covers (some
  datasets ship manipulation masks; to be verified).

### 1.3 Where the specification simplifies, and what we will do

| The specification says | Why that is not enough on its own | What we will do | Milestone |
|---|---|---|---|
| Take the strongest FFT bin as the heart rate | A noise peak can win, and the bin gives no quality signal | Report a signal-to-noise ratio with every estimate and flag unreliable ones | 6 |
| A 10 s clip gives 6 bpm steps | The frequency step is 1 / duration | Print the bin width with every bpm. Make the window length configurable. Add sub-bin peak interpolation later, clearly labelled | 6 |
| Read the frame rate with `CAP_PROP_FPS` | Container metadata can be wrong, and phone videos often have a variable frame rate | Read a timestamp for every frame, detect irregular sampling and resample to a uniform grid when needed | 3, 6 |
| Use green first, then CHROM or POS | OpenCV decodes BGR while MediaPipe expects RGB. Mixing them up silently breaks CHROM and POS | Convert colours in exactly one place and test it with a synthetic frame of known colour | 4–5 |
| Use `scipy.signal.coherence` per window | Coherence is biased upward when a window has few Welch segments. With one segment it is exactly 1, even for pure noise (verified, §7) | Choose window and segment lengths that give several segments. Validate on synthetic signals. Add simpler agreement measures | 8 |
| Use the forehead and both cheeks | FakeCatcher used 32 regions, and swap masks may not cover the forehead | Start with 3 regions and extend to about 6 once stable. Check which regions each manipulation touches | 4, 8, 11 |
| Majority vote over windows | Gives a label, not a calibrated probability | Report the vote share and mean score as a **decision score**, explicitly not a probability | 10–12 |
| `GroupKFold` by person | A fake links two source videos (target and source identity) | Build groups from all linked source videos and assert that no group crosses the split | 9–10 |

### 1.4 What the specification leaves open (we decide with data, later)

- **Exact landmark indices for each region:** chosen by plotting index numbers on
  your frames, as the specification suggests (M4).
- **Window length and overlap for features:** depend on clip durations in the
  chosen dataset (M8–M9).
- **Frames with no face or a bad region:** interpolate only short gaps. Otherwise
  mark the window invalid. Always count and report affected frames (M5–M6).
- **Which manipulation types to train on:** start with face swaps and report
  every type separately (M9–M11).
- **Thresholds for "unreliable video" warnings:** derived from M7 and M11
  measurements, never guessed (M12).

---

## 2. Architecture and data flow

### 2.1 Pipeline

```text
video file (your recording or a dataset video)
   │  M3  video.py ............ open, metadata, per-frame timestamps, frame-rate checks
   ▼
frames (BGR, uint8) + timestamps
   │  M4  landmarks.py ........ MediaPipe Face Landmarker, VIDEO mode, 478 points
   │      roi.py .............. landmark polygons → forehead / left cheek / right cheek masks
   ▼
per-frame region masks + quality flags (face found, pixel count)
   │  M5  extract.py .......... mean R, G, B per region per frame
   ▼
data/interim/<video_id>.csv + .json   ◄── cache of the only slow step (face tracking)
   │  M6  signal_processing.py  uniform resampling, detrend, band-pass, FFT, SNR
   │      rppg.py ............. GREEN first; CHROM and POS later
   ├──────────────────────────►  M7  reference.py: bpm vs reference pulse → MAE
   ▼
filtered pulse signal per region
   │  M8  features.py ......... sliding windows → agreement features per window
   ▼
data/processed/features_<dataset>.csv   (one row per window, with label and group)
   │  M9  datasets.py ......... manifest: video, label, method, compression, groups
   │  M10 model.py ............ StandardScaler + LogisticRegression or SVM, group-aware CV
   ▼
models/<run_id>/   (model.joblib, config.json)
   │  M11 evaluate.py ......... window- and video-level metrics, error analysis
   ▼
outputs/reports/<run_id>/
   │  M12 cli.py + quality.py . `truepulse predict video.mp4` → label, decision score, warnings
```

### 2.2 Modules (`src/truepulse/`, planned)

Names may change as we build. Any change will be explained when it happens.

| Module | Milestone | Responsibility |
|---|---|---|
| `video.py` | 3 | Open a video. Report frame rate (reported and measured), frame count, size, duration and codec. Yield frames with timestamps. |
| `landmarks.py` | 4 | Wrap MediaPipe Face Landmarker: model download with checksum, BGR→RGB, increasing timestamps. |
| `roi.py` | 4 | Region definitions (landmark indices), pixel masks, region quality checks, overlay drawing. |
| `extract.py` | 5 | Per-frame mean R, G, B for every region → CSV, plus JSON metadata. |
| `signal_processing.py` | 6 | Sampling checks and resampling, detrend, band-pass, FFT heart rate, SNR, short-clip guards. |
| `rppg.py` | 6 | Turn RGB traces into a pulse trace: GREEN first, then CHROM and POS. |
| `reference.py` | 7 | Load reference pulses (UBFC-rPPG ground truth, your oximeter readings). |
| `features.py` | 8 | Windowed multi-region agreement features. |
| `datasets.py` | 9 | Manifests, labels, group IDs, split-integrity checks. |
| `model.py` | 10 | scikit-learn pipelines, group-aware cross-validation, training, saving. |
| `evaluate.py` | 11 | Metrics, confusion matrix, ROC, bootstrap confidence intervals, error analysis. |
| `quality.py`, `cli.py` | 12 | Video quality warnings; the `truepulse predict` command. |

### 2.3 Data contracts (the files that connect the stages)

- **Signal CSV** (`data/interim/`): one row per decoded frame. Columns:
  - `frame_idx`, `t_s`, `face_found`
  - per region: `<region>_r`, `<region>_g`, `<region>_b`, `<region>_npix`

  Channels are in RGB order. A missing face or region is written as an empty
  value (NaN) with its flag. Nothing is filled in at this stage; gap handling
  happens in M6 and is logged.
- **Metadata JSON** next to every signal CSV:
  - source video path, size and SHA-256 hash
  - reported and measured frame rate, frame counts, resolution, duration
  - region definitions and package versions
- **Feature CSV** (`data/processed/`): one row per window. Columns: `dataset`,
  `video_id`, `group_id`, `label`, `manipulation`, `compression`, `window_idx`,
  `t_start_s`, `t_end_s`, then the feature columns.
- **Manifest CSV** (`data/processed/`): one row per video. Columns: `video_id`,
  `path`, `label`, `dataset`, `manipulation`, `compression`, `subject_ids`,
  `source_video_ids`, `group_id`, `split`.

### 2.4 Design decisions

1. **Cache the slow step.** Face tracking runs once per video. Everything after
   it reads CSVs and re-runs in seconds, so you can experiment freely.
2. **Plain files** (CSV, JSON, PNG). You can open every intermediate result in a
   spreadsheet or text editor.
3. **One place** for colour conversion and one place for timestamps.
4. **Every number carries its context.** bpm comes with bin width and SNR.
   Metrics come with sample sizes and confidence intervals.
5. **Fail loudly.** Invalid input raises an error that says what to do. Skipped
   frames or videos are counted and reported, never dropped silently.
6. **Reproducibility.** Dependencies are pinned, random seeds are fixed, and
   package versions are recorded in every output's metadata.
7. **Separation.** Library code lives in `src/truepulse/`, one thin runnable
   script per milestone in `scripts/`, and pytest tests in `tests/`. The tests use
   synthetic signals, which are test inputs only and are never reported as results.

---

## 3. Project structure

The parts that exist now are marked **(M1)**. The rest is added by the milestone
shown.

```text
TruePulse-AI/
├── README.md                    overview, status, data rules                    (M1)
├── prompt.txt                   your project brief (unchanged)
├── Deepfake detection from your pulse_ how it works and how to build it.md
│                                the specification (unchanged)
├── .gitignore                   keeps videos, datasets and models out of git    (M1)
├── pyproject.toml               package metadata and pinned dependencies        (M2)
├── docs/
│   ├── 01_plan_and_architecture.md   this document                              (M1)
│   ├── datasets.md                   access, labels, licences                   (M9)
│   ├── results/                      heart-rate and classifier results          (M7, M11)
│   └── reproducibility.md            how to reproduce everything                (M13)
├── src/truepulse/               importable library code
│   ├── __init__.py                                                              (M1)
│   └── video.py, landmarks.py, roi.py, extract.py, ...                          (M3–M12)
├── scripts/                     one runnable script per milestone (mNN_*.py)
│   └── m01_system_report.py     collects the facts needed for Milestone 2      (M1)
├── tests/                       pytest tests on synthetic signals              (M2 onward)
├── data/                        contents ignored by git; see data/README.md     (M1)
│   ├── raw/                     untouched videos and dataset downloads
│   ├── interim/                 per-video RGB signals (CSV + JSON)
│   └── processed/               manifests, splits, feature tables
├── models/                      MediaPipe model file, trained classifiers (ignored by git)
└── outputs/                     figures, reports, logs (ignored by git)
```

---

## 4. Milestone plan

Each milestone ends with a checklist, exact test commands and the evidence to
share if something fails. Nothing moves forward without your confirmation.

| # | Goal | Planned deliverables | How we verify it |
|---|---|---|---|
| 1 | Turn the specification into a plan and create the structure | This document, repository skeleton, `scripts/m01_system_report.py` | You review the plan, run the report on your machine and answer §8 |
| 2 | Reproducible environment | `pyproject.toml` with pinned versions, setup commands for your OS, an environment check script | Every import works and prints its version. Exactly one OpenCV package is installed. A synthetic video is written and read back. The MediaPipe model downloads, its SHA-256 matches, and a Face Landmarker **is actually created**: missing system libraries only show up at that point, not at import. Your webcam opens and returns frames, if you use one. |
| 3 | Inspect a video | `video.py`, inspection script | Reported and measured frame rate, frame count, resolution and duration. Timestamp regularity. A contact sheet of sample frames. |
| 4 | Face and regions | `landmarks.py`, `roi.py`, visualisation script | Landmark-index plot. Region overlays across the clip. Detection rate and region pixel counts per frame. You confirm the placement visually. |
| 5 | RGB time series | `extract.py`, extraction script | One CSV row per decoded frame. Missing faces are explicit and counted. Raw R, G, B plots per region. |
| 6 | Heart rate | `signal_processing.py`, `rppg.py` (GREEN), tests | Synthetic tests: a known frequency gives a known bpm within one bin. Sampling rate and bin width are printed. Short or invalid signals are rejected with clear errors. Your video's result is compared with a pulse counted at your wrist. |
| 7 | Validate heart rate | `reference.py`, validation script, results write-up | Per-clip table, MAE (plus RMSE and correlation) against the reference, limitations written down. |
| 8 | Multi-region features | `features.py`, tests | Synthetic checks: identical signals give high agreement, independent noise gives low agreement with a known bias. No NaN or infinite values. Feature distributions plotted. |
| 9 | Dataset | `datasets.py`, manifest CSV, `docs/datasets.md` | Counts by label, method and compression. Group construction checked. Licence and access documented before any training. |
| 10 | Baseline classifier | `model.py`, training script | Automated leakage check. Cross-validated scores. Comparison with chance and with a quality-only baseline. |
| 11 | Evaluation | `evaluate.py`, evaluation report | Accuracy, precision, recall, F1, ROC-AUC and confusion matrix, at window and video level. Broken down by method and quality. Confidence intervals and error analysis. |
| 12 | Command-line inference | `cli.py`, `quality.py` | `truepulse predict video.mp4` returns a label, a decision score and warnings. Tested on held-out videos and on deliberately degraded ones. |
| 13 | Documentation | README, `docs/reproducibility.md` | A fresh clone reproduces the reported numbers by following the docs. |

---

## 5. Key technical risks

| # | Risk | Why it matters | Mitigation | Milestone |
|---|---|---|---|---|
| R1 | Weak pulse in compressed, low-resolution or moving video | The pulse changes the average skin colour by a tiny amount, often less than one intensity level out of 255. Compression and motion can erase it (the specification's first pitfall). | Validate on clean video first (UBFC-rPPG, your still recordings). Measure SNR for every estimate. Report results by quality level. | 5–7, 11 |
| R2 | Shortcut learning | Real and fake videos can differ in compression history, resolution or blending artefacts. A classifier will learn those instead of physiology. | Compare with a quality-only baseline (brightness, sharpness, bitrate, face size). Report per manipulation. Inspect which features matter. | 10–11 |
| R3 | Leakage between train and test | Clips of one person look alike, and a fake shares content with its source videos. Random splits inflate scores. | Groups are built from linked source videos. Use `StratifiedGroupKFold`. An automated check confirms no group is on both sides. | 9–10 |
| R4 | Fakes with an inherited pulse | Good face-swaps can carry the driving video's pulse (Seibold et al., 2025). | Never treat "pulse present" as evidence. Test spatial agreement. Report per manipulation method. | 8, 11 |
| R5 | Biased agreement measures | Coherence with few Welch segments is biased upward. With one segment it is exactly 1, even for pure noise (verified). | Choose windows that contain several segments. Validate with synthetic signals. Add simpler measures (correlation, per-region bpm agreement). | 8 |
| R6 | Wrong or irregular sampling rate | Every bpm value scales with the frame rate. Phone videos are often variable-frame-rate, and metadata can be wrong. | Per-frame timestamps, regularity checks, resampling to a uniform grid, optional ffprobe cross-check. | 3, 6 |
| R7 | Frame rate too low for the band | A 0.7–4 Hz band needs more than 8 fps. FakeCatcher's 0.7–14 Hz band needs more than 28 fps, so it cannot be used as-is on 25 fps video. | Check each band against the video's Nyquist frequency (half the frame rate) and stop with a clear message if it is exceeded. | 6, 8 |
| R8 | Colour-channel mix-ups | OpenCV gives BGR and MediaPipe expects RGB. CHROM and POS need the right channels. | One conversion point, plus a unit test with a frame of known colour. | 4–5 |
| R9 | Poor regions of interest | Hair on the forehead, glasses, beards, head turns and landmark jitter add non-skin pixels and noise. | Visual overlay checks, per-frame pixel counts, slightly shrunken polygons and a motion measure. Frames are flagged, not silently dropped. | 4–5 |
| R10 | Lighting | Mains flicker, camera auto-exposure and white balance, and uneven light add noise. The FakeCatcher authors' own false alarms came from movement and uneven light. | Detrending and brightness-jump detection. For your recordings: steady, diffuse light and fixed exposure if your camera allows it. | 5–6 |
| R11 | Short clips | The frequency step is 1 / duration: 10 s gives 6 bpm steps, 30 s gives 2 bpm. | Report the bin width with every bpm. Minimum-duration checks and warnings. | 6, 12 |
| R12 | Skin-tone performance gap | The rPPG signal is weaker on darker skin, and common datasets are not diverse. | Report it as a limitation. Stratify only if a dataset provides labels. Make no fairness claim without data. | 7, 11, 13 |
| R13 | Dataset access, size and licence | FaceForensics++ requires a request form, and the reply can take days. Downloads are large. Terms are typically non-commercial. | Request access now. Start with a subset. Document the terms before use. | 9 |
| R14 | Package churn | The current MediaPipe (1.1.1) has no `mp.solutions`, which most tutorials use. OpenCV is now 5.0. On Linux, MediaPipe needs system EGL libraries. | Use the Tasks API only. Pin exact versions. Record versions in every output. | 2, 4 |
| R15 | Compute time | Face tracking runs per frame on the CPU. | Cache per-video signals, process subsets first and report frames per second. | 5, 9 |
| R16 | Small test sets | A few dozen test videos give wide error bars. | Report sample sizes and bootstrap confidence intervals, resampling whole groups. | 7, 11 |

---

## 6. Assumptions

| # | Assumption | How and when we check it |
|---|---|---|
| A1 | You run the code on your own computer, and a CPU is enough. I write and test code in a Linux cloud sandbox, so camera tests and OS-specific steps happen on your machine. | Your system report (M1), environment checks (M2) |
| A2 | Each video shows one main face, mostly frontal, with the forehead and cheeks visible. | Detection rate and region overlays (M4) |
| A3 | Clips are at least 10 s long with a stable frame rate, typically 24–30 fps for webcams and web video. | Metadata and timestamp checks (M3) |
| A4 | A contact reference pulse is available: UBFC-rPPG and/or your pulse oximeter. | Your answers (§8), M7 |
| A5 | A labelled real/fake dataset with source-video information can be obtained. FaceForensics++ is the specification's default. | Access request now, inspection in M9 |
| A6 | The use is non-commercial learning or research, which matches typical dataset licences. | Your answer (§8), licence review (M9) |
| A7 | FakeCatcher's 32-region CNN is out of scope at first. We start with 3 regions and a linear model or SVM, as the specification suggests. | Revisit after M11 |

---

## 7. Verified facts (checked 2026-10-18)

The brief asks me not to assume that a package API, download URL or dataset link
is valid without checking. Everything below was checked on 2026-10-18 by running
the code in a Linux sandbox (Ubuntu 24.04, x86_64, Python 3.11, 3.12 and 3.13)
unless marked otherwise. Milestone 2 repeats the relevant checks on your machine.

**Python packages (PyPI)**

- `mediapipe` 1.1.1 (released 2026-10-14) has wheels for:
  - Windows x64 and ARM64
  - macOS Apple Silicon
  - Linux x86_64 and aarch64 (glibc 2.28 or newer)

  There is **no macOS Intel wheel**. The newest MediaPipe for Intel Macs is
  0.10.21, which supports Python 3.9–3.12 and requires NumPy < 2.
- `mediapipe` 1.1.1 depends on **`opencv-contrib-python`**, not `opencv-python`.
  Installing both makes them overwrite each other's `cv2` module. Uninstalling
  one then broke `import cv2` until the other was force-reinstalled
  (reproduced). Milestone 2 installs only `opencv-contrib-python`.
- Latest versions: NumPy 2.5.4 and SciPy 1.18.1 need **Python ≥ 3.12**.
  scikit-learn 1.9.1, pandas 3.0.6 and matplotlib 3.11.2 need Python ≥ 3.11.
  OpenCV is 5.0.0 (`opencv-contrib-python` 5.0.0.93).
- The full stack installed and passed the checks below on Python 3.13, 3.12
  and 3.11. On 3.11, pip resolves NumPy 2.4.6 and SciPy 1.17.1.
  **Recommendation: Python 3.12.**

**MediaPipe Face Landmarker**

- `mp.solutions` does not exist in 1.1.1. The Tasks API
  (`mediapipe.tasks.python.vision.FaceLandmarker`) works. Many tutorials still
  use `mp.solutions`. Their code will fail with `AttributeError`.
- In VIDEO mode it returned **478 landmarks** for a real face photo (Google's
  `business-person.png` test asset) and 0 faces for a blank frame. A repeated
  timestamp raised `ValueError: Input timestamp must be monotonically increasing`.
- On Linux, creating the landmarker failed with
  `OSError: libEGL.so.1: cannot open shared object file` until the system
  packages `libegl1` and `libgles2` were installed. `import mediapipe` alone
  succeeded, so an import check is not enough.
- Model URL:
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`.
  It is 3,758,596 bytes with SHA-256
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
  The `.../float16/latest/...` URL returned a file of the same size. We pin the
  versioned `1/` URL.

**OpenCV, SciPy, NumPy, scikit-learn**

- OpenCV 5.0.0 has the FFmpeg backend, and `CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`
  and `CAP_PROP_POS_MSEC` exist. A 300-frame synthetic MP4 written at 30 fps
  read back as 30.0 fps and 300 frames, with timestamps 0, 33.3, 66.7 ms and so on.
- SciPy 1.18.1:
  - The `butter` signature is `butter(N, Wn, btype='low', analog=False, output='ba', fs=None)`.
  - `sosfiltfilt` needs an input longer than its pad length, which is 27 samples
    for an order-4 band-pass (shorter input raises `ValueError`).
- The specification's arithmetic holds:
  - `rfftfreq(300, 1/30)` has 0.1 Hz (6 bpm) steps, and bin 12 = 1.2 Hz = 72 bpm.
  - A synthetic 1.2 Hz sine plus noise came out as 72 bpm after detrend and band-pass.
- **Coherence bias:** two *independent* noise signals gave mean coherence
  **1.000** with one Welch segment (`nperseg=300`) and 0.18 with `nperseg=64`.
- scikit-learn 1.9.1 has `GroupKFold(n_splits, *, shuffle, random_state)`,
  `StratifiedGroupKFold` and `GroupShuffleSplit`.

**Links in the specification.** My sandbox's network policy blocks arXiv, PMC,
Frontiers, Google Sites, GitHub web pages and github.io. I could not open those
pages directly. What I could corroborate through web search:

- *High-quality deepfakes have a heart!*: Seibold et al., Frontiers in Imaging,
  30 April 2025, DOI 10.3389/fimag.2025.1504551 (Fraunhofer HHI and Humboldt
  University Berlin). High-quality deepfakes carried heart-rate signals that
  correlated with their driving videos. The authors report preliminary evidence
  that the *spatial distribution* of blood flow may still reveal fakes.
- FaceForensics++: access is requested through a Google form, and the
  maintainers then email a download script. If there is no reply within a week,
  they say your email address is probably bouncing.
- UBFC-rPPG "DATASET_2", **from secondary sources only**:
  - 42 subjects, Logitech C920 webcam, 640×480, 30 fps
  - uncompressed 8-bit RGB video
  - reference pulse from a CMS50E oximeter; `ground_truth.txt` reportedly holds
    three lines (PPG signal, heart rate, time)

  Sources disagree on whether the download link is public or given on request.
  Uncompressed 640×480 RGB at 30 fps is about 1.7 GB per minute, so expect a
  large download.
- **Please confirm on your machine** that these links open: yarPPG deep dive,
  Verkruysse 2008 (PMC), FakeCatcher (arXiv), UBFC-rPPG page, rPPG-Toolbox.

---

## 8. Information needed from you

**Needed before Milestone 2 (blocking):**

1. **Your system report.** Run `scripts/m01_system_report.py` (commands in the
   README) and paste the whole output. It shows OS, CPU architecture, Python
   version and bitness, free disk space and tools, which decide the exact install
   steps. If you can't run it, tell me your OS and version, your CPU (Intel, AMD,
   Apple Silicon or ARM) and your `python --version`.
2. **Environment tool.** Plain `venv` + `pip` (my default), conda, or uv?
3. **Camera.** Will you record your own videos? With what: laptop webcam, USB
   webcam or phone? On WSL, note that WSL2 does not see USB webcams by default.

**Needed later, but start now because they take time:**

4. **Reference pulse (M7).** Do you have a fingertip pulse oximeter or a
   smartwatch? Can it export readings over time, or does it only display bpm?
5. **Datasets (M7, M9).**
   - Will you request FaceForensics++ access now? The reply can take days.
   - Do you already have any deepfake dataset?
   - Can you download UBFC-rPPG? It may be tens of GB; a few subjects are enough
     to start.
6. **Purpose.** Personal learning, academic work, or work for an employer or
   commercial use? Most deepfake and rPPG datasets are licensed for
   non-commercial research only.

**Optional:**

7. How comfortable are you with Python, signal processing and machine learning?
   I'll calibrate how much I explain.
8. Should I keep committing each milestone to this branch for you to pull, or
   open a pull request per milestone?

---

## 9. References

From the specification (links I could not open from my sandbox are marked †):

- † yarPPG deep dive: <https://samproell.github.io/yarppg/deepdive/>
- † Verkruysse, Svaasand & Nelson (2008), *Remote plethysmographic imaging using ambient light*: <https://pmc.ncbi.nlm.nih.gov/articles/PMC2717852/>
- † Ciftci, Demir & Yin, *FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals*: <https://arxiv.org/abs/1901.02212>
- † Seibold et al. (2025), *High-quality deepfakes have a heart!*, Frontiers in Imaging: <https://www.frontiersin.org/journals/imaging/articles/10.3389/fimag.2025.1504551/full>
- † UBFC-rPPG dataset: <https://sites.google.com/view/ybenezeth/ubfcrppg>
- † rPPG-Toolbox: <https://github.com/ubicomplab/rPPG-Toolbox>

Used in later milestones:

- de Haan & Jeanne (2013), *Robust pulse rate from chrominance-based rPPG* (CHROM), IEEE Transactions on Biomedical Engineering.
- Wang, den Brinker, Stuijk & de Haan (2017), *Algorithmic principles of remote PPG* (POS), IEEE Transactions on Biomedical Engineering.
- † Rössler et al. (2019), *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV: <https://github.com/ondyari/FaceForensics>
- MediaPipe Face Landmarker model (verified): `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
