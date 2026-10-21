# Milestone 2: environment setup

Goal: a reproducible Python environment that can read video and run the
MediaPipe face landmarker, verified by `scripts/m02_check_environment.py`.

## What you need

- 64-bit **Python 3.12** (3.11 and 3.13 also tested). Download the Windows or macOS
  installer from python.org. On Windows, tick "Add python.exe to PATH" (optional:
  the `py` launcher works either way).
- `git`, an internet connection (about 215 MB of packages, plus a 3.8 MB model
  file), and about 1 GB of free disk space for the environment (measured on
  Linux: 794 MB).
- **Not supported:** Intel Macs (the newest MediaPipe has no Intel-Mac build; tell
  me if that is your laptop and I will pin an older stack) and 32-bit Python.

## Install

Run everything from the repository root (the folder that contains `requirements.txt`).

**Windows (PowerShell)**

```powershell
git clone https://github.com/akshat-tumkur/TruePulse-AI.git
cd TruePulse-AI
git checkout claude/cool-edison-kjk0fi
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\m02_check_environment.py
```

If activation fails with "running scripts is disabled on this system", run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (applies to this
window only) and activate again.

**macOS (Apple Silicon) / Linux**

```bash
git clone https://github.com/akshat-tumkur/TruePulse-AI.git
cd TruePulse-AI
git checkout claude/cool-edison-kjk0fi
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/m02_check_environment.py
```

On Linux, MediaPipe also needs two system libraries:
`sudo apt install libegl1 libgles2` (Debian/Ubuntu).

Activate the environment (`.venv\Scripts\Activate.ps1` or
`source .venv/bin/activate`) every time you open a new terminal. Your prompt then
starts with `(.venv)`.

## Check it

```text
python scripts/m02_check_environment.py
python scripts/m02_check_environment.py --video data/raw/ubfc_rppg/subject1/vid.avi
python -m pytest
```

- The first command must end with `0 failed`. Expected lines: PASS for Python
  version, virtual environment, pinned packages, single OpenCV package, imports,
  video round trip, model download and Face Landmarker. The model is saved to
  `models/face_landmarker.task` and its SHA-256 is checked.
- The second command checks one of your UBFC-rPPG videos: it should report
  640x480 at 30 fps (per the dataset description) and find a face with 478
  landmarks. Use the real path of any UBFC video; folder names differ between
  dataset versions. Put the dataset under `data/raw/ubfc_rppg/` (ignored by git).
- Optional: `--camera 0` checks a webcam. You don't need one for this project,
  since you have UBFC-rPPG.
- `python -m pytest` runs the unit tests (no internet needed): expect `6 passed`.

## What `requirements.txt` is

It is a lock file generated from `requirements.in` with
`uv pip compile requirements.in --universal --python-version 3.11 -o requirements.txt`.
Every package, including indirect dependencies, is pinned to an exact version,
with markers where Windows, macOS, Linux or Python 3.11 need something different.
The last entry, `-e .`, installs this project's own code (`src/truepulse`) in
editable mode, so `import truepulse` works from any script.

Only `opencv-contrib-python` is listed. **Never install `opencv-python` or a
`-headless` variant in the same environment**: they share the `cv2` module and
overwrite each other. In my test, adding `opencv-python-headless` replaced
OpenCV 5.0 with 4.12 and downgraded NumPy without any error.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `[FAIL] Pinned packages: N of N ... missing` | The environment is not activated, or the install did not finish. Activate `.venv` and run `python -m pip install -r requirements.txt` again. |
| `[FAIL] Single OpenCV package` | Another OpenCV package got installed. Run the two commands the check prints: uninstall every OpenCV variant, then reinstall from `requirements.txt`. |
| `No matching distribution found for mediapipe==1.1.1` | Unsupported platform: 32-bit Python, an Intel Mac, or Python outside 3.11–3.13. Run `python scripts/m01_system_report.py` and send me the output. |
| `OSError: libEGL.so.1: cannot open shared object file` (Linux) | `sudo apt install libegl1 libgles2`. |
| `[FAIL] MediaPipe model download` with a certificate error (macOS) | Run "Install Certificates.command" in the Python folder under Applications. The downloader already falls back to certifi's certificates. |
| Model download blocked by a firewall | Download the model URL from `src/truepulse/assets.py` in a browser and save it as `models/face_landmarker.task`. The check verifies its SHA-256. |
| `[FAIL] Your video: ... could not open` | Wrong path (quote paths with spaces) or an unsupported codec. Send me the output. |

## Evidence to send if something fails

The full output of `python scripts/m02_check_environment.py`, which starts at
"TruePulse-AI environment check", and of `python scripts/m01_system_report.py`
run inside the activated environment.
