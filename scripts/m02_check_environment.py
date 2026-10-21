#!/usr/bin/env python3
"""Milestone 2: verify that the Python environment can run the TruePulse pipeline.

Run inside the activated virtual environment, from the repository root:

    python scripts/m02_check_environment.py
    python scripts/m02_check_environment.py --video "data/raw/ubfc_rppg/subject1/vid.avi"
    python scripts/m02_check_environment.py --camera 0

Checks, in order: Python version, virtual environment, every package pinned in
requirements.txt, a single OpenCV package, imports, video write/read round trip,
MediaPipe model download (SHA-256 verified) and a real Face Landmarker run.
--video and --camera add checks for one of your videos and for a webcam.

Every check prints PASS, WARN, FAIL or SKIP. A failure does not stop later checks;
the exit code is 1 if anything failed. The model is saved to models/ (ignored by git).
"""

import argparse
import sys
import tempfile
import time
import traceback
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
MODELS_DIR = REPO_ROOT / "models"
SUPPORTED_PYTHONS = ((3, 11), (3, 12), (3, 13))
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)


class Report:
    def __init__(self):
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}

    def add(self, status, name, detail=""):
        self.counts[status] += 1
        print(f"[{status}] {name}")
        for line in str(detail).splitlines():
            print(f"         {line}")

    def run(self, name, check, *args):
        """Run one check; an unexpected exception becomes a FAIL with its traceback."""
        try:
            status, detail = check(*args)
        except Exception:  # report every failure instead of aborting the remaining checks
            status, detail = "FAIL", traceback.format_exc().rstrip()
        self.add(status, name, detail)
        return status


# --------------------------------------------------------------------------- checks


def check_python():
    version = sys.version_info[:2]
    bits = 64 if sys.maxsize > 2**32 else 32
    text = f"Python {sys.version.split()[0]}, {bits}-bit, {sys.executable}"
    if bits != 64:
        return "FAIL", text + "\nMediaPipe needs 64-bit Python."
    if version not in SUPPORTED_PYTHONS:
        return "FAIL", text + "\nSupported versions are 3.11, 3.12 and 3.13 (3.12 recommended)."
    return "PASS", text


def check_venv():
    if sys.prefix != sys.base_prefix:
        return "PASS", f"virtual environment at {sys.prefix}"
    return "WARN", (
        "Not running inside a virtual environment. Activate .venv first so the "
        "project's packages stay separate from the rest of your system."
    )


def check_pins():
    install_hint = "Run: python -m pip install -r requirements.txt"
    try:
        from packaging.markers import Marker
        from packaging.utils import canonicalize_name
    except ImportError:
        return "FAIL", "The project's packages are not installed in this Python. " + install_hint

    if not REQUIREMENTS.exists():
        return "FAIL", f"{REQUIREMENTS} not found."
    problems, checked = [], 0
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue  # comments, blank lines and "-e ." (the project itself)
        spec, _, marker = line.partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate():
            continue  # pinned for another platform or Python version
        name, _, wanted = spec.strip().partition("==")
        checked += 1
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name}: not installed (want {wanted})")
            continue
        if installed != wanted:
            problems.append(f"{name}: {installed} installed, {wanted} pinned")
    checked += 1  # this project itself
    try:
        metadata.distribution(canonicalize_name("truepulse"))
    except metadata.PackageNotFoundError:
        problems.append("truepulse (this project): not installed")
    if len(problems) > 5:
        return "FAIL", (
            f"{len(problems)} of {checked} pinned packages are missing or differ, "
            f"e.g. {problems[0]}. Is the virtual environment activated?\n" + install_hint
        )
    if problems:
        return "FAIL", "\n".join(problems) + "\n" + install_hint
    return "PASS", (
        f"all {checked - 1} pinned packages for this platform match "
        "requirements.txt; truepulse is installed"
    )


def check_single_opencv():
    found = []
    for dist in OPENCV_DISTRIBUTIONS:
        try:
            found.append(f"{dist}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            pass
    if len(found) == 1 and found[0].startswith("opencv-contrib-python=="):
        return "PASS", found[0]
    if not found:
        return "FAIL", "No OpenCV package installed. Run: python -m pip install -r requirements.txt"
    return "FAIL", (
        f"Found {', '.join(found)}. Only opencv-contrib-python may be installed: the others "
        "provide the same cv2 module and overwrite it (and may change numpy). Fix:\n"
        "  python -m pip uninstall -y " + " ".join(OPENCV_DISTRIBUTIONS) + "\n"
        "  python -m pip install -r requirements.txt"
    )


def check_imports():
    import cv2
    import matplotlib
    import mediapipe
    import numpy
    import pandas
    import scipy
    import sklearn

    import truepulse

    lines = [
        f"numpy {numpy.__version__}, scipy {scipy.__version__}, "
        f"scikit-learn {sklearn.__version__}, pandas {pandas.__version__}",
        f"matplotlib {matplotlib.__version__}, opencv {cv2.__version__}, "
        f"mediapipe {mediapipe.__version__}, truepulse {truepulse.__version__}",
    ]
    build = cv2.getBuildInformation()
    ffmpeg = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in build.splitlines()
            if ln.strip().startswith("FFMPEG:")
        ),
        "not listed",
    )
    lines.append(f"OpenCV FFmpeg video backend: {ffmpeg}")
    if not ffmpeg.startswith("YES"):
        return "FAIL", "\n".join(
            lines
        ) + "\nOpenCV was built without FFmpeg; videos cannot be read."
    return "PASS", "\n".join(lines)


def check_video_roundtrip():
    import cv2
    import numpy as np

    fps, n_frames, size = 30.0, 90, (160, 120)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "roundtrip.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            return "FAIL", "cv2.VideoWriter could not open an MP4 file for writing."
        for i in range(n_frames):
            writer.write(np.full((size[1], size[0], 3), (i * 2) % 256, dtype=np.uint8))
        writer.release()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return "FAIL", "cv2.VideoCapture could not open the file it just wrote."
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        decoded, last_ms = 0, None
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
            last_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.release()

    detail = (
        f"wrote {n_frames} frames at {fps:g} fps; read back {decoded} frames, "
        f"reported fps {reported_fps:g}, last timestamp {last_ms:.1f} ms"
    )
    if decoded != n_frames or abs(reported_fps - fps) > 0.01:
        return "FAIL", detail
    return "PASS", detail


def check_model():
    from truepulse.assets import FACE_LANDMARKER_SHA256, ensure_face_landmarker_model

    path = ensure_face_landmarker_model(MODELS_DIR)
    size_mb = path.stat().st_size / 1e6
    return "PASS", f"{path} ({size_mb:.2f} MB, SHA-256 {FACE_LANDMARKER_SHA256[:16]}... verified)"


def _landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    from truepulse.assets import ensure_face_landmarker_model

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_face_landmarker_model(MODELS_DIR))),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def check_landmarker():
    import numpy as np

    mp, landmarker = _landmarker()  # missing system libraries fail here, not at import
    with landmarker:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=blank), 0
        )
    if result.face_landmarks:
        return "FAIL", "The landmarker reported a face in an all-black frame."
    return (
        "PASS",
        "Face Landmarker created (VIDEO mode) and ran; blank frame -> 0 faces as expected",
    )


def check_user_video(video_path):
    import cv2

    path = Path(video_path)
    if not path.is_file():
        return "FAIL", f"File not found: {path.resolve()}"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return "FAIL", f"OpenCV could not open {path} (unsupported codec or damaged file)."
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or "unknown"
    ok, frame = cap.read()
    cap.release()
    lines = [f"{width}x{height}, reported {fps:g} fps, {count} frames reported, codec '{codec}'"]
    if not ok or frame is None:
        return "FAIL", "\n".join(lines + ["Could not decode the first frame."])

    mp, landmarker = _landmarker()
    with landmarker:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), 0
        )
    if not result.face_landmarks:
        return "WARN", "\n".join(lines + ["First frame decoded, but no face was detected in it."])
    lines.append(f"first frame decoded; face found with {len(result.face_landmarks[0])} landmarks")
    if fps <= 0:
        return "WARN", "\n".join(
            lines + ["The file reports no frame rate; Milestone 3 measures it."]
        )
    return "PASS", "\n".join(lines)


def check_camera(index, n_frames=60):
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return "FAIL", (
            f"Camera {index} could not be opened. Close other apps using it, check "
            "OS camera permissions, or try another index (--camera 1)."
        )
    frames, start = 0, time.perf_counter()
    first_shape = None
    try:
        while frames < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            first_shape = first_shape or frame.shape
            frames += 1
    finally:
        elapsed = time.perf_counter() - start
        cap.release()
    if frames == 0:
        return "FAIL", f"Camera {index} opened but returned no frames."
    detail = (
        f"{frames} frames of {first_shape[1]}x{first_shape[0]} in {elapsed:.1f} s "
        f"(about {frames / elapsed:.1f} fps delivered)"
    )
    return ("PASS" if frames == n_frames else "WARN"), detail


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", help="also check that this video opens and shows a face")
    parser.add_argument(
        "--camera", type=int, metavar="INDEX", help="also check webcam INDEX (usually 0)"
    )
    args = parser.parse_args()

    print("TruePulse-AI environment check (Milestone 2)\n")
    report = Report()
    report.run("Python version", check_python)
    report.run("Virtual environment", check_venv)
    pins_ok = report.run("Pinned packages", check_pins) == "PASS"
    report.run("Single OpenCV package", check_single_opencv)
    imports_ok = report.run("Imports", check_imports) == "PASS"

    if imports_ok:
        report.run("Video write/read round trip", check_video_roundtrip)
        model_ok = report.run("MediaPipe model download", check_model) == "PASS"
        if model_ok:
            report.run("Face Landmarker", check_landmarker)
        else:
            report.add("SKIP", "Face Landmarker", "needs the model")
        if args.video:
            if model_ok:
                report.run(f"Your video: {args.video}", check_user_video, args.video)
            else:
                report.add("SKIP", f"Your video: {args.video}", "needs the model")
    else:
        for name in ("Video write/read round trip", "MediaPipe model download", "Face Landmarker"):
            report.add("SKIP", name, "imports failed")
    if args.camera is not None:
        if imports_ok:
            report.run(f"Camera {args.camera}", check_camera, args.camera)
        else:
            report.add("SKIP", f"Camera {args.camera}", "imports failed")
    if not pins_ok:
        print(
            "\nNote: package versions differ from requirements.txt, so results may not match mine."
        )

    c = report.counts
    summary = ", ".join(
        f"{c[k]} {label}"
        for k, label in (
            ("PASS", "passed"),
            ("WARN", "warnings"),
            ("FAIL", "failed"),
            ("SKIP", "skipped"),
        )
    )
    print(f"\nSummary: {summary}")
    print("Copy everything above, starting at 'TruePulse-AI environment check', into the chat.")
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
