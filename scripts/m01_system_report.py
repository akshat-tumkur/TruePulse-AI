#!/usr/bin/env python3
"""Milestone 1: report the system facts needed to plan the Python environment.

Uses only the Python standard library, so it works before anything is installed:

    macOS / Linux:  python3 scripts/m01_system_report.py
    Windows:        py scripts\\m01_system_report.py

It only reads information. It installs nothing, writes no files and makes no
network requests. Paste the complete output into the chat (you may redact your
username from the paths it prints).

The compatibility notes reflect PyPI as checked on 2026-10-18 (section 7 of
docs/01_plan_and_architecture.md). Milestone 2 re-checks them on your machine.
"""

import os
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

if sys.version_info < (3, 8):  # noqa: UP036  (clear message instead of an ImportError)
    sys.exit(
        "This script needs Python 3.8 or newer, but it is running on Python "
        + platform.python_version()
        + ". Install Python 3.12 and run it again."
    )

from importlib import metadata  # noqa: E402  (needs the version check above)

CHECKED_ON = "2026-10-18"
REPO_ROOT = Path(__file__).resolve().parent.parent

# (operating system, interpreter architecture) pairs with MediaPipe 1.1.1 wheels on PyPI.
MEDIAPIPE_WHEEL_PLATFORMS = {
    ("Windows", "x86_64"),
    ("Windows", "arm64"),
    ("Darwin", "arm64"),
    ("Linux", "x86_64"),
    ("Linux", "aarch64"),
}
MIN_GLIBC = (2, 28)  # MediaPipe's Linux wheels are tagged manylinux_2_28
RECOMMENDED_PYTHON = (3, 12)
TESTED_PYTHONS = {(3, 11), (3, 12), (3, 13)}
INTEL_MAC_PYTHON_RANGE = ((3, 9), (3, 12))  # MediaPipe 0.10.21, the last Intel-Mac release
MIN_FREE_DISK_GB = 30
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)
PACKAGES = ("mediapipe",) + OPENCV_DISTRIBUTIONS + (
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pandas",
    "pytest",
)


# --------------------------------------------------------------------------- probes


def first_line(cmd):
    """Return the first line a command prints, or None if it cannot be run."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0].strip() if lines else None


def describe_os(system):
    if system == "Darwin":
        return "macOS " + (platform.mac_ver()[0] or "(version unknown)")
    if system == "Windows":
        release, build = platform.win32_ver()[:2]
        try:
            # Older Pythons report Windows 11 as release "10"; builds >= 22000 are Windows 11.
            if int(build.split(".")[-1]) >= 22000:
                release = "11"
        except ValueError:
            pass
        return f"Windows {release or '?'} (build {build or '?'})"
    if system == "Linux":
        try:
            with open("/etc/os-release", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return "Linux (distribution unknown)"
    return f"{system} {platform.release()}"


def interpreter_arch(system):
    """CPU architecture of this Python build, which decides the wheels pip installs."""
    if struct.calcsize("P") * 8 == 32:
        return "32-bit"
    if system == "Windows":
        # platform.machine() describes Windows itself; sysconfig describes this Python.
        tag = sysconfig.get_platform()
        return {"win-amd64": "x86_64", "win-arm64": "arm64"}.get(tag, tag)
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "arm64" if system == "Darwin" else "aarch64"
    return machine or "unknown"


def mac_is_apple_silicon():
    """True on Apple Silicon (even under Rosetta), False on Intel Macs, None if unknown."""
    if first_line(["sysctl", "-in", "hw.optional.arm64"]) == "1":
        return True
    brand = first_line(["sysctl", "-n", "machdep.cpu.brand_string"])
    if brand is None:
        return None
    return "intel" not in brand.lower()


def glibc_version():
    name, version = platform.libc_ver()
    if name != "glibc" or not version:
        return None
    try:
        return tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return None


def egl_library():
    import ctypes.util

    return ctypes.util.find_library("EGL")


def python_environment():
    if sys.prefix != sys.base_prefix:
        return "virtual environment at " + sys.prefix
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_env and conda_prefix and Path(conda_prefix).resolve() == Path(sys.prefix).resolve():
        return f"conda environment '{conda_env}'"
    return "none (system or base interpreter)"


def free_disk_gb(path):
    """Return (free gigabytes, None) or (None, error message)."""
    try:
        return shutil.disk_usage(str(path)).free / 1e9, None
    except OSError as exc:
        return None, str(exc)


def tool_version(name, flag):
    path = shutil.which(name)
    if path is None:
        return None
    return first_line([path, flag]) or f"found at {path} (version unknown)"


def installed_packages():
    found = {}
    for dist in PACKAGES:
        try:
            found[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            found[dist] = None
        except Exception as exc:  # a broken install should be reported, not hidden
            found[dist] = f"error reading metadata: {exc}"
    return found


def collect():
    system = platform.system()
    info = {
        "system": system,
        "os": describe_os(system),
        "platform": platform.platform(),
        "arch": interpreter_arch(system),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "environment": python_environment(),
        "cpu_count": os.cpu_count(),
        "apple_silicon": mac_is_apple_silicon() if system == "Darwin" else None,
        "glibc": glibc_version() if system == "Linux" else None,
        "egl": egl_library() if system == "Linux" else None,
        "wsl": system == "Linux" and "microsoft" in platform.release().lower(),
        "git": tool_version("git", "--version"),
        "ffmpeg": tool_version("ffmpeg", "-version"),
        "ffprobe": tool_version("ffprobe", "-version"),
        "packages": installed_packages(),
    }
    info["free_gb"], info["disk_error"] = free_disk_gb(REPO_ROOT)
    return info


# --------------------------------------------------------------------------- verdicts


def compatibility_notes(info, py):
    """Turn the collected facts into (level, message) pairs.

    Levels: BLOCKER (the planned stack cannot install), WARNING (needs attention),
    NOTE (for information) and OK.
    """
    notes = []
    system, arch = info["system"], info["arch"]
    py_text = f"{py[0]}.{py[1]}"
    intel_mac = False

    if arch == "32-bit":
        notes.append(("BLOCKER", "This is a 32-bit Python. MediaPipe only publishes 64-bit "
                                 "wheels, so install 64-bit Python 3.12."))
    elif system == "Darwin" and arch == "x86_64":
        if info["apple_silicon"]:
            notes.append(("BLOCKER", "This Python is an Intel (x86_64) build running under "
                                     "Rosetta on an Apple Silicon Mac. MediaPipe 1.x has no "
                                     "Intel-Mac wheel, so install a native arm64 Python 3.12 "
                                     "(python.org universal2 installer, or Homebrew in "
                                     "/opt/homebrew)."))
        elif info["apple_silicon"] is False:
            intel_mac = True
            notes.append(("WARNING", "Intel Mac: the newest MediaPipe with an Intel-Mac wheel is "
                                     "0.10.21 (Python 3.9-3.12, NumPy below 2). Milestone 2 will "
                                     "pin that older stack."))
        else:
            notes.append(("WARNING", "Could not tell whether this Mac has Apple Silicon or an "
                                     "Intel CPU. Please tell me (Apple menu > About This Mac)."))
    elif (system, arch) not in MEDIAPIPE_WHEEL_PLATFORMS:
        notes.append(("BLOCKER", f"MediaPipe 1.1.1 has no wheel for {system} on {arch}. "
                                 "Tell me about this machine so we can find an alternative."))

    if intel_mac:
        low, high = INTEL_MAC_PYTHON_RANGE
        if low <= py <= high:
            notes.append(("NOTE", f"Python {py_text} is inside MediaPipe 0.10.21's supported range "
                                  "(3.9-3.12)."))
        else:
            notes.append(("BLOCKER", f"Python {py_text} is outside MediaPipe 0.10.21's supported "
                                     "range (3.9-3.12). Install Python 3.12."))
    elif py == RECOMMENDED_PYTHON:
        notes.append(("OK", f"Python {py_text} is the recommended version."))
    elif py in TESTED_PYTHONS:
        notes.append(("OK", f"Python {py_text} works with the planned packages (tested); 3.12 is "
                            "recommended."))
    else:
        notes.append(("WARNING", f"Python {py_text} has not been tested with this project. Please "
                                 "install Python 3.12 (3.11 and 3.13 were also tested)."
                                 ))

    if system == "Linux":
        if info["glibc"] is None:
            notes.append(("WARNING", "Could not detect glibc (for example on Alpine/musl). "
                                     "MediaPipe's Linux wheels need glibc 2.28 or newer."))
        elif info["glibc"] < MIN_GLIBC:
            major, minor = info["glibc"]
            notes.append(("BLOCKER", f"glibc {major}.{minor} is older than 2.28, which MediaPipe's "
                                     "Linux wheels need. Use a newer distribution (for example "
                                     "Ubuntu 20.04 or later)."))
        if not info["egl"]:
            notes.append(("WARNING", "libEGL was not found. MediaPipe needs it at runtime "
                                     "(Debian/Ubuntu: sudo apt install libegl1 libgles2)."))
        if info["wsl"]:
            notes.append(("NOTE", "WSL detected: WSL2 does not see USB webcams by default. "
                                  "Record videos on Windows and read the files from WSL, or "
                                  "run camera tests with Windows Python."))

    opencv = [dist for dist in OPENCV_DISTRIBUTIONS if info["packages"].get(dist)]
    if len(opencv) > 1:
        notes.append(("NOTE", f"This Python has several OpenCV packages ({', '.join(opencv)}). "
                              "They share the 'cv2' module and overwrite each other. Milestone 2 "
                              "uses a fresh virtual environment with only opencv-contrib-python, "
                              "the one MediaPipe depends on."))

    if info["free_gb"] is None:
        notes.append(("WARNING", f"Could not read free disk space: {info['disk_error']}"))
    elif info["free_gb"] < MIN_FREE_DISK_GB:
        notes.append(("WARNING", f"Only {info['free_gb']:.1f} GB free on this drive. Datasets "
                                 "need space (the specification puts FaceForensics++ at about "
                                 "10 GB, and uncompressed 640x480 30 fps video is about 1.7 GB "
                                 "per minute), so we will plan partial downloads."))

    if not info["git"]:
        notes.append(("WARNING", "git was not found on PATH. You need it to pull each "
                                 "milestone's changes."))
    if not info["ffprobe"]:
        notes.append(("NOTE", "ffprobe (part of FFmpeg) was not found. It is optional: "
                              "Milestone 3 can use it to cross-check the frame rate OpenCV "
                              "reports."))

    if not any(level in ("BLOCKER", "WARNING") for level, _ in notes):
        notes.append(("OK", "No blocking issues found for Milestone 2."))
    return notes


# --------------------------------------------------------------------------- output


def section(title):
    print()
    print(f"[{title}]")


def row(label, value):
    print(f"  {label:<31}: {value}")


def yes_no_unknown(value):
    return "unknown" if value is None else ("yes" if value else "no")


def print_report(info, notes):
    print("TruePulse-AI system report (Milestone 1)")
    print(f"Compatibility notes reflect PyPI as checked on {CHECKED_ON}.")

    section("Operating system")
    row("OS", info["os"])
    row("Platform", info["platform"])
    if info["system"] == "Darwin":
        row("Apple Silicon", yes_no_unknown(info["apple_silicon"]))
    if info["system"] == "Linux":
        row("glibc", ".".join(str(p) for p in info["glibc"]) if info["glibc"] else "not detected")
        row("libEGL (needed by MediaPipe)", info["egl"] or "not found")
        row("WSL", yes_no_unknown(info["wsl"]))

    section("Python")
    row("Version", f"{info['python']} ({info['implementation']})")
    row("Architecture of this Python", info["arch"])
    row("Executable", info["executable"])
    row("Environment", info["environment"])

    section("Hardware")
    row("Logical CPU cores", info["cpu_count"] or "unknown")
    if info["free_gb"] is None:
        row("Free disk (repository drive)", f"unknown ({info['disk_error']})")
    else:
        row("Free disk (repository drive)", f"{info['free_gb']:.1f} GB")

    section("Tools on PATH")
    for tool in ("git", "ffmpeg", "ffprobe"):
        row(tool, info[tool] or "not found")

    section("Packages already installed in this Python")
    for dist, version in info["packages"].items():
        row(dist, version or "not installed")

    section("Compatibility notes")
    for level, message in notes:
        print(textwrap.fill(message, width=88, initial_indent=f"  {level:<8} ",
                            subsequent_indent=" " * 11))

    print()
    print("Copy everything above, starting at 'TruePulse-AI system report', into the chat.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        # Never crash on a console that cannot print an unusual character in a path.
        sys.stdout.reconfigure(errors="backslashreplace")
    info = collect()
    print_report(info, compatibility_notes(info, sys.version_info[:2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
