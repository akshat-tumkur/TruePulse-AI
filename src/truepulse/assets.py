"""Download and verify external files the pipeline needs (currently the MediaPipe model).

Files are downloaded to a temporary ".part" file, checked against a pinned SHA-256
hash and only then moved into place, so a truncated or tampered download is never
mistaken for a good one.
"""

import hashlib
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

# MediaPipe Face Landmarker, float16 version 1 (verified 2026-10-18; see
# docs/01_plan_and_architecture.md, section 7).
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
FACE_LANDMARKER_FILENAME = "face_landmarker.task"


class AssetError(RuntimeError):
    """A required file could not be downloaded or failed verification."""


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssl_contexts():
    """Yield the system trust store first, then certifi's bundle if it is installed.

    python.org builds on macOS often have no usable system certificates until
    "Install Certificates.command" is run; certifi (a MediaPipe dependency)
    covers that case.
    """
    yield "system certificates", ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return
    yield "certifi bundle", ssl.create_default_context(cafile=certifi.where())


def _download(url, destination, timeout_s):
    errors = []
    for label, context in _ssl_contexts():
        try:
            with urllib.request.urlopen(url, timeout=timeout_s, context=context) as response:
                with open(destination, "wb") as out:
                    while True:
                        chunk = response.read(1 << 16)
                        if not chunk:
                            break
                        out.write(chunk)
            return
        except urllib.error.URLError as exc:
            errors.append(f"{label}: {exc.reason}")
            if not isinstance(exc.reason, ssl.SSLError):
                break  # a network error will not be fixed by another certificate bundle
        except OSError as exc:
            errors.append(f"{label}: {exc}")
            break
    raise AssetError(
        f"Could not download {url}\n  " + "\n  ".join(errors) + "\n"
        "Check your internet connection, firewall or proxy. You can also download the "
        f"file in a browser and save it as {destination.with_suffix('')}"
    )


def ensure_file(url, destination, expected_sha256, timeout_s=60):
    """Return `destination`, downloading it first if it is missing.

    Raises AssetError if the existing or downloaded file does not match
    `expected_sha256`. A mismatching existing file is never used or deleted
    silently; the error explains what to do.
    """
    destination = Path(destination)
    if destination.exists():
        actual = sha256_of(destination)
        if actual != expected_sha256:
            raise AssetError(
                f"{destination} exists but its SHA-256 is {actual}, expected "
                f"{expected_sha256}. Delete the file and run again to re-download it."
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        _download(url, partial, timeout_s)
        actual = sha256_of(partial)
        if actual != expected_sha256:
            raise AssetError(
                f"Downloaded {url} but its SHA-256 is {actual}, expected {expected_sha256}. "
                "The file may be truncated or the URL may now serve a different file."
            )
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def ensure_face_landmarker_model(models_dir, timeout_s=60):
    """Return the path of the verified MediaPipe Face Landmarker model in `models_dir`."""
    return ensure_file(
        FACE_LANDMARKER_URL,
        Path(models_dir) / FACE_LANDMARKER_FILENAME,
        FACE_LANDMARKER_SHA256,
        timeout_s=timeout_s,
    )
