"""Open videos, read their metadata and check the timing of every frame.

Every heart-rate value depends on the sampling rate, so this module never trusts a
single number: it compares the frame rate the file reports with the timestamps of
the decoded frames, and counts frames instead of relying on the header.

Frames are always read sequentially. Seeking with CAP_PROP_POS_FRAMES is not
used: in testing (OpenCV 5.0) a seek to frame 30 of an uncompressed AVI landed on
frame 26.

Note that container timestamps are often just frame_index / declared_fps, so they
cannot independently confirm the true capture rate; an external clock (such as
the time column of UBFC-rPPG's ground_truth.txt) can.

Never re-encode a video with OpenCV before analysing it: even its "uncompressed"
writer (FourCC 0) stores YUV and changes pixel values (grey 5 reads back as
(1, 4, 2)), which alters the colour signal rPPG measures.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# A frame interval this far from the median interval counts as irregular.
IRREGULAR_INTERVAL_TOLERANCE = 0.25


class VideoError(RuntimeError):
    """A video could not be opened or decoded."""


@dataclass
class VideoInfo:
    """What the file's header says. Nothing here is verified yet."""

    path: str
    size_bytes: int
    width: int
    height: int
    fps_reported: float
    frame_count_reported: int
    codec: str

    @property
    def duration_reported_s(self):
        if self.fps_reported <= 0:
            return float("nan")
        return self.frame_count_reported / self.fps_reported

    def to_dict(self):
        return {**asdict(self), "duration_reported_s": self.duration_reported_s}


@dataclass
class TimingReport:
    """Timing measured by decoding every frame."""

    frames_decoded: int
    first_timestamp_ms: float
    last_timestamp_ms: float
    median_interval_ms: float
    min_interval_ms: float
    max_interval_ms: float
    fps_from_timestamps: float
    irregular_intervals: int
    non_increasing_timestamps: int

    @property
    def duration_from_timestamps_s(self):
        # Last frame start minus first frame start, plus one typical frame interval.
        return (self.last_timestamp_ms - self.first_timestamp_ms + self.median_interval_ms) / 1000

    def to_dict(self):
        return {**asdict(self), "duration_from_timestamps_s": self.duration_from_timestamps_s}


def _open(path):
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"Video file not found: {path.resolve()}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"OpenCV could not open {path} (unsupported format or damaged file).")
    return cap


def _codec_name(fourcc):
    if fourcc == 0:
        return "uncompressed (FourCC 0)"
    text = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    return text if text.isprintable() else f"unknown (FourCC {fourcc})"


def probe_video(path):
    """Return the header metadata of a video as a VideoInfo."""
    cap = _open(path)
    try:
        return VideoInfo(
            path=str(path),
            size_bytes=Path(path).stat().st_size,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps_reported=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count_reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            codec=_codec_name(int(cap.get(cv2.CAP_PROP_FOURCC))),
        )
    finally:
        cap.release()


def iter_frames(path):
    """Yield (frame_index, timestamp_ms, frame_bgr) for every frame, in order.

    Frames are BGR uint8 arrays, as OpenCV decodes them.
    """
    cap = _open(path)
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, float(cap.get(cv2.CAP_PROP_POS_MSEC)), frame
            index += 1
    finally:
        cap.release()


def read_timestamps(path):
    """Return the timestamp (ms) of every frame, without decoding pixel data."""
    cap = _open(path)
    timestamps = []
    try:
        while cap.grab():
            timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
    finally:
        cap.release()
    if not timestamps:
        raise VideoError(f"No frames could be read from {path}.")
    return np.asarray(timestamps)


def analyse_timestamps(timestamps_ms):
    """Summarise frame timing. Needs at least two frames."""
    ts = np.asarray(timestamps_ms, dtype=float)
    if ts.size < 2:
        raise VideoError(f"Need at least 2 frames to measure timing, got {ts.size}.")
    intervals = np.diff(ts)
    increasing = intervals[intervals > 0]
    if increasing.size == 0:
        raise VideoError("Frame timestamps never increase; the video has no usable timing.")
    median = float(np.median(increasing))
    span = ts[-1] - ts[0]
    return TimingReport(
        frames_decoded=int(ts.size),
        first_timestamp_ms=float(ts[0]),
        last_timestamp_ms=float(ts[-1]),
        median_interval_ms=median,
        min_interval_ms=float(intervals.min()),
        max_interval_ms=float(intervals.max()),
        fps_from_timestamps=(ts.size - 1) / span * 1000 if span > 0 else float("nan"),
        irregular_intervals=int(
            np.sum(np.abs(intervals - median) > IRREGULAR_INTERVAL_TOLERANCE * median)
        ),
        non_increasing_timestamps=int(np.sum(intervals <= 0)),
    )


def evenly_spaced_indices(n_frames, n_samples):
    """Indices of `n_samples` frames spread evenly from the first to the last frame."""
    if n_frames <= 0 or n_samples <= 0:
        return []
    n_samples = min(n_samples, n_frames)
    return sorted({int(round(i)) for i in np.linspace(0, n_frames - 1, n_samples)})


def read_frames_at(path, indices):
    """Return {index: (timestamp_ms, frame_bgr)} for the requested frame indices.

    Reads sequentially (no seeking), grabbing frames it does not need without
    decoding them.
    """
    wanted = sorted(set(indices))
    if not wanted:
        return {}
    cap = _open(path)
    found = {}
    try:
        index = 0
        for target in wanted:
            while index < target:
                if not cap.grab():
                    return found
                index += 1
            ok, frame = cap.read()
            if not ok:
                return found
            found[target] = (float(cap.get(cv2.CAP_PROP_POS_MSEC)), frame)
            index += 1
    finally:
        cap.release()
    return found
