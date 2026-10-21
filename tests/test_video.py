"""Tests for truepulse.video, using small synthetic videos written by OpenCV."""

import cv2
import numpy as np
import pytest

from truepulse.video import (
    VideoError,
    analyse_timestamps,
    evenly_spaced_indices,
    iter_frames,
    probe_video,
    read_frames_at,
    read_timestamps,
)

N_FRAMES, FPS, SIZE = 50, 25.0, (64, 48)


def _write(path, fourcc, fps=FPS, n=N_FRAMES):
    writer = cv2.VideoWriter(str(path), fourcc, fps, SIZE)
    assert writer.isOpened()
    for i in range(n):
        writer.write(np.full((SIZE[1], SIZE[0], 3), i * 5, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def raw_avi(tmp_path):
    """Lossless AVI: frame i has every pixel equal to 5 * i.

    FFV1 is used because OpenCV's "uncompressed" writer (FourCC 0) stores YUV and
    changes pixel values (grey 5 is read back as (1, 4, 2)), which would make
    exact frame checks impossible.
    """
    writer = cv2.VideoWriter(str(tmp_path / "t.avi"), cv2.VideoWriter_fourcc(*"FFV1"), FPS, SIZE)
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write lossless FFV1 video")
    writer.release()
    return _write(tmp_path / "lossless.avi", cv2.VideoWriter_fourcc(*"FFV1"))


def test_probe_reports_header(raw_avi):
    info = probe_video(raw_avi)
    assert (info.width, info.height) == SIZE
    assert info.frame_count_reported == N_FRAMES
    assert info.fps_reported == pytest.approx(FPS)
    assert info.codec.upper() == "FFV1"
    assert info.duration_reported_s == pytest.approx(N_FRAMES / FPS)


def test_codec_names(tmp_path):
    assert probe_video(_write(tmp_path / "m.avi", cv2.VideoWriter_fourcc(*"MJPG"))).codec == "MJPG"
    assert probe_video(_write(tmp_path / "r.avi", 0)).codec == "uncompressed (FourCC 0)"


def test_timestamps_and_timing(raw_avi):
    ts = read_timestamps(raw_avi)
    assert ts.size == N_FRAMES
    timing = analyse_timestamps(ts)
    assert timing.frames_decoded == N_FRAMES
    assert timing.fps_from_timestamps == pytest.approx(FPS, rel=1e-6)
    assert timing.median_interval_ms == pytest.approx(1000 / FPS)
    assert timing.irregular_intervals == 0
    assert timing.duration_from_timestamps_s == pytest.approx(N_FRAMES / FPS)


def test_iter_frames_yields_every_frame_in_order(raw_avi):
    values = [int(frame[0, 0, 0]) for _, _, frame in iter_frames(raw_avi)]
    assert values == [i * 5 for i in range(N_FRAMES)]


def test_read_frames_at_returns_the_requested_frames(raw_avi):
    # Seeking is unreliable (a seek to frame 30 landed on frame 26 in testing);
    # read_frames_at must return exactly the frames asked for.
    frames = read_frames_at(raw_avi, [30, 0, 49, 30])
    assert sorted(frames) == [0, 30, 49]
    for index, (t_ms, frame) in frames.items():
        assert int(frame[0, 0, 0]) == index * 5
        assert t_ms == pytest.approx(index * 1000 / FPS)


def test_read_frames_at_past_the_end_returns_what_exists(raw_avi):
    assert sorted(read_frames_at(raw_avi, [10, 500])) == [10]


def test_variable_frame_rate_is_flagged():
    ts = np.concatenate([np.arange(60) * 33.333, 60 * 33.333 + np.arange(1, 31) * 66.667])
    timing = analyse_timestamps(ts)
    assert timing.irregular_intervals == 30
    assert timing.non_increasing_timestamps == 0


def test_non_increasing_timestamps_are_counted():
    timing = analyse_timestamps([0.0, 40.0, 40.0, 80.0, 70.0, 120.0])
    assert timing.non_increasing_timestamps == 2


@pytest.mark.parametrize("timestamps", [[0.0], [5.0, 5.0, 5.0]])
def test_unusable_timing_raises(timestamps):
    with pytest.raises(VideoError):
        analyse_timestamps(timestamps)


def test_missing_file_raises(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        probe_video(tmp_path / "missing.avi")


def test_non_video_file_raises(tmp_path):
    path = tmp_path / "notes.avi"
    path.write_text("not a video")
    with pytest.raises(VideoError):
        read_timestamps(path)


def test_evenly_spaced_indices():
    assert evenly_spaced_indices(100, 5) == [0, 25, 50, 74, 99]
    assert evenly_spaced_indices(3, 10) == [0, 1, 2]
    assert evenly_spaced_indices(0, 5) == []
