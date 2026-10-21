#!/usr/bin/env python3
"""Milestone 3: inspect a video's metadata, frame timing and sample frames.

    python scripts/m03_inspect_video.py data/raw/ubfc_rppg/subject1/vid.avi

If a UBFC-rPPG ground_truth.txt sits next to the video it is used automatically
to cross-check the timing (or pass --ground-truth PATH, or --no-ground-truth).

Writes to outputs/m03/ (ignored by git):
  <video_id>_frames.png   sample frames with their index and timestamp
  <video_id>_info.json    everything printed here, for later milestones

Exit code: 0 if the report was produced (warnings allowed), 1 if a check failed.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUT = REPO_ROOT / "outputs" / "m03"

FPS_TOLERANCE = 0.005  # 0.5 %: above this, two frame-rate estimates "disagree"
MIN_DURATION_S = 10.0  # below this, FFT bins are wider than 6 bpm
HR_BAND_HZ = (0.7, 4.0)  # 42-240 bpm, from the specification
FAKECATCHER_BAND_HZ = (0.7, 14.0)


class Findings:
    def __init__(self):
        self.items = []

    def add(self, status, message):
        self.items.append((status, message))
        print(f"  [{status}] {message}")

    @property
    def failed(self):
        return any(status == "FAIL" for status, _ in self.items)


def video_id_for(path):
    """Unique, readable ID. UBFC names every video vid.avi, so folders must be included."""
    path = Path(path).resolve()
    try:
        parts = path.relative_to(RAW_DIR).with_suffix("").parts
    except ValueError:
        parts = (path.parent.name, path.stem)
    return "_".join(p for p in parts if p)


def relative_difference(a, b):
    return abs(a - b) / b if b else float("inf")


def save_contact_sheet(path, frames, info, out_png, columns=4):
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(frames) / columns)
    aspect = info.height / info.width if info.width else 0.75
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.2 * columns, 3.2 * aspect * rows + 0.8), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, (index, (t_ms, frame)) in zip(axes.flat, sorted(frames.items()), strict=False):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {index}  |  t = {t_ms / 1000:.2f} s", fontsize=9)
    fig.suptitle(
        f"{Path(path).name}: {info.width}x{info.height}, {info.fps_reported:g} fps reported",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inspect a video (Milestone 3).")
    parser.add_argument("video", help="path to the video file")
    gt = parser.add_mutually_exclusive_group()
    gt.add_argument("--ground-truth", help="UBFC-rPPG ground_truth.txt to cross-check timing")
    gt.add_argument("--no-ground-truth", action="store_true", help="skip the cross-check")
    parser.add_argument("--frames", type=int, default=12, help="sample frames to save (12)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (outputs/m03)")
    args = parser.parse_args()

    from truepulse.ubfc import GroundTruthError, load_ubfc_ground_truth
    from truepulse.video import (
        VideoError,
        analyse_timestamps,
        evenly_spaced_indices,
        probe_video,
        read_frames_at,
        read_timestamps,
    )

    findings = Findings()
    video_id = video_id_for(args.video)
    out_dir = Path(args.out)
    print(f"TruePulse-AI video inspection (Milestone 3)\nVideo: {args.video}\nID:    {video_id}\n")

    try:
        info = probe_video(args.video)
        timestamps = read_timestamps(args.video)
        timing = analyse_timestamps(timestamps)
    except VideoError as exc:
        print(f"  [FAIL] {exc}")
        return 1

    print("Header (what the file claims)")
    print(f"  resolution           : {info.width} x {info.height}")
    print(f"  codec                : {info.codec}")
    print(f"  frame rate           : {info.fps_reported:.4f} fps")
    print(f"  frame count          : {info.frame_count_reported}")
    print(f"  duration             : {info.duration_reported_s:.2f} s")
    print(f"  file size            : {info.size_bytes / 1e6:.1f} MB\n")

    print("Measured (by reading every frame)")
    print(f"  frames decoded       : {timing.frames_decoded}")
    print(f"  frame rate           : {timing.fps_from_timestamps:.4f} fps (from timestamps)")
    print(
        f"  frame interval       : median {timing.median_interval_ms:.3f} ms, "
        f"min {timing.min_interval_ms:.3f}, max {timing.max_interval_ms:.3f}"
    )
    print(f"  duration             : {timing.duration_from_timestamps_s:.2f} s\n")

    print("Checks")
    if timing.frames_decoded == info.frame_count_reported:
        findings.add("PASS", f"decoded frame count matches the header ({timing.frames_decoded})")
    else:
        findings.add(
            "WARN",
            f"header says {info.frame_count_reported} frames but {timing.frames_decoded} were "
            "decoded; later milestones use the decoded count",
        )

    if info.fps_reported <= 0:
        findings.add("WARN", "the header has no frame rate; timestamps are the only source")
    elif relative_difference(timing.fps_from_timestamps, info.fps_reported) <= FPS_TOLERANCE:
        findings.add("PASS", "frame rate from timestamps agrees with the header (within 0.5 %)")
    else:
        findings.add(
            "WARN",
            f"frame rate from timestamps ({timing.fps_from_timestamps:.4f}) differs from the "
            f"header ({info.fps_reported:.4f}); trust the timestamps",
        )

    if timing.irregular_intervals or timing.non_increasing_timestamps:
        findings.add(
            "WARN",
            f"{timing.irregular_intervals} irregular frame intervals and "
            f"{timing.non_increasing_timestamps} non-increasing timestamps: variable frame "
            "rate or dropped frames. Milestone 6 resamples to an even time grid.",
        )
    else:
        findings.add("PASS", "frame intervals are regular (constant frame rate)")

    if info.fps_reported > 0:
        expected = [i * 1000 / info.fps_reported for i in range(len(timestamps))]
        if max(abs(a - b) for a, b in zip(timestamps, expected, strict=True)) < 1.0:
            findings.add(
                "NOTE",
                "timestamps are exactly frame_index / header fps, so they cannot independently "
                "confirm the true capture rate (an external clock such as ground truth can)",
            )

    duration = timing.duration_from_timestamps_s
    resolution_bpm = 60 / duration
    status = "PASS" if duration >= MIN_DURATION_S else "WARN"
    findings.add(
        status,
        f"{duration:.1f} s of video: whole-clip FFT bins are {1 / duration:.4f} Hz = "
        f"{resolution_bpm:.2f} bpm apart",
    )

    nyquist = timing.fps_from_timestamps / 2
    for name, (low, high) in (("heart-rate", HR_BAND_HZ), ("FakeCatcher", FAKECATCHER_BAND_HZ)):
        if high < nyquist:
            findings.add(
                "PASS",
                f"{name} band {low}-{high} Hz is below the Nyquist limit ({nyquist:.2f} Hz)",
            )
        else:
            findings.add(
                "WARN",
                f"{name} band {low}-{high} Hz exceeds the Nyquist limit ({nyquist:.2f} Hz); "
                "it cannot be used on this video",
            )

    gt_summary = None
    gt_path = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
    elif not args.no_ground_truth and (Path(args.video).parent / "ground_truth.txt").is_file():
        gt_path = Path(args.video).parent / "ground_truth.txt"
    if gt_path is not None:
        print(f"\nGround truth: {gt_path}")
        try:
            gt = load_ubfc_ground_truth(gt_path)
        except GroundTruthError as exc:
            findings.add("FAIL", str(exc))
        else:
            gt_summary = {
                "path": str(gt_path),
                "samples": gt.n_samples,
                "time_start_s": float(gt.time_s[0]),
                "time_end_s": float(gt.time_s[-1]),
                "sample_rate_hz": gt.sample_rate_hz,
                "hr_bpm_min": float(gt.hr_bpm.min()),
                "hr_bpm_max": float(gt.hr_bpm.max()),
            }
            print(
                f"  samples {gt.n_samples}, time {gt.time_s[0]:.3f} to {gt.time_s[-1]:.3f}, "
                f"rate {gt.sample_rate_hz:.4f} per second, "
                f"heart rate {gt.hr_bpm.min():.0f}-{gt.hr_bpm.max():.0f} bpm"
            )
            if gt.n_samples == timing.frames_decoded:
                findings.add("PASS", "one ground-truth sample per video frame")
            else:
                findings.add(
                    "WARN",
                    f"{gt.n_samples} ground-truth samples vs {timing.frames_decoded} frames; "
                    "Milestone 7 aligns them by time instead of by index",
                )
            if not 5 <= gt.sample_rate_hz <= 1000:
                findings.add(
                    "WARN",
                    f"ground-truth rate {gt.sample_rate_hz:.4f}/s is implausible for seconds; "
                    "the time row may use another unit",
                )
            elif gt.n_samples == timing.frames_decoded:
                diff = relative_difference(gt.sample_rate_hz, timing.fps_from_timestamps)
                status = "PASS" if diff <= FPS_TOLERANCE else "WARN"
                findings.add(
                    status,
                    f"ground-truth clock gives {gt.sample_rate_hz:.4f} fps vs "
                    f"{timing.fps_from_timestamps:.4f} from the video ({diff * 100:.2f} % apart)",
                )

    indices = evenly_spaced_indices(timing.frames_decoded, args.frames)
    frames = read_frames_at(args.video, indices)
    out_png = out_dir / f"{video_id}_frames.png"
    if len(frames) == len(indices) and frames:
        save_contact_sheet(args.video, frames, info, out_png)
        findings.add("PASS", f"saved {len(frames)} sample frames to {out_png}")
    else:
        findings.add("FAIL", f"could only read {len(frames)} of {len(indices)} sample frames")

    out_json = out_dir / f"{video_id}_info.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "header": info.to_dict(),
                "measured": timing.to_dict(),
                "ground_truth": gt_summary,
                "findings": [{"status": s, "message": m} for s, m in findings.items],
            },
            indent=2,
        )
    )
    print(f"  [INFO] wrote {out_json}")

    counts = {
        s: sum(1 for status, _ in findings.items if status == s) for s in ("PASS", "WARN", "FAIL")
    }
    print(f"\nSummary: {counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed")
    return 1 if findings.failed else 0


if __name__ == "__main__":
    sys.exit(main())
