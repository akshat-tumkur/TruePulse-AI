# Milestone 3: load and inspect a video

Goal: know exactly what a video contains (resolution, codec, frame count and, above
all, the real frame rate) before any signal processing. Every bpm value is
`peak frequency x 60`, and the frequency axis is built from the frame rate, so a
frame rate that is 2.5 % off makes every heart rate 2.5 % off.

## Run it

Activate the environment first (`.venv\Scripts\Activate.ps1` on Windows), then from
the repository root:

```powershell
python scripts\m03_inspect_video.py data\raw\ubfc_rppg\subject1\vid.avi
python -m pytest
```

If a `ground_truth.txt` sits next to the video, it is used automatically to
cross-check the timing. Options: `--ground-truth PATH`, `--no-ground-truth`,
`--frames N` (sample frames to save, default 12) and `--out DIR` (default
`outputs/m03`).

## What it does

1. Reads the header: resolution, codec, frame rate, frame count and duration.
   This is only what the file *claims*.
2. Reads the timestamp of every frame (without decoding the pixels) and
   measures the frame count, frame rate, the regularity of frame intervals and
   the duration.
3. Compares the two, and with UBFC-rPPG, compares both with the clock in
   `ground_truth.txt`.
4. Explains what the timing means for later milestones: the bpm resolution of
   a whole-clip FFT (`60 / duration`) and whether the heart-rate band
   (0.7–4 Hz) and FakeCatcher's band (0.7–14 Hz) fit below the Nyquist limit
   (half the frame rate).
5. Saves 12 evenly spaced frames as a PNG sheet, and everything printed as JSON.

Outputs (ignored by git), named after the video's path under `data/raw`:

```text
outputs/m03/ubfc_rppg_subject1_vid_frames.png
outputs/m03/ubfc_rppg_subject1_vid_info.json
```

## How to read the result

| Line | Meaning |
|---|---|
| `codec: uncompressed (FourCC 0)` | Normal for UBFC-rPPG, which stores raw video. This is why Milestone 2 printed `codec 'unknown'`. |
| `[NOTE] timestamps are exactly frame_index / header fps` | The file has no independent clock: its timestamps are computed from the declared frame rate. Only the ground-truth clock can confirm the real rate. |
| `ground-truth clock gives X fps vs Y` | The key check. If it says PASS, the video's 29.26 fps is confirmed by an external clock. If it is a WARN, tell me: Milestone 6 must then use the ground-truth time axis. |
| `[WARN] ... irregular frame intervals` | Variable frame rate or dropped frames (typical for phone videos). Milestone 6 resamples to an even time grid. |
| `whole-clip FFT bins are N bpm apart` | The heart-rate resolution of one FFT over the whole clip. Milestone 6 also uses shorter windows, with wider bins. |
| `FakeCatcher band ... exceeds the Nyquist limit` | At 25 fps or less, frequencies above 12.5 Hz cannot be measured, so that band cannot be used. |

**Check the PNG yourself:** skin colours should look natural (red and blue are
not swapped), the face should be visible in every sample frame, and frame
numbers and times should increase evenly.

## Known behaviour discovered in this milestone

- **Seeking is unreliable.** In testing, asking OpenCV 5.0 for frame 30 of an
  uncompressed AVI returned frame 26. All code reads frames sequentially.
- **OpenCV's "uncompressed" writer changes colours**: it stores YUV, so grey 5
  reads back as (1, 4, 2). Never re-encode a video with OpenCV before analysis.
  Your UBFC files are not affected; they were recorded by the dataset authors.
- **Variable-frame-rate files report a meaningless average.** A test file
  recorded at 30 fps and then 15 fps reported 22.9 fps. Per-frame timestamps
  reveal the truth, and the script flags it.

## Ground-truth format (to be confirmed with your files)

The reader expects the DATASET_2 layout described by secondary sources: three
rows (PPG waveform, heart rate in bpm, time in seconds), one value per sample.
If your file is different, the script prints a FAIL that says exactly what it
found. Send me that line.

## Checklist and evidence

- [ ] `python -m pytest` shows `27 passed`.
- [ ] The inspection of `subject1\vid.avi` ends with `0 failed`.
- [ ] The PNG shows natural colours and a visible face in all 12 frames.
- [ ] Optional: run it on two or three more subjects to see whether their frame
      rates differ.

Send me the full console output of each inspection, plus the `pytest` line. If
the ground-truth check fails, also send the output of:

```powershell
Get-Content data\raw\ubfc_rppg\subject1\ground_truth.txt -TotalCount 3 | ForEach-Object { $_.Substring(0, [Math]::Min(200, $_.Length)) }
```
