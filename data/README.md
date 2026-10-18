# data/

Git ignores everything in this folder except this README and the `.gitkeep`
placeholders.

| Folder | Contents | First used in |
|---|---|---|
| `raw/` | Untouched inputs: your recordings and dataset downloads. Treat them as read-only and never edit files here. | M3, M7, M9 |
| `interim/` | Per-video RGB time series (CSV) with metadata (JSON). This caches the slow face-tracking step, so it is safe to delete and regenerate. | M5 |
| `processed/` | Dataset manifests, split files and feature tables for training. They are regenerated from `raw/` and `interim/`. | M8–M10 |

## Rules

1. **Never commit videos or per-person derived data.** Colour averages of a face
   are derived biometric data too.
2. **Record only yourself, or people who have given informed consent** to being
   filmed and having their pulse estimated.
3. **Respect dataset licences.** Most require registration, allow only
   non-commercial research and forbid redistribution. Their terms, access steps
   and citations go into `docs/datasets.md` (Milestone 9).
4. **Keep raw files unchanged.** Every derived file will record its source in its
   JSON metadata (path, size and SHA-256 hash), so results can be traced and
   reproduced.

## Suggested layout for raw inputs

```text
data/raw/
├── own/            your recordings, e.g. own/2026-10-20_desk_lamp_01.mp4
├── ubfc_rppg/      UBFC-rPPG, for heart-rate validation (Milestone 7)
└── faceforensics/  FaceForensics++, for real/fake training data (Milestone 9)
```
