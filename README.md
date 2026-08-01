# land-ct-full

Faithful reimplementation of **LAND** (Oliveras et al., *Scientific Reports* 2026,
DOI [10.1038/s41598-026-51634-4](https://doi.org/10.1038/s41598-026-51634-4)) — an anatomically
guided latent diffusion model for high-resolution 3D chest CT synthesis.

**Goal:** reproduce the published pipeline as faithfully as possible — not improve or modify it —
on the full LIDC-IDRI dataset (1,010 patients), including the dedicated mask VAE (v2 scope; v1 was
a preprint-stage, 72-patient, lung-only-mask partial reproduction, kept read-only for reference,
see `docs/00_v1_reference.md`).

Every implementation choice answers one question: *is this what the authors actually did?*
Deviate only when physically impossible (11GB VRAM vs. the authors' A100), and log the deviation
as a decision in the project's memory pack.

## Status

| Stage | Status |
|---|---|
| 0. Repo + environment | ✅ Done |
| 1. Dataset acquisition (LIDC-IDRI, 1,010 patients) | ✅ Done, verified (133.15GB, 1,010/1,010 folders) |
| 2. Preprocessing (`src/data/preprocessing.py`) | ✅ Done, fully verified — 1,010/1,010, random spot-checked |
| 3. Nodule + lung mask extraction | ✅ Done (folded into Phase 2 — one script does both, no separate step) |
| 4. Image VAE training | 🟡 Setup done (`src/vae/`, `scripts/train_vae.ps1`) — not yet run, first real GPU test |
| 5. Mask VAE training | 🔴 Not started |
| 6. Latent precompute | 🔴 Not started |
| 7. Diffusion U-Net training | 🔴 Not started |
| 8. Sampling | 🔴 Not started |
| 9. Evaluation (FID, downstream Dice) | 🔴 Not started |

For the authoritative, actively maintained status/roadmap/decision log, see the project's
memory pack (kept separately from this repo). This repo's `docs/` folder holds point-in-time
setup and fidelity-verification records; where the two disagree, the memory pack wins.

## Pipeline

Mirrors the official repo's four-stage pipeline (`reference_official_repo/readme.md`):

1. **Preprocess** — `python -m src.data.preprocessing` (or `scripts\preproc_data.ps1`) converts
   raw LIDC-IDRI DICOMs into `.npy` CT volumes + conditioning masks (lung + nodule + texture, in
   one pass — the official pipeline doesn't have a separate mask-extraction stage).
2. **Train the image VAE** — compresses CT volumes into a latent space.
3. **Train the mask VAE** — compresses conditioning masks into a latent space.
4. **Train the diffusion U-Net** — in the joint latent space, conditioned on the frozen VAE
   checkpoints from steps 2–3.

## Repo layout

```
src/
  data/            preprocessing.py, utils_lidc3D.py — DICOM → npy (Stage 1)
  models/          image/mask VAE + U-Net architectures (Stage 2+, not yet implemented)
  training/        training loops (Stage 2+, not yet implemented)
  inference/       sampling (Stage 4, not yet implemented)
scripts/
  preproc_data.ps1 Windows wrapper for the preprocessing stage
docs/              setup + fidelity-verification records (see note above)
reference_official_repo/  the authors' official implementation, cloned as a read-only reference —
                           not built on top of, only verified against (git-ignored, not tracked
                           in this repo's own history)
data/, checkpoints/        git-ignored; populated locally by running the pipeline
```

## Fidelity notes

`src/data/preprocessing.py` and `src/data/utils_lidc3D.py` are adapted from the official repo's
`src/utils/preproc_lidc_npy.py` and `src/utils/utils_lidc3D.py` under Apache License 2.0 — see
each file's header for exactly what changed (Windows path fix, a trimmed dependency, a moved
import) and `reference_official_repo/LICENSE` / `NOTICE`. Everywhere the published paper's prose
and the official code disagree (mask VAE epoch count, one class weight, the exact HU-normalization
function), this project follows the code, since that's what actually produced the paper's
reported results — see `docs/03_verified_paper_repo_spec.md` for the full list, each traced
directly to a specific line of paper text or repo source.

## Machine

RTX 2080 Ti, 11GB VRAM, conda env `land-ct-v2`, torch 2.6.0+cu124. Shared Windows account with one
other project (LungDDPM) — `git config --local` only, never `--global`. See `docs/01_env_setup.md`.
