# CHANGELOG

## v2 — Phase 4: image VAE training setup (July 29, 2026)

- Full 1,010-patient preprocessing run completed and verified: 1,010/1,010 processed, 8 random
  patients spot-checked (correct `(256,256,256)` float32 `[0,1]` shape/range, correct mask class
  values). Phase 2 marked done in `README.md`.
- Added `src/vae/` — the official repo's entire `vae/` module, copied unmodified (Decision 0014
  reuse pattern): `autoencoder_kl.py`, `vae_train.py`, `vae_masks_train.py`, `vae_inference.py`,
  `vae_masks_inference.py`, `utils/{transforms,losses,masks_utils}.py`, and all
  `configs/config_vae*.json` files. Confirmed self-contained (no imports outside `src/vae/`)
  before copying, so no path/import adaptation was needed. Mask-VAE files came along now
  (Phase 5) to avoid re-copying the folder later.
- Added `scripts/train_vae.ps1`, a Windows-local wrapper for the official SLURM
  `scripts/train_vae.sh` — same config files, same hyperparameters, only the SLURM/job-scheduling
  scaffolding removed.
- Added `docs/04_vae_training_setup.md` documenting new pip dependencies not yet in `land-ct-v2`
  (`monai==1.5.0`, `wandb==0.21.0`, `lpips==0.1.4`, `tensorboard==2.19.0`), how to run, and one
  open question (exact `rand_zoom` augmentation range in `utils/transforms.py`, not yet verified
  against the paper).
- Not yet run: this is the first step in the whole pipeline that touches the GPU. VRAM behavior
  on the RTX 2080 Ti at 128³/batch-1 is unverified until actually run.

## v2 — Phase 2: preprocessing (July 29, 2026)

- Added `src/data/preprocessing.py` and `src/data/utils_lidc3D.py`, adapted from the official
  repo's `preproc_lidc_npy.py` / `utils_lidc3D.py` (Apache 2.0), verified line-by-line against
  the official source rather than reimplemented from the paper. See each file's header for the
  documented changes (Windows path fix, trimmed dependency, moved `rasterio` import).
- Added `scripts/preproc_data.ps1`, a Windows-local wrapper for the official SLURM
  `preproc_data.sh`, supporting both single-patient smoke-test runs and the full 1,010-patient run.
- Added `src/__init__.py` and `src/data/__init__.py` so `python -m src.data.preprocessing`
  resolves as a package.
- Added root `README.md` (previously missing) and corrected stale hardware/disk figures in
  `docs/01_env_setup.md` (GTX 1080 → RTX 2080 Ti; ~500GB unconfirmed → 851GB confirmed).
- Resolved two previously-open items in `docs/03_verified_paper_repo_spec.md`: `LATENT_SIZE`
  (= `latent_channels: 1` in the mask-VAE config) and the epochs-vs-steps question for the U-Net
  (`num_epochs=500` × 1,010 patients ≈ 505,000 steps, reconciling with the paper's reported
  500,000). Also added the exact channel widths for both VAEs, read directly from their configs.
- Not yet run against real DICOMs — next step is a 3-patient smoke test before the full run.

## v1 (July 25, 2026)
- Project restarted on new lab PC: full published paper (Scientific Reports, DOI
  10.1038/s41598-026-51634-4), full LIDC-IDRI dataset (1,010 patients), mask VAE included per
  Sir's instruction.
- BRAIN scaffolded from scratch (BOOT.md, README.md, 00_INDEX.md, decisions/, bugs/, chats/)
  before any code exists — earlier in the project lifecycle than v1's own pack was created.
- Decision 0001 logged: founding scope decision (full paper / full dataset / new machine).
- v1's pack (old project, preprint + 72 patients) kept as read-only reference — see
  docs/00_v1_reference.md for what carries over vs. what gets rebuilt.
