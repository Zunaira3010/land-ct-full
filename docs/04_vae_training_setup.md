# 04 — Image VAE Training Setup (Phase 4)

## What was added

`src/vae/` is the official repo's `src/vae/` directory, copied over unmodified
(Decision 0014 pattern: reuse, don't rewrite, when the goal is faithful
reproduction). Contents:

- `autoencoder_kl.py` — `AutoencoderKlReducedMaisi` model definition.
- `vae_train.py` — image VAE training loop (used now).
- `vae_masks_train.py` — mask VAE training loop (Phase 5, not yet).
- `vae_inference.py`, `vae_masks_inference.py` — inference/eval scripts (later).
- `utils/transforms.py`, `utils/losses.py`, `utils/masks_utils.py` — dependencies
  of the above, all self-contained within `src/vae/` (no imports outside this
  folder — confirmed by reading every import in every file before copying).
- `configs/config_vae.json`, `configs/config_vae_train.json` — architecture and
  training hyperparameters, **used exactly as shipped**, no edits:
  - Architecture: 3 resolution levels, channels `[32, 64, 128]`, 1 residual
    block/level, 4-channel latent, `in_channels`/`out_channels` = 1.
  - Training: 100 epochs, AdamW lr=1e-4, batch size 1, 128³ patches,
    `perceptual_weight` 0.3, `kl_weight` 1e-7, `adv_weight` 0.1, recon loss L1,
    AMP on.
- The other `configs/config_vae_masks*.json` files came along too since they're
  needed for Phase 5 (mask VAE) — no reason to re-copy the folder later.

`scripts/train_vae.ps1` — Windows/local equivalent of the official repo's
SLURM `scripts/train_vae.sh`. Same args, same config files, same hyperparameters.
Only the job-scheduling scaffolding (SLURM directives, `module load`, cluster
conda path) was removed — see the comment header in the script itself.

## New dependencies (not yet installed in `land-ct-v2`, as of Phase 2)

The official repo's `environment.yml` pins these; install into the existing
`land-ct-v2` env rather than creating a new one:

```powershell
conda activate land-ct-v2
pip install monai==1.5.0 wandb==0.21.0 lpips==0.1.4 tensorboard==2.19.0 diffusers==0.34.0
```

This list was verified by reading every `import`/`from` line in the actual
training path — `vae_train.py`, `autoencoder_kl.py`, `utils/transforms.py`,
`utils/losses.py` — not just `vae_train.py` alone. (First pass missed
`diffusers`, which `autoencoder_kl.py` needs for `ModelMixin`/`ConfigMixin`;
caught and fixed via the smoke test below, which is exactly what it's for.)

Note: `cv2`, `imageio`, `pandas`, and `scikit-learn` appear elsewhere in
`src/vae/` (`vae_inference.py`, `utils/masks_utils.py`) but are **not** needed
to run `vae_train.py` — don't install them preemptively; deal with them when
Phase 5 (mask VAE) or inference actually needs them.

`torch==2.6.0+cu124` is already installed and matches the official repo's pin —
no change needed there. `wandb` is imported unconditionally by `vae_train.py`
even when not used; `scripts/train_vae.ps1` and `scripts/smoke_test_vae.ps1`
both set `WANDB_MODE=disabled` so no login/account is required (same as the
original `.sh`).

## Running it

**Smoke test first, always** (same pattern as preprocessing — verify before committing GPU-hours):

```powershell
.\scripts\smoke_test_vae.ps1
```

This junctions 8 already-processed patients into `data\processed_smoke` (no data duplicated),
runs 2 epochs using `configs/config_vae_train_smoke.json` (identical to the real config except
`n_epochs: 2`), against the **real, unmodified** `config_vae.json` architecture. Checks: no CUDA
OOM, loss isn't NaN, a checkpoint actually saves, tensorboard logs reconstruction images. This is
also the first real read on peak VRAM at batch-1/128³ on the RTX 2080 Ti, before trusting it for
100 epochs.

**Then the real run:**

```powershell
.\scripts\train_vae.ps1
```

Defaults: `data\processed` as the dataset, 90/10 train/val split, checkpoints to
`checkpoints\vae\`, logs to `logs\vae\`.

**This is the first real VRAM test of the pipeline on the RTX 2080 Ti** — nothing
in Phase 2 touched the GPU. Batch size 1 at 128³ patches should fit in 11GB, but
watch the first iteration before backgrounding the job. Once it's stable,
background it the same way as `preproc_data.ps1` (see the script's header
comment for the exact `Start-Process` invocation).

## Open question — not yet resolved

`config_vae_train.json`'s `data_option.spacing_type` is `"rand_zoom"` with
`spacing: null`. This is a random-resample augmentation whose exact range isn't
pinned down in the paper's prose — need to check `utils/transforms.py`'s
`VAE_Transform` to confirm what range it actually applies before treating this
as fully verified against the paper (flagging here rather than silently
assuming it matches; follows Decision 0011's pattern of checking code, not
prose, for anything not yet directly inspected).
