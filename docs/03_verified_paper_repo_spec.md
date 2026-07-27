# 03 — Verified Spec: Paper + Official Repo Cross-Check

*Single source of truth for exact fidelity (Decision 0002). Every line here is traced to either
the paper text or the official repo's README — nothing here is inferred or assumed. Anything
still unconfirmed is listed explicitly at the bottom, not silently filled in.*

## Architecture — confirmed, paper and repo agree

| Component | Spec |
|---|---|
| Image VAE | 3 resolution levels, 1 residual block/level, balanced encoder/decoder channel widths (not doubled in decoder like base MAISI). Loss: MAE + LPIPS + adversarial (discriminator C) + KL. |
| Mask VAE | Same architecture family as image VAE, different input channels (one-hot). Loss: WCE + GDL + KL (β=1e-7). Class weights: α_nodule=10, α_lung=1, α_other=0.1. |
| U-Net | 5 resolution levels, 2 residual blocks/level, **additive skips** (not concat), cross-attention at every level. v-prediction target. Linear noise schedule β₁=1e-4→β_T=0.02, T=1000. Min-SNR-γ loss, γ=5.0 (confirmed against repo in earlier session). |
| Latent shape | Image: 4 × 64×64×64 for a 256³ input (4x downsampling per axis). |

## Mask conditioning modes — corrected/refined from repo (more precise than paper naming alone)

| `mask_mode` (repo param) | Classes | Paper name |
|---|---|---|
| `nodule` | 2 | (not a named ablation variant in the paper's tables, but a valid repo mode) |
| `nodule+lung` | 3 | **LAND-LatentMask** — our target config (Decision 0002, 0005) |
| `nodule+lung+texture` | 7 | LAND-LatentMask+ |

Mask value convention (paper): lung = 0.5, nodule = 1 (uniform) or 1–5 (texture-graded), else 0.

## Hyperparameters — confirmed

| Param | Value | Source |
|---|---|---|
| Image VAE epochs | 100 | Paper |
| Image VAE LR | 1e-4 (AdamW) | Paper |
| Mask VAE training | same procedure as image VAE | Paper |
| U-Net training | 500,000 **optimization steps** (not epochs) | Paper |
| U-Net LR | 1e-5 (AdamW) | Paper |
| Batch size (all 3 networks) | 1 | Paper + repo (`bsz` example = 1) |
| Diffusion steps T | 1000 | Paper + repo (`num_inference_steps` example = 1000) |
| Prediction type | v_prediction | Paper + repo (`prediction_type` example) |
| Resolution | 256³, 1mm isotropic | Paper + repo |
| `--attention` flag | used in **all** experiments in the paper | Repo README |

## ⚠️ One nuance to resolve before writing the training script — NOT yet reconciled

The repo's `train_ldm.sh` config table lists `num_epochs` (example value: 500) as the controlling
parameter, but the **paper reports 500,000 steps**, not 500 epochs. These aren't necessarily
contradictory (500 epochs over N samples could equal ~500k steps depending on dataset size and
batch size), but **don't assume they reconcile — compute it explicitly** once the real dataset
size is known: `steps = epochs × (num_samples / batch_size)`. If the numbers don't line up, trust
the paper's explicit step count (500,000) as the fidelity target, since Decision 0002 treats the
paper text as the primary source and the repo as an implementation reference.

## Data — confirmed

- Training: LIDC-IDRI only, 1,010 volumes, both VAEs + U-Net.
- NLST: inference/eval only, never training. 881 volumes with nodule annotations used for FID/MMD.
- Preprocessing: HU-clip [−1000, 0.01 upper percentile] → resample 1mm isotropic → center-crop
  256³ → normalize [0,1].
- **Precomputed masks available from the authors for both LIDC and NLST** (Decision 0005) — only
  real CT `.npy` volumes need generating locally via `scripts/preproc_data.sh`.

## Pipeline stage order — confirmed (repo README, matches BRAIN's existing phase plan)

1. `preproc_data.sh` → DICOMs to `.npy` CT + masks
2. `train_vae.sh` → image VAE
3. `train_vae_masks.sh` → mask VAE (needs `MASK_MODE=nodule+lung` for our target config)
4. `train_ldm.sh` → U-Net, loads both frozen VAE checkpoints (`vae_dir`, `vae_mask_dir`)
5. `inference_ldm.sh` → sampling

## License — confirmed (Decision 0005)

Code: Apache 2.0. Paper: CC BY-NC-ND 4.0. No conflict with building, referencing, or eventually
citing this work in a publication.

## Still unconfirmed — do not assume, verify when the actual config files are reachable

- `LATENT_SIZE` parameter for the mask VAE (repo table shows example value `1`, described only as
  "latent bottleneck size" — not explained further in the README or paper text). Check the actual
  `src/vae/configs/` JSON once cloned locally, don't guess what this controls.
- Exact channel widths/counts for the "balanced" VAE configuration (paper says "balanced" and
  "lightweight," doesn't give literal channel numbers) — read from `config_vae.json` directly.
- Whether `num_epochs` and the paper's 500k-step figure actually reconcile for the full
  1,010-patient dataset (see nuance above) — compute once dataset size is finalized.
