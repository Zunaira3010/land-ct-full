# 03 — Verified Spec: Paper + Official Repo Cross-Check

*Single source of truth for exact fidelity (Decision 0002). Every line here is traced to either
the paper text or the official repo's README/source — nothing here is inferred or assumed.
Anything still unconfirmed is listed explicitly at the bottom, not silently filled in.*

## Architecture — confirmed, paper and repo agree

| Component | Spec |
|---|---|
| Image VAE | 3 resolution levels, 1 residual block/level, balanced encoder/decoder channel widths (not doubled in decoder like base MAISI). Loss: MAE + LPIPS + adversarial (discriminator C) + KL. |
| Mask VAE | Same architecture family as image VAE, different input channels (one-hot). Loss: WCE + GDL + KL (β=1e-7). Class weights **as stated in the paper text**: α_nodule=10, α_lung=1, α_other=0.1 — but see the correction below, the actual training code uses a different background weight. |
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
| Mask VAE training | paper says "same procedure" (100 epochs); **code actually uses 150** | Paper text vs. `config_vae_masks_train.json` — see corrections below |
| U-Net training | 500,000 **optimization steps** (not epochs) | Paper |
| U-Net LR | 1e-5 (AdamW) | Paper |
| Batch size (all 3 networks) | 1 | Paper + repo (`bsz` example = 1) |
| Diffusion steps T | 1000 | Paper + repo (`num_inference_steps` example = 1000) |
| Prediction type | v_prediction | Paper + repo (`prediction_type` example) |
| Resolution | 256³, 1mm isotropic | Paper + repo |
| `--attention` flag | used in **all** experiments in the paper | Repo README |

## Resolved — epochs vs. steps for the U-Net

`train_ldm.sh`'s actual `num_epochs=500` (not just an example value, confirmed by reading the
script directly). With all 1,010 LIDC-IDRI patients and `bsz=1`, that's 500 × 1,010 ≈ 505,000
optimizer steps — matching the paper's reported 500,000 steps closely enough (~1% over) to
confirm these reconcile rather than conflict. Treat `num_epochs=500` as the operational target;
it isn't a separate, smaller run than the paper describes.

## Data — confirmed

- Training: LIDC-IDRI only, 1,010 volumes, both VAEs + U-Net.
- NLST: inference/eval only, never training. 881 volumes with nodule annotations used for FID/MMD.
- Preprocessing: HU-clip [−1000, 0.01 upper percentile] → resample 1mm isotropic → center-crop
  256³ → normalize [0,1] (paper's prose reading — see the corrected normalization entry below for
  what the code actually does).
- **Masks are extracted independently, not downloaded precomputed** (Decision 0007, supersedes
  the original Decision 0005 plan) — the authors' precomputed mask package is gated behind
  Eurecat's SharePoint tenant. `preprocessing.py` generates both CT `.npy` volumes and mask
  `.npy` files in the same pass, via `pylidc.consensus()` (nodule annotations) and `lungmask`'s
  `LMInferer` (lung segmentation) — the same tools the paper itself cites, not a downstream
  artifact of them.

## Pipeline stage order — confirmed (repo README, matches BRAIN's existing phase plan)

1. `preproc_data.sh` → DICOMs to `.npy` CT + masks (both generated locally, see above)
2. `train_vae.sh` → image VAE
3. `train_vae_masks.sh` → mask VAE (needs `MASK_MODE=nodule+lung` for our target config)
4. `train_ldm.sh` → U-Net, loads both frozen VAE checkpoints (`vae_dir`, `vae_mask_dir`)
5. `inference_ldm.sh` → sampling

## License — confirmed (Decision 0005)

Code: Apache 2.0. Paper: CC BY-NC-ND 4.0. No conflict with building, referencing, or eventually
citing this work in a publication.

## Resolved — read directly from `reference_official_repo/src/vae/configs/*.json`

- **`LATENT_SIZE`** = `latent_channels` in the mask-VAE config. `config_vae_masks_nodule+lung_latent1.json`
  (our target `nodule+lung` mode) confirms `latent_channels: 1` — so `LATENT_SIZE=1` means a
  1-channel latent bottleneck, not an ambiguous "size" knob.
- **Image VAE channel widths** (`config_vae.json`): `num_channels: [32, 64, 128]`, 1 residual
  block per level, `norm_num_groups: 32`, `latent_channels: 4` — matches the paper's "4-channel
  latent" and "3 resolution levels, 1 block each."
- **Mask VAE channel widths** (`config_vae_masks.json` / `config_vae_masks_nodule+lung_latent1.json`):
  `num_channels: [16, 32, 64]`, `norm_num_groups: 16` — narrower than the image VAE, consistent
  with masks carrying less information than raw CT intensity.

## Corrections — paper text vs. actual training code (verified by reading both directly)

These are genuine paper/code discrepancies, not our own deviations — logged as Decisions
0008–0011 in the memory pack. Our implementation follows the **code** (the actual training
configuration used to produce the paper's results), not the paper's prose, per Decision 0002.

- **Mask VAE epochs**: paper says "same procedure as the image VAE" (100 epochs, implying the
  same). `config_vae_masks_train.json` sets `n_epochs: 150`. Use 150.
- **Mask VAE background class weight**: paper text states α_other=0.1. `vae/utils/masks_utils.py`'s
  `vae_loss_segmentation` hardcodes `class_weights = [0.5, 10.0, 1.0]` for the 3-class
  (`nodule+lung`) case — background weight is **0.5**, not 0.1. Use 0.5.
- **Image VAE patch training**: not stated in the paper's main text at all (only visible in
  `config_vae_train.json`'s `patch_size: [128, 128, 128]`). This is the authors' own training
  setup on their A100, not a compromise specific to our smaller GPU — no deviation to log here,
  just something the paper omits and the code reveals.
- **HU normalization**: paper's prose implies a fixed clip range; the actual function used
  (`normalize_ct_vol_wdm`) clips at the running volume's 99.9th percentile and does per-volume
  min-max, not a fixed range. The fixed-range function (`normalize_ct_vol`) exists in the repo
  but is dead code — never called by the pipeline. Use `normalize_ct_vol_wdm`.
- **Nodule texture values**: `preprocess_dicom()` stores each nodule's texture as the mean of all
  contributing radiologists' scores (can be non-integer, e.g. 2.33), not a clean 1–5. Separately,
  `LIDCMasks.__getitem__` randomly reassigns each nodule's texture label 1–5 by default
  (`original_textures=False`) when `mask_mode="nodule+lung+texture"` — irrelevant to our current
  `nodule+lung` (T=3) target, since that mode has no texture channel, but relevant if/when the
  T=7 variant is attempted later; pass `--original_textures` to preserve real annotated scores.

## Still unconfirmed — do not assume

- Whether patch-based training is needed for the mask VAE (256³ full-volume, per
  `config_vae_masks_train.json`) on the RTX 2080 Ti's 11GB, or whether it fits as-is like the
  paper's own A100 run — test empirically via the Phase 5 smoke test before committing to the
  full 150-epoch run.
