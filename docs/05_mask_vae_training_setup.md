# 05 — Mask VAE Training Setup (Phase 5)

## Target config

`nodule+lung` mode, T=3 (paper's **LAND-LatentMask** — the headline conditional config, Decision
0002/0005). Not `nodule+lung+texture` (T=7, LAND-LatentMask+) — that's a later, separate target if
pursued at all; don't default to it.

- `--mask_mode nodule+lung --num_classes 3`
- Model config: `src/vae/configs/config_vae_masks_nodule+lung_latent1.json`
  (`in_channels`/`out_channels` get overwritten to `num_classes` at runtime by
  `vae_masks_train.py` itself regardless of what's in the file — confirmed by reading
  `load_configurations()` — so this is really about `latent_channels: 1` (`LATENT_SIZE=1`) and
  `num_channels: [16, 32, 64]`, the actual architecture-defining fields.)
- Train config: `src/vae/configs/config_vae_masks_train.json` — **150 epochs**, not the 100 the
  paper's prose implies (Decision 0009), **background class weight 0.5**, not the 0.1 the paper
  states (Decision 0010, hardcoded in `masks_utils.py`, not a config field — nothing to set here).

## What's already in place, unmodified (Decision 0014 reuse pattern)

Everything under `src/vae/` needed for this stage already exists, copied verbatim from the
official repo in Phase 4 alongside the image VAE files — `vae_masks_train.py`,
`utils/masks_utils.py` (`LIDCMasks` dataset class + `vae_loss_segmentation`), and all three
`config_vae_masks*.json` variants. No new source files needed for this phase, only new scripts.

## New this phase

- `scripts/train_vae_masks.ps1` — Windows-local wrapper for the official `train_vae_masks.sh`,
  same pattern as `train_vae.ps1`. Hyperparameters unchanged from the shipped configs.
- `scripts/smoke_test_vae_masks.ps1` — same junction-based smoke-test pattern as
  `smoke_test_vae.ps1`, 8 patients, 2 epochs.
- `src/vae/configs/config_vae_masks_train_smoke.json` — identical to
  `config_vae_masks_train.json` except `n_epochs: 2`, matching the existing
  `config_vae_train_smoke.json` pattern for the image VAE.

## Real difference from the image VAE stage — read before running

The image VAE trains on **128³ patches** (Decision 0008 — the authors' own choice, already
confirmed to fit comfortably in ~7.2GB peak on this card). The mask VAE trains on **full 256³
volumes**, batch 1, no patching at all — this is what `config_vae_masks_train.json`'s
`patch_size: [256,256,256]` means in practice: `vae_masks_train.py`'s data pipeline
(`LIDCMasks.__getitem__`) loads the whole mask volume every time, it does not apply any
patch-cropping transform.

Whether 11GB is enough for this is **explicitly unconfirmed** —
`docs/03_verified_paper_repo_spec.md`'s "still unconfirmed" section flags exactly this. The paper
authors ran it on a 20GB A100; we don't yet know if it fits on the 2080 Ti. **This is what the
smoke test is for.** If it OOMs, that would be a genuine hardware-forced deviation (patch-based
mask VAE training) — log it as its own decision if/when it happens, don't just silently patch it.

## Checkpoint/resume — already native, nothing to port from Decision 0015

Unlike `vae_train.py` (which needed a ChatGPT-authored patch to gain checkpoint/resume support —
Decision 0015), `vae_masks_train.py` **already has this built in** in the official code:
`--resume_checkpoint <path to last_checkpoint.pth>` restores epoch, optimizer, scheduler, scaler,
and `best_val_loss`, then resumes at `epoch + 1`. `06_NEXT_SESSION.md`'s open question ("decide
whether to port Decision 0015's pattern into `vae_masks_train.py`") is resolved: **no porting
needed**, the shipped script already does it. `train_vae_masks.ps1` exposes this via
`-ResumeCheckpoint`.

One thing this native version does *not* have that Decision 0015's patch added for the image VAE:
per-epoch checkpoint atomicity and run-name scoping are the same two flagged-but-unfixed gaps
(non-atomic `torch.save`, `checkpoint_last.pt`-style path not scoped beyond the run's own
timestamped folder — actually `vae_masks_train.py` *does* scope its checkpoint inside
`sub_folder_dir`, i.e. the timestamped run folder, so the run-name-collision gap doesn't apply
here the way it did for `vae_train.py`'s flat `model_dir/checkpoint_last.pt` path). Non-atomicity
is still a real, low-probability risk (a kill mid-`torch.save` could corrupt the one checkpoint
file) — not fixed here, consistent with treating it as low-priority per Decision 0015.

## Early stopping — on by default, this can end the run before epoch 150

`vae_masks_train.py`'s CLI defaults: `--early_stopping_patience 10 --early_stopping_min_delta
0.0`. If validation loss doesn't improve for 10 consecutive validation epochs, training stops and
saves, even if `n_epochs` (150) hasn't been reached. This is official-repo default behavior, not
something this project added — if the run stops early, check `06_NEXT_SESSION.md`/the training log
for the actual stopping epoch before assuming something went wrong.

## Running it

**Smoke test first, always:**

```powershell
.\scripts\smoke_test_vae_masks.ps1
```

Checks: no CUDA OOM at 256³/batch-1 (the open question above), loss isn't NaN, a checkpoint
actually saves, TensorBoard logs colorized mask reconstruction videos. This is the first real
VRAM read for this specific configuration — don't skip it just because the image VAE smoke test
already passed; that was a different patch size on a different architecture.

**Then the real run:**

```powershell
.\scripts\train_vae_masks.ps1
```

Defaults: `data\processed` as the dataset, 90/10 train/val split, `nodule+lung`/3 classes,
checkpoints to `checkpoints\vaeMasks\`, logs to `logs\vaeMasks\`. Same foreground-first,
background-once-stable discipline as `train_vae.ps1` — watch the first epoch for OOM before
walking away, then relaunch as a background process (see the script's own header comment for the
exact `Start-Process` invocation, same pattern as `train_vae.ps1`/`preproc_data.ps1`).

## Open question, not yet resolved

Whether 256³/batch-1 fits in 11GB at all — see above. Test empirically, don't assume either way.
