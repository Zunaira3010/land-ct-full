# train_vae_masks.ps1
# Adapted from the official LAND repository's scripts/train_vae_masks.sh
# (Apache 2.0 -- see reference_official_repo/LICENSE, /NOTICE).
#
# CHANGE FROM ORIGINAL: same class of change as train_vae.ps1 -- the original
# is a SLURM cluster job script (module load, sbatch directives, a shared
# cluster conda env path). This is a plain local-run version for a single
# Windows machine -- no SLURM, `conda activate` instead of `source activate`.
# The actual training call, model config, train config, mask_mode, and all
# hyperparameters are UNCHANGED -- Decision 0002 (fidelity-to-paper policy):
# this trains against the same config_vae_masks_nodule+lung_latent1.json /
# config_vae_masks_train.json the paper's authors shipped, not a modified
# version.
#
# Target config: nodule+lung (T=3, "LAND-LatentMask" -- Decision 0002/0005),
# 150 epochs (Decision 0009, not the paper prose's 100), background class
# weight 0.5 baked into masks_utils.py (Decision 0010, not the paper's 0.1).
# Neither of those is a flag here -- both are hardcoded in the shipped code
# this script calls unmodified, so there's nothing to override.
#
# NOTE: unlike vae_train.py (which needed Decision 0015's ChatGPT-authored
# checkpoint/resume patch), vae_masks_train.py already has native
# checkpoint/resume support in the official code (--resume_checkpoint).
# Nothing to port from Decision 0015 for this stage.
#
# NOTE: early stopping is on by default (--early_stopping_patience 10,
# --early_stopping_min_delta 0.0, both official-repo defaults, unchanged
# here). A 150-epoch run may legitimately stop earlier than epoch 150 if
# val loss plateaus for 10 straight validation epochs -- that is expected
# behavior from the shipped code, not a bug, if/when it happens.
#
# Usage:
#   .\scripts\train_vae_masks.ps1                  # full 150-epoch run, resumable via checkpoints\vaeMasks
#   .\scripts\train_vae_masks.ps1 -ResumeCheckpoint "checkpoints\vaeMasks\vaeMasksLAND_<timestamp>\last_checkpoint.pth"
#
# Runs in the foreground by default -- watch the first few iterations for
# OOM before walking away (256^3 full-volume batch-1 has NOT been VRAM-
# tested on this card yet, unlike the image VAE's 128^3 patches -- see
# docs/05_mask_vae_training_setup.md). Once epoch 1 completes cleanly, background
# it the same way as train_vae.ps1 was:
#   Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$PWD'; conda activate land-ct-v2; .\scripts\train_vae_masks.ps1 *>&1 | Tee-Object -FilePath train_vae_masks_full_run.log`""

param(
    [string]$DatasetPath = $null,
    [string]$RunName = "vaeMasksLAND",
    [double]$TrainPortion = 0.9,
    [string]$MaskMode = "nodule+lung",
    [int]$NumClasses = 3,
    [string]$ResumeCheckpoint = $null
)

$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -ne "land-ct-v2") {
    conda activate land-ct-v2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $DatasetPath) {
    $DatasetPath = Join-Path $repoRoot "data\processed"
}

# nodule+lung -> config_vae_masks_nodule+lung_latent1.json (T=3, LATENT_SIZE=1).
# If ever targeting the texture variant later, this is the file to swap,
# alongside -MaskMode "nodule+lung+texture" -NumClasses 7 -- log it as its
# own decision when that day comes, don't silently change the default here.
$ModelConfig = Join-Path $repoRoot "src\vae\configs\config_vae_masks_nodule+lung_latent1.json"
$TrainConfig = Join-Path $repoRoot "src\vae\configs\config_vae_masks_train.json"
$ModelDir = Join-Path $repoRoot "checkpoints\vaeMasks"
$LogPath = Join-Path $repoRoot "logs\vaeMasks\logging"
$TensorboardLogDir = Join-Path $repoRoot "logs\vaeMasks\tensorboard"

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogPath | Out-Null
New-Item -ItemType Directory -Force -Path $TensorboardLogDir | Out-Null

Write-Host "Dataset path : $DatasetPath"
Write-Host "Mask mode    : $MaskMode ($NumClasses classes)"
Write-Host "Model config : $ModelConfig"
Write-Host "Train config : $TrainConfig"
Write-Host "Checkpoints  : $ModelDir"
if ($ResumeCheckpoint) { Write-Host "Resuming from: $ResumeCheckpoint" }

# Same as the original: disable wandb rather than require a login.
$env:WANDB_MODE = "disabled"

# IMPORTANT: switch to Continue before invoking python -- same reasoning as
# train_vae.ps1/preproc_data.ps1 (Bug 003): with Stop still set, redirecting
# stderr via *>&1 for logging turns any harmless stderr line into a
# terminating error.
$ErrorActionPreference = "Continue"

$pyArgs = @(
    "-B", "$repoRoot\src\vae\vae_masks_train.py",
    "--tensorboard_log_path", "$TensorboardLogDir",
    "--model_dir", "$ModelDir",
    "--model_config_file", "$ModelConfig",
    "--train_config_file", "$TrainConfig",
    "--dataset_path", "$DatasetPath",
    "--train_portion", "$TrainPortion",
    "--mask_mode", "$MaskMode",
    "--num_classes", "$NumClasses",
    "--log_path", "$LogPath",
    "--run_name", "$RunName"
)
if ($ResumeCheckpoint) {
    $pyArgs += @("--resume_checkpoint", "$ResumeCheckpoint")
}

python @pyArgs
