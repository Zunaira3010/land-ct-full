# smoke_test_vae_masks.ps1
#
# Same purpose as smoke_test_vae.ps1: verify the mask VAE training pipeline
# actually works -- data loads, model builds, a forward/backward pass runs,
# checkpoints save, loss isn't NaN -- on a tiny subset and for 2 epochs,
# BEFORE committing GPU-hours to the real 150-epoch/1,010-patient run.
#
# This is also the FIRST real VRAM test of 256^3 FULL-VOLUME batch-1 on the
# RTX 2080 Ti for this project -- unlike the image VAE (128^3 patches,
# already confirmed to fit in ~7.2GB), the mask VAE trains on the whole
# volume with no patching (Decision 0008 -- this asymmetry between the two
# VAEs is real, not a mistake). Whether 11GB is enough is explicitly listed
# as "still unconfirmed" in docs/03_verified_paper_repo_spec.md -- this
# smoke test is what answers that, empirically, before the full run.
#
# Uses directory JUNCTIONS (not copies), same safe pattern as
# smoke_test_vae.ps1 -- do NOT clean up with `Remove-Item -Recurse` on the
# junction folder, use `cmd /c rmdir` instead (see smoke_test_vae.ps1's
# header for why).

param(
    [int]$NumPatients = 8,
    [string]$MaskMode = "nodule+lung",
    [int]$NumClasses = 3
)

$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -ne "land-ct-v2") {
    conda activate land-ct-v2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$ProcessedDir = Join-Path $repoRoot "data\processed"
$SmokeDataDir = Join-Path $repoRoot "data\processed_smoke_masks"
$ModelDir = Join-Path $repoRoot "checkpoints\vaeMasks_smoke"
$LogPath = Join-Path $repoRoot "logs\vaeMasks_smoke\logging"
$TensorboardLogDir = Join-Path $repoRoot "logs\vaeMasks_smoke\tensorboard"

# --- clean up any previous smoke run first ---
if (Test-Path $SmokeDataDir) {
    Write-Host "Removing previous smoke dataset junction folder..."
    Get-ChildItem $SmokeDataDir | ForEach-Object { cmd /c rmdir "`"$($_.FullName)`"" }
    cmd /c rmdir "`"$SmokeDataDir`""
}
if (Test-Path $ModelDir) { Remove-Item -Recurse -Force $ModelDir }
if (Test-Path (Join-Path $repoRoot "logs\vaeMasks_smoke")) { Remove-Item -Recurse -Force (Join-Path $repoRoot "logs\vaeMasks_smoke") }

New-Item -ItemType Directory -Force -Path $SmokeDataDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogPath | Out-Null
New-Item -ItemType Directory -Force -Path $TensorboardLogDir | Out-Null

# --- pick N patients (sorted, deterministic -- matches LIDCMasks' own
#     sorted glob) and junction them in ---
$patients = Get-ChildItem $ProcessedDir -Directory | Sort-Object Name | Select-Object -First $NumPatients
if ($patients.Count -lt 2) {
    Write-Error "Need at least 2 processed patients in $ProcessedDir to run a smoke test (need both a train and val split)."
    exit 1
}
Write-Host "Smoke-testing with $($patients.Count) patients:"
foreach ($p in $patients) {
    Write-Host "  $($p.Name)"
    New-Item -ItemType Junction -Path (Join-Path $SmokeDataDir $p.Name) -Target $p.FullName | Out-Null
}

$ModelConfig = Join-Path $repoRoot "src\vae\configs\config_vae_masks_nodule+lung_latent1.json"
$TrainConfig = Join-Path $repoRoot "src\vae\configs\config_vae_masks_train_smoke.json"

Write-Host ""
Write-Host "Mask mode    : $MaskMode ($NumClasses classes)"
Write-Host "Model config : $ModelConfig (unchanged -- real architecture)"
Write-Host "Train config : $TrainConfig (n_epochs=2, everything else identical to the real run)"
Write-Host "Dataset      : $SmokeDataDir ($($patients.Count) patients, junctions -- no data duplicated)"
Write-Host ""

$env:WANDB_MODE = "disabled"
$ErrorActionPreference = "Continue"

python -B "$repoRoot\src\vae\vae_masks_train.py" `
    --tensorboard_log_path "$TensorboardLogDir" `
    --model_dir "$ModelDir" `
    --model_config_file "$ModelConfig" `
    --train_config_file "$TrainConfig" `
    --dataset_path "$SmokeDataDir" `
    --train_portion 0.9 `
    --mask_mode "$MaskMode" `
    --num_classes $NumClasses `
    --log_path "$LogPath" `
    --run_name "vaeMasksLAND_smoketest"

Write-Host ""
Write-Host "=== Smoke test finished. Before trusting it, check: ==="
Write-Host "1. No CUDA OOM / crash above -- this is the real question for this stage (256^3 full-volume, not 128^3 patches)."
Write-Host "2. Loss values (vae_loss/train, vae_loss/val in the log) are real numbers, not NaN/Inf."
Write-Host "3. A checkpoint file exists: Get-ChildItem '$ModelDir' -Recurse"
Write-Host "4. Tensorboard logged something: tensorboard --logdir '$TensorboardLogDir'"
Write-Host "   -- open it and look at the colorized mask reconstruction videos, not just scalars."
Write-Host "5. Peak VRAM: if you watched nvidia-smi during the run, note the peak -- this is the number"
Write-Host "   that decides whether the full 150-epoch/1,010-patient run fits on this 11GB card as-is,"
Write-Host "   or needs a hardware-forced patch-training deviation (would need its own decision entry)."
Write-Host ""
Write-Host "If all good, clean up the smoke run (safe -- these are junctions/smoke-only checkpoints):"
Write-Host "  cmd /c rmdir `"$SmokeDataDir`"   <- do NOT use Remove-Item -Recurse here, see smoke_test_vae.ps1 header"
Write-Host "  Remove-Item -Recurse -Force '$ModelDir'"
Write-Host "  Remove-Item -Recurse -Force '$(Join-Path $repoRoot "logs\vaeMasks_smoke")'"
Write-Host ""
Write-Host "Then run the real thing: .\scripts\train_vae_masks.ps1"
