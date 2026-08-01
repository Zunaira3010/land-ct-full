# smoke_test_vae.ps1
#
# Same purpose as the preprocessing smoke test before the full 1,010-patient
# run: verify the VAE training pipeline actually works -- data loads, model
# builds, a forward/backward pass runs, checkpoints save, loss isn't NaN --
# on a tiny subset and for 2 epochs, BEFORE committing GPU-hours to the real
# 100-epoch/1,010-patient run. This is also the first real test of whether
# batch-1/128^3 patches actually fit in the RTX 2080 Ti's 11GB.
#
# Uses directory JUNCTIONS (not copies) to point a small subset dataset at a
# handful of already-processed patients, so this doesn't duplicate ~1.3GB+
# of data on disk. Junctions don't need admin rights on Windows (unlike
# symlinks).
#
# ONE IMPORTANT WARNING: when cleaning up afterwards, do NOT run
# `Remove-Item -Recurse -Force` directly on the `processed_smoke` junction
# folder. In some PowerShell versions this has a known bug where it follows
# the junction and deletes the REAL target contents (your actual processed
# patient data), not just the link. This script's own cleanup step below
# uses `cmd /c rmdir` instead, which only removes the junction itself and is
# safe. If you ever clean this up manually later, use the same approach:
#   cmd /c rmdir "data\processed_smoke"
# NOT `Remove-Item -Recurse data\processed_smoke`.

param(
    [int]$NumPatients = 8
)

$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -ne "land-ct-v2") {
    conda activate land-ct-v2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$ProcessedDir = Join-Path $repoRoot "data\processed"
$SmokeDataDir = Join-Path $repoRoot "data\processed_smoke"
$ModelDir = Join-Path $repoRoot "checkpoints\vae_smoke"
$LogPath = Join-Path $repoRoot "logs\vae_smoke\logging"
$TensorboardLogDir = Join-Path $repoRoot "logs\vae_smoke\tensorboard"

# --- clean up any previous smoke run first ---
if (Test-Path $SmokeDataDir) {
    Write-Host "Removing previous smoke dataset junction folder..."
    Get-ChildItem $SmokeDataDir | ForEach-Object { cmd /c rmdir "`"$($_.FullName)`"" }
    cmd /c rmdir "`"$SmokeDataDir`""
}
if (Test-Path $ModelDir) { Remove-Item -Recurse -Force $ModelDir }
if (Test-Path (Join-Path $repoRoot "logs\vae_smoke")) { Remove-Item -Recurse -Force (Join-Path $repoRoot "logs\vae_smoke") }

New-Item -ItemType Directory -Force -Path $SmokeDataDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogPath | Out-Null
New-Item -ItemType Directory -Force -Path $TensorboardLogDir | Out-Null

# --- pick N patients (sorted, deterministic -- matches vae_train.py's own
#     sorted glob, so this is a faithful small-scale stand-in for the real
#     dataset, not an arbitrary sample) and junction them in ---
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

$ModelConfig = Join-Path $repoRoot "src\vae\configs\config_vae.json"
$TrainConfig = Join-Path $repoRoot "src\vae\configs\config_vae_train_smoke.json"

Write-Host ""
Write-Host "Model config : $ModelConfig (unchanged -- real architecture)"
Write-Host "Train config : $TrainConfig (n_epochs=2, everything else identical to the real run)"
Write-Host "Dataset      : $SmokeDataDir ($($patients.Count) patients, junctions -- no data duplicated)"
Write-Host ""

$env:WANDB_MODE = "disabled"
$ErrorActionPreference = "Continue"

python -B "$repoRoot\src\vae\vae_train.py" `
    --tensorboard_log_path "$TensorboardLogDir" `
    --model_dir "$ModelDir" `
    --model_config_file "$ModelConfig" `
    --train_config_file "$TrainConfig" `
    --dataset_path "$SmokeDataDir" `
    --train_portion 0.9 `
    --log_path "$LogPath" `
    --run_name "vaeLAND_smoketest"

Write-Host ""
Write-Host "=== Smoke test finished. Before trusting it, check: ==="
Write-Host "1. No CUDA OOM / crash above (obviously, if you're reading this it didn't crash)."
Write-Host "2. Loss values printed during training are real numbers, not NaN/Inf."
Write-Host "3. A checkpoint file exists: Get-ChildItem '$ModelDir' -Recurse"
Write-Host "4. Tensorboard logged something: tensorboard --logdir '$TensorboardLogDir'"
Write-Host "   -- open it and look at the reconstruction images/videos it saves, not just scalars."
Write-Host "5. Peak VRAM: if you watched nvidia-smi during the run, note the peak -- this tells you"
Write-Host "   how much headroom you actually have for the real 100-epoch run."
Write-Host ""
Write-Host "If all good, clean up the smoke run (safe -- these are junctions/smoke-only checkpoints):"
Write-Host "  cmd /c rmdir `"$SmokeDataDir`"   <- do NOT use Remove-Item -Recurse here, see script header"
Write-Host "  Remove-Item -Recurse -Force '$ModelDir'"
Write-Host "  Remove-Item -Recurse -Force '$(Join-Path $repoRoot "logs\vae_smoke")'"
Write-Host ""
Write-Host "Then run the real thing: .\scripts\train_vae.ps1"
