# train_vae.ps1
# Adapted from the official LAND repository's scripts/train_vae.sh
# (Apache 2.0 -- see reference_official_repo/LICENSE, /NOTICE).
#
# CHANGE FROM ORIGINAL: the original is a SLURM cluster job script
# (module load, sbatch directives, a shared cluster conda env path,
# a hardcoded external dataset path). This is a plain local-run version
# for a single Windows machine -- no SLURM, no module system, `conda
# activate` instead of `source activate`. The actual training call,
# model config, train config, and all hyperparameters are UNCHANGED --
# Decision 0002 (fidelity-to-paper policy): this project trains against
# the same config_vae.json / config_vae_train.json the paper's authors
# shipped, not a modified version.
#
# Usage:
#   .\scripts\train_vae.ps1                  # full 100-epoch run, resumable via checkpoints\vae
#
# Runs in the foreground by default -- this is the first real VRAM test
# of the pipeline on the RTX 2080 Ti (11GB); watch the first few iterations
# for OOM before walking away. Once epoch 1 completes cleanly, it's safe to
# re-launch as a background process the same way preproc_data.ps1 was run:
#   Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$PWD'; conda activate land-ct-v2; .\scripts\train_vae.ps1 *>&1 | Tee-Object -FilePath train_vae_full_run.log`""

param(
    [string]$DatasetPath = $null,
    [string]$RunName = "vaeLAND",
    [double]$TrainPortion = 0.9
)

$ErrorActionPreference = "Stop"

# Activate env if not already active
if ($env:CONDA_DEFAULT_ENV -ne "land-ct-v2") {
    conda activate land-ct-v2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $DatasetPath) {
    $DatasetPath = Join-Path $repoRoot "data\processed"
}

$ModelConfig = Join-Path $repoRoot "src\vae\configs\config_vae.json"
$TrainConfig = Join-Path $repoRoot "src\vae\configs\config_vae_train.json"
$ModelDir = Join-Path $repoRoot "checkpoints\vae"
$LogPath = Join-Path $repoRoot "logs\vae\logging"
$TensorboardLogDir = Join-Path $repoRoot "logs\vae\tensorboard"

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogPath | Out-Null
New-Item -ItemType Directory -Force -Path $TensorboardLogDir | Out-Null

Write-Host "Dataset path : $DatasetPath"
Write-Host "Model config : $ModelConfig"
Write-Host "Train config : $TrainConfig"
Write-Host "Checkpoints  : $ModelDir"

# Same as the original: disable wandb rather than require a login. Set
# to "online" yourself first if you want real wandb logging instead.
$env:WANDB_MODE = "disabled"

# IMPORTANT: switch to Continue before invoking python -- same reasoning
# as preproc_data.ps1 (Bug 003): with Stop still set, redirecting stderr
# via *>&1 for logging turns any harmless stderr line (deprecation
# warnings, MONAI's print_config banner, etc.) into a terminating error.
$ErrorActionPreference = "Continue"

python -B "$repoRoot\src\vae\vae_train.py" `
    --tensorboard_log_path "$TensorboardLogDir" `
    --model_dir "$ModelDir" `
    --model_config_file "$ModelConfig" `
    --train_config_file "$TrainConfig" `
    --dataset_path "$DatasetPath" `
    --train_portion $TrainPortion `
    --log_path "$LogPath" `
    --run_name "$RunName"
