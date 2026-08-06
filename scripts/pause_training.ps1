# pause_training.ps1
# Requests a clean pause of a currently-running .\scripts\train_vae_masks.ps1 job.
#
# HOW IT WORKS: drops a PAUSE_REQUESTED file into the run's checkpoint folder.
# vae_masks_train.py checks for this file once per epoch, right after that
# epoch's checkpoint has already been safely written to disk (atomic write --
# it can't be corrupted, and nothing already-completed can be lost this way).
# When found, training exits cleanly with a normal exit code, not a crash.
#
# This is the SAFE way to stop training on demand -- e.g. a friend needs the
# PC. Prefer this over closing the terminal or Ctrl+C, since those interrupt
# mid-epoch and lose whatever that epoch had done so far (still safe, thanks
# to auto-resume + atomic checkpoints, just wastes a partial epoch's compute).
# This script waits for the current epoch to finish instead, so nothing is
# wasted -- typically a few minutes at most.
#
# RESUME: no separate "resume" script needed -- just run train_vae_masks.ps1
# again, the exact same command as before. With -AutoResume on (the default),
# it finds the checkpoint and continues automatically from the next epoch.
#
# Usage:
#   .\scripts\pause_training.ps1                  # pauses the default run (vaeMasksLAND)
#   .\scripts\pause_training.ps1 -RunName "myRun"  # if you launched with a custom -RunName

param(
    [string]$RunName = "vaeMasksLAND"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$ModelDir = Join-Path $repoRoot "checkpoints\vaeMasks"
$RunDir = Join-Path $ModelDir $RunName
$FlagPath = Join-Path $RunDir "PAUSE_REQUESTED"

if (-not (Test-Path $RunDir)) {
    Write-Host "No run folder found at $RunDir -- is training actually running with -RunName '$RunName'?" -ForegroundColor Red
    Write-Host "If you launched with a different -RunName, pass it here too: .\scripts\pause_training.ps1 -RunName '<name>'" -ForegroundColor Yellow
    exit 1
}

New-Item -ItemType File -Force -Path $FlagPath | Out-Null

Write-Host "Pause requested for run '$RunName'."
Write-Host "Training will stop cleanly after it finishes the epoch currently in progress"
Write-Host "(usually a few minutes) -- watch its terminal for the confirmation message."
Write-Host ""
Write-Host "To resume later: just run .\scripts\train_vae_masks.ps1 again, same as always."
