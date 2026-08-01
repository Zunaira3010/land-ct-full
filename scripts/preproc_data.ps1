# preproc_data.ps1
# Adapted from the official LAND repository's scripts/preproc_data.sh
# (Apache 2.0 — see reference_official_repo/LICENSE, /NOTICE).
#
# CHANGE FROM ORIGINAL: the original is a SLURM cluster job script
# (module load, sbatch directives, a shared cluster conda path). This is
# a plain local-run version for a single Windows machine — no SLURM,
# no module system, `conda activate` instead of `source activate`.
# The actual preprocessing call and flags (--normalize --resample
# --central_crop) are unchanged.
#
# Usage:
#   .\scripts\preproc_data.ps1                  # full 1,010-patient run
#   .\scripts\preproc_data.ps1 -SampleIdx 1      # single patient smoke test

param(
    [int]$SampleIdx = $null,
    [string]$DicomDir = "C:\Users\24COB\Downloads\lidc_idri",
    [string]$NpyDir = "C:\Users\24COB\Documents\land-ct-full\data\processed"
)

$ErrorActionPreference = "Stop"

# Ensure pylidc's config points to the DICOM root.
# On Windows, pylidc reads ~/pylidc.conf (not ~/.pylidcrc, which is the
# Unix-style name it uses elsewhere) -- write BOTH so this works regardless
# of which one this pylidc/OS combination actually reads. Known quirk,
# see docs/01_env_setup.md / ENVIRONMENT.md ("keep .pylidcrc and pylidc.conf
# in sync"). Confirmed necessary on Windows: without pylidc.conf, pylidc
# raises "Could not establish path to dicom files" even with .pylidcrc set.
$dicomConfigContent = "[dicom]`npath = $DicomDir"
$pylidcrc = "$env:USERPROFILE\.pylidcrc"
$pylidcconf = "$env:USERPROFILE\pylidc.conf"
$dicomConfigContent | Out-File -Encoding ascii -FilePath $pylidcrc
$dicomConfigContent | Out-File -Encoding ascii -FilePath $pylidcconf
Write-Host "Set .pylidcrc and pylidc.conf to use DICOM path: $DicomDir"

# Activate env if not already active
if ($env:CONDA_DEFAULT_ENV -ne "land-ct-v2") {
    conda activate land-ct-v2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# IMPORTANT: switch to Continue before invoking python. With ErrorActionPreference
# still set to Stop, redirecting a native command's stderr (e.g. via *>&1, used when
# piping this script's output to a log file) makes PowerShell treat ANY stderr line
# -- including harmless warnings like pylidc's pkg_resources deprecation notice --
# as a terminating error, aborting the entire run after the first such line. Bug 003.
$ErrorActionPreference = "Continue"

if ($PSBoundParameters.ContainsKey('SampleIdx')) {
    python -m src.data.preprocessing --dicom_dir "$DicomDir" --npy_dir "$NpyDir" --normalize --resample --central_crop --sample_idx $SampleIdx
} else {
    python -m src.data.preprocessing --dicom_dir "$DicomDir" --npy_dir "$NpyDir" --normalize --resample --central_crop
}
