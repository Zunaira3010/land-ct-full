# Environment Setup — New Lab PC

## Confirmed hardware (July 27, 2026)

**RTX 2080 Ti, 11GB VRAM** — corrected from the "same as old PC, GTX 1080" assumption. See
Decision 0003. Driver: WDDM, KMD 610.62, CUDA UMD 13.3.

## 0. Conda isn't installed yet — completely fresh machine

`conda` not being recognized is expected on a brand-new Windows install. Install Miniconda first:

```powershell
winget install -e --id Anaconda.Miniconda3
```

After it finishes, **close and reopen PowerShell** (the PATH update needs a fresh shell), then
confirm:

```powershell
conda --version
```

If `conda` still isn't recognized after reopening, run this once to register it with PowerShell,
then reopen the shell again:

```powershell
conda init powershell
```

## 1. Create the environment

```powershell
conda create -n land-ct-v2 python=3.10
conda activate land-ct-v2
```

## 2. Repo + git (local config only — shared PC)

```powershell
cd land-ct-full   # wherever you unzipped the scaffold
git init
git config --local user.email "zunaira@..."
git config --local user.name "Zunaira3010"
git config --local --list
```

## Known quirks carried forward from v1 (verify each still applies on the new PC)

- `setuptools < 81` required if using `pylidc` (needs `pkg_resources`).
- PyTorch CUDA build: driver reports CUDA UMD 13.3, which is newer than assumed — a cu124 wheel
  should still work (PyTorch CUDA wheels are backward-compatible with newer drivers), but verify
  with `torch.cuda.is_available()` after install rather than assuming. If it returns `False`,
  check the current PyTorch install page for the latest recommended wheel instead of forcing
  cu124.
- `pylidc` config: keep both `~/.pylidcrc` and `~/pylidc.conf` in sync.
- Windows: line continuation in cmd.exe is `^`, not `\`.
- DataLoader transforms must be picklable — `functools.partial`, never `lambda`.
- `git config --local`, never `--global`, if this PC is shared with anyone — **check this on
  day one**, don't assume it's a solo account just because it's "your account."
- matplotlib + PyTorch bundled OpenMP conflict on Windows/conda (v1 Bug 006) —
  `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` before importing numpy/torch/matplotlib,
  in **every** script that plots, added from the start this time.

## Confirmed (July 25, 2026)

- **Disk:** ~500GB free at initial setup, precisely unchecked at the time. **Updated July 27,
  2026:** confirmed via `Get-PSDrive` — C: has 851GB free of 930GB. Full pipeline footprint is
  ~260GB (raw DICOM ~125GB + processed `.npy` for all 1,010 patients ~130–135GB), comfortably
  covered; no need to relocate to another drive.
- **Sharing:** **Same Windows user account** as one teammate, working on a **LungDDPM** paper
  implementation — corrected from an earlier assumption of "separate accounts, shared PC" (see
  Decision 0004). This is a bigger deal than just a shared machine:
  - Anything under `$env:USERPROFILE` (config files, conda base install, PATH/profile changes)
    is **shared with your teammate**, not just the disk. `git config --local` is still the right
    call for repo-level identity, but it doesn't isolate account-level settings.
  - **`pylidc` config collision risk:** both `~/.pylidcrc` and `~/pylidc.conf` live under the
    shared profile — if your teammate's project also touches LIDC-IDRI via `pylidc`, check for
    an existing config before overwriting it.
  - Conda **environments** are still fine to keep separate by name (`land-ct-v2` vs. whatever
    they're using) — envs don't collide just because the account is shared, only files that live
    directly under the profile root do.
  - PowerShell execution-policy changes (`Set-ExecutionPolicy -Scope CurrentUser`) apply to this
    shared account — reasonable default, but means it also affects your teammate's shell, not
    just yours.
- **Deadline shape:** Weekly briefings to Sir, but no hard external deadline forcing a rushed
  timeline — training is expected to run across the week between briefings. This means
  **checkpoint/reporting cadence matters** (need something concrete to show weekly), but doesn't
  force cutting corners on the full-paper scope. Exact checkpoint-cadence policy is a follow-up
  decision, not decided yet (flagged in BRAIN 00_INDEX open questions).

## Setup checklist — all confirmed

- [x] GPU: **RTX 2080 Ti, 11GB VRAM** confirmed via `nvidia-smi` (see Decision 0003) — not a
      GTX 1080 as originally assumed.
- [x] CUDA/driver: KMD 610.62, CUDA UMD 13.3, confirmed via `nvidia-smi`.
- [x] Exact free disk space: 851GB free of 930GB on C:, confirmed via `Get-PSDrive`.
- [x] Conda install completed, `conda activate land-ct-v2` confirmed working.
- [x] `torch.cuda.is_available()` → `True`, correctly reports the RTX 2080 Ti (torch 2.6.0+cu124).
- [x] `pylidc` + `lungmask` installed, `setuptools<81` pin applied (Bug 001, resolved).

No teammate-collision issues have surfaced so far (shared-account risk noted above), but keep
checking `~/.pylidcrc` before any run if that changes.
