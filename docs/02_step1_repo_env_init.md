# Step 1 — Repo Init + Environment Setup

*Run these on the new PC, in order. Report back what actually happens at each step — especially
anything that errors or looks different from expected — so this file (and BRAIN/chats/) captures
the real state, not the assumed one.*

## 1. Confirm the machine, before installing anything

```cmd
nvidia-smi
```
Confirm: GPU model (should say GTX 1080), VRAM (should say 8GB), driver version, CUDA version
shown at the top. Paste the output back — don't just eyeball it, this becomes part of the record.

## 2. Check your teammate's setup first (shared PC discipline)

```cmd
conda env list
dir C:\Users\%USERNAME%\ 2>nul
```
Just to see what conda envs already exist (so `land-ct-v2` doesn't collide with anything they've
named similarly) and confirm you're on your own Windows user account, not sharing one login.

## 3. Create the environment

```cmd
conda create -n land-ct-v2 python=3.10
conda activate land-ct-v2
```

## 4. Git — local config only (mandatory on a shared machine)

```cmd
git init
git config --local user.email "your-email@..."
git config --local user.name "Zunaira3010"
git config --local --list
```
That last command should show your name/email scoped to *this repo only* — confirm it, don't
assume. If `git config --global` was ever run here before (even by your teammate), a stray global
config could still leak in; `--local --list` output is what actually matters.

## 5. Core installs

```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
This should print `True` and `NVIDIA GeForce GTX 1080` (or similar). If `cuda.is_available()` is
`False`, stop here — that's a driver/CUDA mismatch to fix before anything else.

## 6. Pull the official LAND repo as reference (not to build on top of — to verify against)

```cmd
git clone https://github.com/aolivtous/LAND_3DChestCT.git reference_official_repo
```
This is the same repo v1 used to confirm Min-SNR-γ=5.0 and the additive-skip design — now it's
also the source of truth for the mask-VAE architecture and the exact preprocessing spec, since
this project is implementing the full pipeline, not the simplified v1 one.

---

**Report back:** the `nvidia-smi` output, the `torch.cuda` check result, and whether the git
clone worked cleanly. I'll log whatever's confirmed into BRAIN/00_INDEX.md and we'll move to
Step 2 (preprocessing spec verification + starting the LIDC-IDRI download).
