# v1 Reference — What to Port vs. Rebuild

v1's BRAIN pack (the "LAND-CT-Synthesis — Project Brain v9") is a completed, honestly-scoped
partial reproduction: preprint version, 72 patients, lung-only mask, no mask VAE. Keep it
accessible (copy it onto this PC too, read-only) and cite specific lessons from it — don't rebuild
its findings from scratch, and don't build v2 as a literal fork of its code.

## Port over (still true, still useful)

- All Windows/environment bugs (pickling, OMP conflict, cuDNN benchmark flag) — same OS, same
  class of bug will recur.
- The checkpoint-discipline lesson (track *best*, not *latest* — v1 Decision 0007).
- The "verify, don't trust" operating rule generally.
- Min-SNR-γ=5.0 and additive-skip-connection confirmations — already verified against the
  official repo in v1, no need to re-derive.
- The published-paper citation details (Scientific Reports, DOI, author list, official repo URL)
  — already researched in v1, reusable as-is.

## Rebuild, don't port

- Preprocessing code — v1's only extracted lung masks; this project needs the full nodule +
  lung + texture mask pipeline, which is a different code path, not an extension of the old one.
- Dataset — full 1,010 patients vs. 72; don't assume v1's cached tensors are reusable, they were
  built for a different mask scheme.
- Any config/checkpoint files — architecture may be extended (mask VAE is new), so v1 checkpoints
  aren't a valid init for anything in this project.

## Explicitly unresolved from v1, now relevant again

- v1's `docs/06_open_questions.md` — check this before assuming a question is new; some may
  already have been thought through, even if unresolved.
