# services/

Shell wrappers and process-supervision scripts. The two systemd user
units this repo's live surface depends on:

- `ai-engine.service` -> runs `app/engine.py` (was root `ai_signal_engine.py`)
- `gold-shadow.service` -> runs `app/shadow.py` (was root `shadow_engine.py`)

**Both are stopped as of Phase 1 (2026-08-18) and were not restarted by
this plan.** The actual `systemctl --user` unit definitions live outside
this repo (`systemd-run --user --unit=...`, per each script's own launch
command) -- this directory holds the scripts those units invoke, not the
unit files themselves. Restarting them against the new `app/` paths is a
deliberate later-phase decision, not a Phase 1 deliverable.

**Known pre-existing issue, not fixed in Phase 1:** `model_staleness_watch.sh`
monitors a canonical file set (`features.json`, `ensemble.json`,
`gold_lgb_model_s{42,7,2026}.txt`) from the old v7 LightGBM ensemble --
already superseded before this rebuild (the 2026-08-17 CatBoost pivot
predates Phase 1). Those files now live under `models/archive/legacy-v7/`.
Rewriting this script against `models/registry/`'s active entries is a
real product decision (what "staleness" means for a registry-based
system) deferred to a later phase, not a mechanical path fix -- left
as-is rather than silently patched.
