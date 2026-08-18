# journal/

Journal event contracts (`contracts/journal.py`, re-exported here for
convenience) are defined in Phase 1. The actual journal files
(`trade_journal_ai.jsonl`, `live_outcomes.jsonl`) still live in the
external `cron/output/` directory (see `config/journal.yaml`'s
`output_dir`) and are still written inline by `app/engine.py`/
`app/shadow.py` using their existing ad hoc dict shapes, not these
pydantic contracts yet.

Wiring the live engines to actually construct `SignalEvent`/
`ResolutionEvent`/etc. instances and moving the output location into this
repo's control is explicitly deferred past Phase 1 (the original spec's
Step 9 defines the schema; adopting it in the write path is separate
work).
