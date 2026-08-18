# market/

Holds the MT5 feed connector (`xm_ticker.py`, relocated from repo root in
Phase 1). **This is not integrated live infrastructure yet.**

Today's actual live contract is: `xm_ticker.py` runs as an unmanaged
external process, writing state files (`xm_tick_state.json`,
`.active_signal_ai.json`, `xm_live_bars.jsonl`) to an external directory
(`/home/jith/.hermes/profiles/trading/cron/output/`, see
`config/market.yaml`'s `state_dir`) that `app/` polls. This repo doesn't
manage or version that process's lifecycle.

This is a known-bad interim state, called out explicitly rather than
quietly relied on. Phase 2 replaces it with a real `MarketState`
(`contracts/market_state.py`)-producing pipeline that `app/` owns
directly.
