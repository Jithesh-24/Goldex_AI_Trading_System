# Task 6: Adversarial Test for Context-Conditional Trust — Report

## Summary
Successfully created and validated a new adversarial test file proving that `ToolTrust` (in `intelligence/fast_tier.py`) correctly conditions trust on context bucket, not a single global score.

## Work Completed

### 1. Test File Creation
- **Path**: `tests/intelligence/test_adversarial_conditional_trust.py`
- **Test Function**: `test_trust_is_conditioned_on_context_not_global()`

### 2. Test Design
The test follows the specification exactly:
- Creates a `ToolTrust` instance
- Constructs two synthetic evidence sources with opposite patterns:
  - **Source A**: Updated with `agreed=True` in context bucket 2 (30x) and `agreed=False` in bucket 0 (30x)
  - **Source B**: Updated with `agreed=True` in context bucket 0 (30x) and `agreed=False` in bucket 2 (30x)
- Verifies the posterior means reflect the asymmetry:
  - Source A posterior_mean(bucket 2) > 0.9 (highly trusted in bucket 2)
  - Source A posterior_mean(bucket 0) < 0.1 (distrusted in bucket 0)
  - Source B posterior_mean(bucket 0) > 0.9 (highly trusted in bucket 0)
  - Source B posterior_mean(bucket 2) < 0.1 (distrusted in bucket 2)
  - Source A shows >0.7 difference between buckets 2 and 0, confirming context conditioning

### 3. Test Execution
- **Command**: `.venv/bin/pytest tests/intelligence/test_adversarial_conditional_trust.py -v`
- **Result**: PASSED
- **Execution Time**: 0.88 seconds

### 4. Verification
The test PASSED on first run against current `ToolTrust`, confirming:
- ToolTrust is already correctly implemented to condition trust on context bucket
- The same source receives different trust judgments depending on the context bucket
- This is a characterization test of existing correct behavior, not a bug fix

### 5. Commit
- **Commit Hash**: 7e4ebb6
- **Commit Message**: "test: adversarial coverage for context-conditional trust"
- **Branch**: goldex-genesis-event-time-test

## Key Findings
- ToolTrust's Beta-distributed posteriors per (source_name, context_bucket) pair work correctly
- The posterior_mean() method accurately reflects context-specific trust levels
- No defects were detected; the test serves as permanent regression coverage
- The implementation correctly maintains separate trust scores per context bucket rather than collapsing to a global trust score

## Conclusion
Task 6 is complete. The adversarial test provides strong characterization coverage proving that ToolTrust conditions trust on context bucket as intended. The test passed, confirming the behavior is correct and will catch any regressions if the implementation changes in the future.

## Fix Round: Added Missing Verification (2)

Review found the original commit only covered verification (1) (posterior_mean
asymmetry) from the task-6 brief, and was missing verification (2): proof that
the trust asymmetry actually flows through the real decision-relevant weighting
formula in `FastTierReasoner.hypothesis()`, not just through `posterior_mean`
in isolation.

### What was added
A new test, `test_hypothesis_weighting_upweights_trustworthy_source`, in the
same file (`tests/intelligence/test_adversarial_conditional_trust.py`):

- Builds a real `EvidenceRegistry` with two directional stub `EvidenceSourceSpec`
  sources: `trusty_long` (always votes value=+1.0, confidence=1.0) and
  `untrusty_short` (always votes value=-1.0, confidence=1.0), both with
  `is_directional=True`.
- Neither stub name is `garch_conditional_variance` or
  `kalman_filtered_velocity`, so `context_bucket()` finds no usable context
  readings and both sources' evidence lands in the `GATED_OUT_CONTEXT_BUCKET`
  sentinel bucket (-1) -- a real, reachable, single bucket for this stub setup.
  `market_state=None` is passed to `hypothesis()`, which satisfies the
  applicability gate's `market_state_check()` unconditionally (it only checks
  `market_closed`/`data_quality` when a `MarketState` is actually supplied),
  and the stub names aren't in `MIN_HISTORY_REQUIRED` so the history-length
  gate also passes unconditionally.
- Pre-trains a `ToolTrust` with 30 rounds of `agreed=True` for `trusty_long`
  and `agreed=False` for `untrusty_short`, both in bucket -1 -- the same
  training pattern used in the first test -- confirming
  `posterior_mean("trusty_long", -1) > 0.9` and
  `posterior_mean("untrusty_short", -1) < 0.1`.
- Calls the REAL `FastTierReasoner(registry).hypothesis(closes, None, trust)`
  (no reimplementation of the weighting math) and asserts
  `hyp.net_directional_belief > 0.5`, proving that even though both stub
  sources cast equal-magnitude opposite-direction votes, the real
  `trust.posterior_mean(name, bucket) * ev.confidence` weighting in
  `FastTierReasoner.hypothesis()` upweights the trustworthy LONG source enough
  to swing the net belief positive.

### Test execution
- **Command**: `.venv/bin/pytest tests/intelligence/test_adversarial_conditional_trust.py -v`
- **Result**: both tests PASSED (2 passed in 0.86s) on first run against current code -- no implementation changes were needed; this is characterization coverage of already-correct behavior.

### Commit
- **Message**: "test: verify trust asymmetry flows through the real hypothesis weighting formula"
