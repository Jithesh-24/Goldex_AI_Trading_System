# GOLEX V3 — PHASE 4: SPECIALIST QUANTITATIVE MODEL LAYER

> Verbatim design spec as approved by the user on 2026-08-22. Phases 1-3 are
> complete and approved. This document is Phase 4's design/requirements
> spec; the implementation plan derived from it lives at
> `docs/superpowers/plans/2026-08-22-golex-v3-phase4-specialist-models.md`.

Phase 4 creates and validates the SPECIALIST MODEL LAYER that sits on top of
the V3 quantitative feature fabric.

This is the point where V3 moves from:

```
MARKET STATE
    v
QUANTITATIVE FEATURES
```

to:

```
QUANTITATIVE FEATURES
    v
SPECIALIST AI MODELS
    v
PROBABILISTIC / DISTRIBUTIONAL MARKET INFORMATION
```

The objective is NOT to find one "most powerful AI model."

The objective is: USE THE BEST VALIDATED MODEL FOR EACH SPECIFIC
QUANTITATIVE PROBLEM.

## 1. Phase 4 core principle

Do NOT lock V3 to CatBoost / LightGBM / XGBoost / Transformer / LSTM / TCN /
any single ML architecture. CatBoost is the existing directional baseline
and must remain the benchmark. The final model for each specialist role
must be determined empirically. The architecture must support different
models for different responsibilities:

```
Direction              -> Model A
Opportunity / Meta     -> Model B
Regime                 -> Model C
MAE                    -> Model D
MFE                    -> Model E
Barrier probability    -> Model F
Execution/signal decay -> Model G
```

These may eventually be the same model family or completely different
model families. Do not assume. Research and validate.

## 2. Specialist model roles

### A. Direction model
"Conditional on the current market state, what is the probability of a
meaningful upward/downward move within the target horizon?" Existing
deployed CatBoost direction model is the BASELINE. Do NOT replace it
without evidence. Evaluate whether the V3 quantitative feature fabric
improves or worsens it. Candidate families: CatBoost, LightGBM, XGBoost,
calibrated gradient boosting, probabilistic classifiers, temporal models,
TCN, Transformer, state-space approaches, ensembles — only evaluate models
appropriate to available data/compute.

### B. Opportunity / meta model
"Even if direction has an edge, is this particular opportunity worth
taking?" Must NOT simply duplicate the direction model — model trade
opportunity quality: conditional success probability, probability of
reaching a meaningful favourable barrier, probability of adverse
excursion, opportunity decay, market-state quality. Use the existing meta
model as baseline.

### C. Regime model
"What statistical state/regime is the market currently in?" Possible
states: directional/trending, mean-reverting, compressed, expanding, high
vol, low vol, jump/shock, unstable transition, activity expansion, spread
expansion. Do NOT hard-code arbitrary labels — research statistically
defensible regime definitions. Evaluate HMM, Markov models, clustering,
probabilistic state models, tree-based classification, state-space models.
A regime model is useful only if it adds predictive/decision value — do
not build one merely because the architecture has a "regime" box.

### D. MAE quantile model (high priority)
"Given the current state and a proposed trade direction, what adverse
excursion distribution should we expect?" Do NOT reduce to one fixed SL
multiplier — model the conditional distribution of MAE. Investigate q50/
q75/q90/q95 where sample size supports them. Candidates: CatBoost/LightGBM
quantile regression, gradient boosting quantile models, conformal
approaches if appropriate, distributional models. Use actual OOS evidence.

### E. MFE quantile model
"Given the current state and proposed direction, what favourable excursion
distribution is available?" Investigate q50/q75/q90/q95 where justified.
Should help determine future TP construction — do not implement final TP
decision logic yet.

### F. Barrier probability model
"What is the probability that price reaches a defined favourable barrier
before an adverse barrier within a specified horizon?" Research
competing-risk approaches, discrete-time hazard models, probabilistic
classifiers, survival methods, first-passage models, gradient boosting
probability models. Target definition must be carefully researched. Do not
use future information in features.

### G. Execution / signal decay model
"How quickly does a detected opportunity decay after the signal state
appears?" Matters because the system produces Telegram signals and the
user manually executes them. Research time-to-decay, post-signal drift,
probability decay, expected adverse movement after delay,
execution-latency sensitivity. Do not fabricate execution data. If
historical human execution data is insufficient, build the model
infrastructure but classify it as data-limited until real execution
observations accumulate.

## 3. First task — research

Before implementing models, perform serious quantitative research. For
each specialist role: (1) define the prediction problem, (2) define the
target mathematically, (3) define the forecast horizon, (4) define the
available information at prediction time, (5) define candidate model
families, (6) identify assumptions, (7) identify data requirements, (8)
identify leakage risks, (9) identify calibration requirements, (10)
identify computational requirements, (11) identify appropriate evaluation
metrics, (12) compare candidate approaches, (13) select candidates for
empirical testing. Do not implement a model simply because it is
fashionable.

## 4. Target design is more important than model choice

Before comparing models, rigorously define targets. For every target
document: TARGET, HORIZON, ENTRY REFERENCE, BARRIER DEFINITION, LABELING
METHOD, CENSORING, TIMEOUT, DATA REQUIREMENTS, CAUSALITY, LEAKAGE RISKS.
The target must represent the actual trading question — avoid arbitrary
labels.

## 5. Multi-horizon design

Short-horizon gold signal system — do not optimize only for one arbitrary
horizon. Research multiple practical horizons (very short / short /
medium-short). Use actual market behaviour and data resolution to
determine appropriate values. Do not create dozens of horizons merely to
increase model count — each horizon must have a reason.

## 6. Feature selection by specialist

Phase 3's fabric has 125 registered features (28 REQUIRED baseline, 17
USEFUL candidates, 5 OPTIONAL live microstructure, 75 REDUNDANT/
research-only). DO NOT feed all 125 into every model — each specialist
must have its own feature schema, selected by OOS importance, redundancy,
stability, leakage safety, model robustness, computational cost, economic
relevance. Do not select features merely because they have high in-sample
importance.

## 7. New live microstructure features

Phase 3's five live-only microstructure features currently have
synthetic-only evidence. They MUST be validated against real XM tick data
before being considered useful. Do not automatically add them to
production models. Evaluate distribution, stability, variance, usefulness,
redundancy, OOS predictive contribution. If they do not demonstrate value,
leave them OPTIONAL or reject them.

## 8. Model candidate research

Consider where technically justified: tree boosting (CatBoost/LightGBM/
XGBoost), probabilistic (calibrated classifiers, probabilistic regression,
survival/hazard models, distributional models), quantile (quantile
boosting, conditional quantile regression, conformal methods),
sequential (TCN, Transformer, state-space, HMM/Markov), ensembles
(weighted, stacking, specialist ensembles). Do NOT implement every
candidate automatically — use research to determine which deserve
empirical testing.

## 9. Important model selection rule

"Best" is NOT highest training accuracy. Selection criteria: walk-forward
OOS performance, calibration, stability across periods/regimes,
performance after realistic transaction cost/spread, drawdown
characteristics where relevant, tail behaviour, probability quality,
feature stability, computational latency, memory requirements, complexity,
reproducibility. A model 0.5% better but unstable and 20x slower may be
inferior.

## 10. Walk-forward validation

Causal walk-forward only, never shuffle financial time-series data.
Respect TRAIN -> VALIDATION -> TEST and rolling/expanding walk-forward
structure. Document training/validation/test windows, embargo/purge if
needed, feature warmup, retraining schedule, prediction horizon. Avoid
leakage through overlapping labels.

## 11. Purging / embargo

Because some targets overlap future horizons, research and implement
appropriate purging/embargo/event-overlap handling where necessary. Do not
claim independent samples when labels overlap.

## 12. Evaluation — direction

Not accuracy alone: log loss, Brier score, calibration error, ROC-AUC/
PR-AUC where meaningful, precision/recall at useful operating regions,
probability distribution, regime stability, economic performance when
converted into the existing trade framework. Existing direction CatBoost
remains the baseline.

## 13. Evaluation — quantile models

For MAE/MFE: pinball loss, empirical coverage, calibration of quantiles,
interval coverage, sharpness, conditional coverage, regime coverage, tail
coverage. A q90 that covers 90% globally but fails badly in high-vol
regimes is not sufficient.

## 14. Evaluation — barrier models

Log loss, Brier, calibration curve, reliability, discrimination, horizon
stability, regime stability. Do not judge only by classification accuracy.

## 15. Evaluation — regime models

Regime persistence, regime stability, transition behaviour, separation of
statistical distributions, downstream predictive usefulness, robustness
across time. A visually attractive regime chart is not evidence of trading
value.

## 16. Calibration

Every probabilistic specialist must have a calibration strategy: Platt,
isotonic, beta calibration, temperature scaling where appropriate, or
other suitable methods. Calibration learned only from appropriate
training/validation data — never calibrate on the final test period.

## 17. Baseline-first rule

Every specialist must have a simple baseline (Direction: existing
CatBoost; MAE/MFE: historical conditional quantile baseline; Barrier:
simple empirical probability baseline; Regime: simple statistical state
baseline; Opportunity: existing meta model). A complex model must beat its
baseline OOS to survive.

## 18. Model registry

Extend the Phase 1 model registry. Every candidate model must record:
model_id, role, model_family, target, horizon, feature_schema, training
period, validation period, test period, hyperparameters, calibration
method, metrics, creation timestamp, data version, status, artifact path.
Statuses (minimum): CANDIDATE, VALIDATED, CHAMPION, REJECTED, ARCHIVED. Do
NOT implement automatic champion promotion yet.

## 19. Model router

Extend `decision/router.py` only as an architectural seam: "for this
specialist role, which approved model artifact should be loaded?" Must NOT
dynamically choose a model based on a few recent trades — model selection
belongs to research validation. The live router should consume an
approved model registry configuration. Do not connect all specialist
outputs to the production signal decision yet.

## 20. Model versioning

A model artifact must be inseparable from its feature schema, target
definition, horizon, calibration, training data version, code version. A
model must fail validation if its required feature schema does not match.

## 21. Data leakage audit

Before accepting any model, test for: future feature leakage, label
leakage, overlapping target leakage, normalization leakage, calibration
leakage, train/test contamination, feature-selection leakage,
model-selection leakage. Create automated tests wherever possible.

## 22. Real-data microstructure validation

Use the Phase 2 real XM tick pipeline. Evaluate the five Phase 3 live-only
features using actual observations — do NOT substitute synthetic replay
for this analysis (synthetic may be used for unit/performance tests only).
Document sample size, market period, distribution, missingness, stability,
usefulness, redundancy.

## 23. MAE/MFE priority

Treat conditional MAE/MFE quantile modelling as HIGH-PRIORITY, but do not
assume it wins. Compare simple conditional empirical quantiles, CatBoost
quantile, LightGBM quantile, other justified approaches. Output should
eventually answer: expected adverse excursion q50/q75/q90/q95, expected
favourable excursion q50/q75/q90/q95. Do not yet turn these into
production SL/TP.

## 24. Distributional thinking

The final V3 system should not rely exclusively on P(win) — it should
eventually understand P(direction), P(opportunity), P(MAE distribution),
P(MFE distribution), P(barrier), P(decay), P(regime). Phase 4 establishes
these specialist outputs; Phase 5 will combine them. Do NOT implement the
final combination/EV engine in Phase 4.

## 25. Model ensembles

Research whether ensembles improve specialist performance. Only ensemble
if errors are sufficiently complementary, OOS performance improves,
calibration improves or remains sound, latency remains acceptable,
complexity is justified. A single strong model is preferable to a
pointless ensemble.

## 26. Sequential models

Do not automatically implement Transformers/TCNs/LSTMs. First determine
whether data resolution supports them, whether sample size supports them,
whether the signal is genuinely sequential beyond engineered state
features, latency requirements, training complexity, OOS evidence. If not
justified, document as evaluated/rejected/deferred. If justified, implement
as a candidate challenger, not automatically as production champion.

## 27. No data fabrication

Never fabricate trade tape, order flow, order-book data, execution delay,
volume, historical spread, unavailable market variables. If a specialist
requires unavailable data: mark it DATA_LIMITED / UNSUPPORTED_BY_DATA.

## 28. No production decision changes

Do NOT change production signal behaviour yet: production entry
threshold, production SL, production TP, Telegram signal, current
direction model, current meta model must all remain unchanged. Specialist
models may run in research/replay/shadow/candidate evaluation only.

## 29. Performance

Benchmark each specialist: inference latency, batch training time,
memory, model size, feature preparation cost, cold start, warm inference.
Compare models on both predictive value and operational cost. The live
system must remain responsive to tick-level market updates.

## 30. Testing

At minimum: TARGET CORRECTNESS (known synthetic market paths -> expected
labels), CAUSALITY (features/targets never use future information),
PURGING (overlapping labels handled correctly), WALK-FORWARD (no
train/test contamination), CALIBRATION (known probabilities -> expected
calibration behaviour), QUANTILE COVERAGE (known distributions -> expected
coverage), MODEL SCHEMA (correct feature schema enforced), MODEL REGISTRY
(artifacts/metadata validate), ROUTER (role -> approved model mapping
works), MODEL FAILURE (missing/incompatible model fails safely),
MICROSTRUCTURE (real-data validation pipeline works), REPRODUCIBILITY
(same dataset/config -> reproducible result within tolerance), PERFORMANCE
(inference benchmark recorded).

## 31. Research artifacts

Every model experiment must leave reproducible evidence: experiment
configuration, target definition, feature schema, model parameters,
training/validation period, OOS predictions, metrics, calibration, feature
importance where applicable, model artifact, rejection reason if rejected.
Do not leave model decisions only in terminal output.

## 32. What Phase 4 must NOT do

Final EV gate, final trade decision fusion, dynamic production SL, dynamic
production TP, automatic trade management, EOD learning, automatic
retraining, automatic champion promotion, final Telegram redesign. Those
belong to later phases.

## 33. Completion criteria

Phase 4 is complete only when: specialist target definitions are
documented; candidate model families are researched; direction baseline is
established; opportunity/meta baseline is established; regime modelling
has been evaluated; MAE modelling is evaluated; MFE modelling is
evaluated; barrier modelling is evaluated where data supports it;
execution/decay modelling is evaluated or explicitly marked data-limited;
real XM microstructure features are evaluated; model-specific feature
schemas exist; model registry supports specialist roles; model routing
supports specialist roles; walk-forward OOS validation exists; calibration
exists where applicable; leakage tests pass; quantile coverage is
measured; model performance is compared against baselines; model
operational cost is measured; candidate/champion/rejected statuses are
recorded; production decision behaviour remains unchanged; tests pass;
documentation is complete. Do not declare Phase 4 successful merely
because models trained successfully — a specialist model survives only if
it demonstrates genuine OOS value and operational suitability.

## 34. Final Phase 4 report format

A. RESEARCH — what model families were investigated and why.
B. TARGETS — exact target definitions for every specialist.
C. BASELINES — baseline used for each role.
D. MODEL COMPARISON — candidate models and OOS results.
E. FINAL MODEL SELECTION — per role: ROLE, CHOSEN MODEL, WHY, OOS METRICS,
   CALIBRATION, FEATURE SCHEMA, LATENCY, LIMITATIONS.
F. DIRECTION — comparison against existing CatBoost baseline.
G. OPPORTUNITY — comparison against existing meta model.
H. REGIME — whether a regime specialist justified itself.
I. MAE — quantile results and coverage.
J. MFE — quantile results and coverage.
K. BARRIER — probability-model results.
L. EXECUTION / DECAY — results or explicit data limitation.
M. MICROSTRUCTURE — real XM-data validation of the five live-only
   features.
N. MODEL ROUTING — final role -> model mapping.
O. REGISTRY — candidate/champion/rejected model inventory.
P. LEAKAGE — tests and results.
Q. PERFORMANCE — inference/training/resource benchmarks.
R. LIMITATIONS — what remains unresolved.
S. NEXT PHASE — recommend Phase 5 only; do not implement Phase 5
   automatically.

## Final principle

Phase 4 creates the INTELLIGENCE SPECIALISTS of V3 — independent
quantitative specialists answering different questions (where is price
likely to go -> is this opportunity worth taking -> what regime are we in
-> how much adverse excursion is expected -> how much favourable excursion
is available -> what is the probability of reaching each barrier -> how
fast does the opportunity decay). Only after those answers exist will
Phase 5 combine them into the final quantitative decision:

```
MARKET
  v
MARKETSTATE
  v
QUANTITATIVE FEATURE FABRIC
  v
SPECIALIST MODELS
  v
CALIBRATED PROBABILITIES / DISTRIBUTIONS
  v
PHASE 5 DECISION ENGINE
```

Optimize for: causality, OOS evidence, calibration, stability, robustness,
low latency, economic relevance, model appropriateness. Research first.
Then implement. Then validate. Then document.
