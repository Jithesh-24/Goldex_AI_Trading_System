"""
Turns a feature row + trained primary/meta models into a concrete trade
signal (side, entry, TP, SL, confidence) or None. This is the ONLY place
live-engine and backtest code should compute a signal, so both paths are
guaranteed to agree.

No hardcoded pip/dollar distances anywhere: TP/SL are the same vol-scaled
barrier widths (pt_mult*vol, sl_mult*vol) the models were trained against,
read from feature_cols.json so training and inference can never silently
drift apart.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


@dataclass
class Signal:
    side: int          # +1 long, -1 short
    entry: float
    tp: float
    sl: float
    p_win: float        # meta model's P(TP before SL)
    primary_proba: float  # primary model's proba for the fired class


class SignalEngine:
    def __init__(self, router, meta_prob_threshold: float):
        """router: a decision.router.ModelRouter already configured with
        config/models.yaml's role map. meta_prob_threshold: the decision
        policy threshold from config/decision.yaml -- kept separate from
        the router/registry because it's a tunable operating parameter,
        not something training-locked to a specific model artifact."""
        direction_entry = router.resolve("direction")
        meta_entry = router.resolve("opportunity_meta")
        if direction_entry is None or meta_entry is None:
            raise RuntimeError(
                "SignalEngine requires both 'direction' and 'opportunity_meta' "
                "roles configured in config/models.yaml"
            )
        self.primary_cols = direction_entry.feature_cols
        self.meta_cols = meta_entry.feature_cols
        self.pt_mult = meta_entry.training_config["tb_cfg_trade"]["pt_mult"]
        self.sl_mult = meta_entry.training_config["tb_cfg_trade"]["sl_mult"]
        self.horizon_vol_scale = meta_entry.training_config["horizon_vol_scale"]
        self.max_holding = meta_entry.training_config["max_holding"]
        self.meta_prob_threshold = meta_prob_threshold

        self.primary = CatBoostClassifier()
        self.primary.load_model(router.artifact_path("direction"))
        self.meta = CatBoostClassifier()
        self.meta.load_model(router.artifact_path("opportunity_meta"))

    def score(self, feat_row: pd.Series, close: float, vol: float, is_cusum_event: bool,
              prob_threshold: float = None) -> Signal | None:
        """feat_row: a single row of Tier1+Tier2 features (same columns
        build_features produces). `vol` = raw per-bar ewma_vol (feat_row's
        own "ewma_vol" column) -- this method applies the same
        horizon_vol_scale*sqrt(max_holding) scaling used in training, so the
        caller never has to remember to pre-scale it. `is_cusum_event` = has
        the live CUSUM filter (same threshold as training: cusum_k *
        ewma_vol * close) fired on this bar -- primary is binary (up/down,
        no flat class) so the event gate is the ONLY "is there an
        opportunity at all" check; call this only when it's True. Returns
        None if no event, or meta confidence is below threshold -> no
        trade, stay flat."""
        if not is_cusum_event:
            return None
        thresh = prob_threshold if prob_threshold is not None else self.meta_prob_threshold
        vol = vol * self.horizon_vol_scale * (self.max_holding ** 0.5)

        x_primary = feat_row[self.primary_cols].to_frame().T
        proba = self.primary.predict_proba(x_primary)[0]  # [down, up]
        cls = int(np.argmax(proba))
        side = 1 if cls == 1 else -1
        primary_proba = float(proba[cls])

        x_meta = feat_row[self.primary_cols].copy()
        x_meta["assumed_side"] = float(side)
        x_meta = x_meta[self.meta_cols].to_frame().T
        p_win = float(self.meta.predict_proba(x_meta)[0, 1])

        if p_win < thresh:
            return None  # primary sees an opportunity, meta says not precise enough

        if side == 1:
            tp = close * (1 + self.pt_mult * vol)
            sl = close * (1 - self.sl_mult * vol)
        else:
            tp = close * (1 - self.pt_mult * vol)
            sl = close * (1 + self.sl_mult * vol)

        return Signal(side=side, entry=close, tp=tp, sl=sl,
                      p_win=p_win, primary_proba=primary_proba)
