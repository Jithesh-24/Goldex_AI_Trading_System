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
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Signal:
    side: int          # +1 long, -1 short
    entry: float
    tp: float
    sl: float
    p_win: float        # meta model's P(TP before SL)
    primary_proba: float  # primary model's proba for the fired class


class SignalEngine:
    def __init__(self, model_dir: str = os.path.join(BASE, "models")):
        with open(os.path.join(model_dir, "feature_cols.json")) as f:
            meta_cfg = json.load(f)
        self.primary_cols = meta_cfg["primary"]
        self.meta_cols = meta_cfg["meta"]
        self.pt_mult = meta_cfg["tb_cfg"]["pt_mult"]
        self.sl_mult = meta_cfg["tb_cfg"]["sl_mult"]
        self.meta_prob_threshold = meta_cfg["meta_prob_threshold"]

        self.primary = CatBoostClassifier()
        self.primary.load_model(os.path.join(model_dir, "primary.cbm"))
        self.meta = CatBoostClassifier()
        self.meta.load_model(os.path.join(model_dir, "meta.cbm"))

    def score(self, feat_row: pd.Series, close: float, vol: float,
              prob_threshold: float = None) -> Signal | None:
        """feat_row: a single row of Tier1+Tier2 features (same columns
        build_features produces). Returns None if primary is flat or meta
        confidence is below threshold -> no trade, stay flat."""
        thresh = prob_threshold if prob_threshold is not None else self.meta_prob_threshold

        x_primary = feat_row[self.primary_cols].to_frame().T
        proba = self.primary.predict_proba(x_primary)[0]  # [down, flat, up]
        cls = int(np.argmax(proba))
        if cls == 1:
            return None  # flat -> no directional opportunity

        side = 1 if cls == 2 else -1
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
