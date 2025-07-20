"""
Regime Detector Utility
Classifies market regime using rolling statistics or ML
"""

import pandas as pd
import numpy as np
from typing import Literal

RegimeType = Literal["trending", "mean_reverting", "volatile", "calm"]

def detect_regime(data: pd.DataFrame, lookback: int = 50) -> RegimeType:
    """
    Classify market regime based on rolling statistics
    Returns: 'trending', 'mean_reverting', 'volatile', or 'calm'
    """
    if len(data) < lookback:
        return "calm"
    close = data['close']
    returns = close.pct_change().dropna()
    volatility = returns.rolling(lookback).std().iloc[-1]
    # Trend strength via linear regression slope
    x = np.arange(lookback)
    y = close.iloc[-lookback:].values
    slope = np.polyfit(x, y, 1)[0]
    # Heuristics for regime
    if volatility > 0.02:
        return "volatile"
    elif abs(slope) > 0.001:
        return "trending"
    elif volatility < 0.005:
        return "calm"
    else:
        return "mean_reverting" 