"""
ML-Based Signal Strategy
Uses a pre-trained ML model to generate trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import joblib  # For loading scikit-learn models
from .base_strategy import BaseStrategy, Signal, SignalType

class MLSignalStrategy(BaseStrategy):
    def __init__(self, symbols: List[str], model_path: str, feature_list: List[str], capital: float = 100000, **kwargs):
        super().__init__("MLSignal", symbols, capital, **kwargs)
        self.model = joblib.load(model_path)  # Placeholder: path to pre-trained model
        self.feature_list = feature_list

    def compute_features(self, data: pd.DataFrame) -> np.ndarray:
        # Example: compute features for the last row
        features = []
        for feature in self.feature_list:
            if feature == "roc_20":
                features.append((data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20])
            elif feature == "vol_20":
                features.append(data['close'].rolling(20).std().iloc[-1])
            elif feature == "zscore_20":
                mean = data['close'].rolling(20).mean().iloc[-1]
                std = data['close'].rolling(20).std().iloc[-1]
                features.append((data['close'].iloc[-1] - mean) / std if std > 0 else 0)
            # Add more features as needed
        return np.array(features).reshape(1, -1)

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        if len(data) < max(20, max([int(f.split('_')[-1]) for f in self.feature_list if '_' in f] + [1])):
            return None
        features = self.compute_features(data)
        prob = self.model.predict_proba(features)[0]  # Assumes binary classifier
        # Example: prob[1] is probability of 'buy', prob[0] is 'sell'
        if prob[1] > 0.7:
            return Signal(
                timestamp=data.index[-1],
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=prob[1],
                price=data['close'].iloc[-1],
                confidence=prob[1],
                metadata={"ml_prob": prob[1]}
            )
        elif prob[0] > 0.7:
            return Signal(
                timestamp=data.index[-1],
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=prob[0],
                price=data['close'].iloc[-1],
                confidence=prob[0],
                metadata={"ml_prob": prob[0]}
            )
        return None 