"""
Meta-Strategy/Ensemble Module
Aggregates signals from multiple strategies and selects/weights them
"""

from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from .base_strategy import BaseStrategy, Signal, SignalType

class MetaStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy], symbols: List[str], capital: float = 100000, meta_model=None, **kwargs):
        super().__init__("MetaStrategy", symbols, capital, **kwargs)
        self.strategies = strategies
        self.meta_model = meta_model  # Optional: ML model for meta-decision

    def generate_signal(self, symbol: str, data: Dict[str, any]) -> Optional[Signal]:
        # Collect signals from all strategies
        signals = []
        for strat in self.strategies:
            if hasattr(strat, 'generate_signal'):
                sig = strat.generate_signal(symbol, data.get(symbol, data))
                if sig:
                    signals.append(sig)
        if not signals:
            return None
        # Meta-model logic (placeholder: simple voting)
        votes = {SignalType.BUY: 0, SignalType.SELL: 0, SignalType.HOLD: 0}
        for sig in signals:
            votes[sig.signal_type] += sig.confidence if hasattr(sig, 'confidence') else 1
        best_type = max(votes, key=votes.get)
        # Optionally use meta_model to select/weight signals
        # if self.meta_model: ...
        # Aggregate metadata/confidence
        avg_conf = np.mean([sig.confidence for sig in signals if hasattr(sig, 'confidence')])
        return Signal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type=best_type,
            strength=avg_conf,
            price=data[symbol]['close'].iloc[-1] if isinstance(data[symbol], dict) and 'close' in data[symbol] else None,
            confidence=avg_conf,
            metadata={"meta_votes": votes, "sub_signals": signals}
        ) 