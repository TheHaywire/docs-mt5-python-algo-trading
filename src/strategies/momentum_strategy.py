"""
Momentum Strategy for MT5 Python Algo Trading System
Captures trending moves with regime detection and adaptive parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy that captures trending moves using:
    - Multi-timeframe momentum analysis
    - Regime detection (trending vs. ranging)
    - Volume confirmation
    - Adaptive parameters
    - Breakout detection
    """
    
    def __init__(self, 
                 symbols: List[str],
                 capital: float = 100000,
                 momentum_lookback: int = 20,      # Periods for momentum calculation
                 regime_lookback: int = 50,        # Periods for regime detection
                 volume_confirmation: bool = True, # Require volume confirmation
                 breakout_threshold: float = 0.02, # Breakout threshold (2%)
                 trend_strength_threshold: float = 0.6, # Minimum trend strength
                 **kwargs):
        
        super().__init__("Momentum", symbols, capital, **kwargs)
        
        # Momentum-specific parameters
        self.momentum_lookback = momentum_lookback
        self.regime_lookback = regime_lookback
        self.volume_confirmation = volume_confirmation
        self.breakout_threshold = breakout_threshold
        self.trend_strength_threshold = trend_strength_threshold
        
        # Multi-timeframe analysis
        self.timeframes = ['1m', '5m', '15m', '1h']
        self.momentum_data: Dict[str, Dict[str, pd.Series]] = {}
        
        # Regime detection
        self.regime_data: Dict[str, str] = {}  # 'trending' or 'ranging'
        self.volatility_data: Dict[str, pd.Series] = {}
        
        # Performance tracking
        self.trend_captures = 0
        self.false_breakouts = 0
        self.avg_trend_duration = 0.0
        
        logger.info(f"Initialized Momentum Strategy with {len(symbols)} symbols")
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate momentum signals based on:
        1. Multi-timeframe momentum alignment
        2. Regime detection (trending vs. ranging)
        3. Volume confirmation
        4. Breakout detection
        5. Trend strength analysis
        """
        if len(data) < self.regime_lookback:
            return None
        
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        
        # 1. Detect market regime
        regime = self.detect_regime(symbol, data)
        if regime == 'ranging':
            return None  # Don't trade in ranging markets
        
        # 2. Calculate multi-timeframe momentum
        momentum_scores = self.calculate_multi_timeframe_momentum(symbol, data)
        
        # 3. Check for breakout
        breakout_signal = self.detect_breakout(symbol, data)
        
        # 4. Calculate trend strength
        trend_strength = self.calculate_trend_strength(symbol, data)
        
        # 5. Volume confirmation
        volume_confirmed = self.check_volume_confirmation(symbol, data) if self.volume_confirmation else True
        
        # Generate signal based on combined factors
        signal_strength = self.calculate_signal_strength(
            momentum_scores, breakout_signal, trend_strength, volume_confirmed
        )
        
        if signal_strength > 0.6:  # Moderate signal threshold for momentum
            signal_type = self.determine_signal_direction(
                momentum_scores, breakout_signal, trend_strength
            )
            
            confidence = min(signal_strength, 0.9)
            
            return Signal(
                timestamp=current_time,
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                price=current_price,
                confidence=confidence,
                metadata={
                    'regime': regime,
                    'momentum_scores': momentum_scores,
                    'breakout_signal': breakout_signal,
                    'trend_strength': trend_strength,
                    'volume_confirmed': volume_confirmed,
                    'strategy': 'momentum'
                }
            )
        
        return None
    
    def detect_regime(self, symbol: str, data: pd.DataFrame) -> str:
        """
        Detect market regime: trending or ranging
        """
        if len(data) < self.regime_lookback:
            return 'ranging'
        
        # Calculate price volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=self.regime_lookback).std()
        
        # Calculate trend strength using linear regression
        x = np.arange(len(data))
        y = data['close'].values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope by price level
        normalized_slope = slope / data['close'].mean()
        
        # Calculate R-squared (trend strength)
        y_pred = slope * x + np.polyfit(x, y, 1)[1]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine regime
        if r_squared > 0.3 and abs(normalized_slope) > 0.001:
            regime = 'trending'
        else:
            regime = 'ranging'
        
        self.regime_data[symbol] = regime
        return regime
    
    def calculate_multi_timeframe_momentum(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate momentum across multiple timeframes
        """
        momentum_scores = {}
        
        # Calculate momentum for different lookback periods
        lookbacks = [10, 20, 50, 100]
        
        for lookback in lookbacks:
            if len(data) >= lookback:
                # Rate of change
                roc = (data['close'].iloc[-1] - data['close'].iloc[-lookback]) / data['close'].iloc[-lookback]
                
                # RSI-like momentum
                gains = data['close'].diff().where(data['close'].diff() > 0, 0)
                losses = -data['close'].diff().where(data['close'].diff() < 0, 0)
                
                avg_gain = gains.rolling(window=lookback).mean().iloc[-1]
                avg_loss = losses.rolling(window=lookback).mean().iloc[-1]
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
                
                # Combine ROC and RSI
                momentum_score = (roc * 100 + (rsi - 50) / 50) / 2
                momentum_scores[f'{lookback}'] = momentum_score
        
        return momentum_scores
    
    def detect_breakout(self, symbol: str, data: pd.DataFrame) -> float:
        """
        Detect price breakouts from recent range
        """
        if len(data) < 20:
            return 0.0
        
        # Calculate recent high and low
        recent_high = data['high'].rolling(window=20).max().iloc[-1]
        recent_low = data['low'].rolling(window=20).min().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Calculate breakout percentage
        range_size = recent_high - recent_low
        if range_size == 0:
            return 0.0
        
        # Check if price broke above high or below low
        if current_price > recent_high:
            breakout_pct = (current_price - recent_high) / range_size
            return min(breakout_pct, 1.0)
        elif current_price < recent_low:
            breakout_pct = (recent_low - current_price) / range_size
            return -min(breakout_pct, 1.0)
        else:
            return 0.0
    
    def calculate_trend_strength(self, symbol: str, data: pd.DataFrame) -> float:
        """
        Calculate trend strength using multiple indicators
        """
        if len(data) < 50:
            return 0.0
        
        # ADX (Average Directional Index) calculation
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high - high.shift()
        dm_minus = low.shift() - low
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smoothed values
        tr_smooth = tr.rolling(window=14).mean()
        dm_plus_smooth = dm_plus.rolling(window=14).mean()
        dm_minus_smooth = dm_minus.rolling(window=14).mean()
        
        # DI values
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=14).mean()
        
        # Normalize ADX to 0-1 scale
        trend_strength = min(adx.iloc[-1] / 100, 1.0)
        
        return trend_strength
    
    def check_volume_confirmation(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Check if volume confirms the price move
        """
        if len(data) < 20:
            return False
        
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        
        # Volume should be above average for confirmation
        return current_volume > avg_volume * 1.2
    
    def calculate_signal_strength(self, 
                                momentum_scores: Dict[str, float], 
                                breakout_signal: float, 
                                trend_strength: float, 
                                volume_confirmed: bool) -> float:
        """
        Calculate overall signal strength (0.0 to 1.0)
        """
        # Weighted combination of factors
        weights = {
            'momentum': 0.35,    # Multi-timeframe momentum
            'breakout': 0.25,    # Breakout detection
            'trend_strength': 0.25, # Trend strength
            'volume': 0.15       # Volume confirmation
        }
        
        # Average momentum across timeframes
        if momentum_scores:
            avg_momentum = np.mean(list(momentum_scores.values()))
            momentum_score = abs(avg_momentum) / 2  # Normalize to 0-1
        else:
            momentum_score = 0.0
        
        # Breakout score
        breakout_score = abs(breakout_signal)
        
        # Volume score
        volume_score = 1.0 if volume_confirmed else 0.0
        
        # Calculate weighted score
        signal_strength = (
            weights['momentum'] * momentum_score +
            weights['breakout'] * breakout_score +
            weights['trend_strength'] * trend_strength +
            weights['volume'] * volume_score
        )
        
        return signal_strength
    
    def determine_signal_direction(self, 
                                 momentum_scores: Dict[str, float], 
                                 breakout_signal: float, 
                                 trend_strength: float) -> SignalType:
        """
        Determine signal direction based on momentum and breakout
        """
        # Strong breakout signal
        if abs(breakout_signal) > self.breakout_threshold:
            if breakout_signal > 0:
                return SignalType.BUY
            else:
                return SignalType.SELL
        
        # Momentum-based signal
        if momentum_scores:
            avg_momentum = np.mean(list(momentum_scores.values()))
            if abs(avg_momentum) > 0.3:  # Strong momentum
                if avg_momentum > 0:
                    return SignalType.BUY
                else:
                    return SignalType.SELL
        
        # Trend strength signal
        if trend_strength > self.trend_strength_threshold:
            # Use short-term momentum for direction
            if momentum_scores and '10' in momentum_scores:
                if momentum_scores['10'] > 0:
                    return SignalType.BUY
                else:
                    return SignalType.SELL
        
        return SignalType.HOLD
    
    def update_data(self, symbol: str, new_data: pd.DataFrame):
        """
        Override to update momentum-specific data
        """
        super().update_data(symbol, new_data)
        
        # Update volatility data
        returns = new_data['close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std()
        self.volatility_data[symbol] = volatility
    
    def process_signal(self, signal: Signal) -> bool:
        """
        Override to add momentum-specific logic
        """
        # Check regime
        regime = signal.metadata.get('regime', 'unknown')
        if regime == 'ranging':
            return False
        
        # Check trend strength
        trend_strength = signal.metadata.get('trend_strength', 0)
        if trend_strength < self.trend_strength_threshold:
            return False
        
        # Process signal using base logic
        executed = super().process_signal(signal)
        
        if executed:
            # Track momentum metrics
            if signal.metadata.get('breakout_signal', 0) > 0:
                self.trend_captures += 1
            logger.info(f"Momentum signal executed: {signal.symbol} @ {signal.price}, regime: {regime}")
        
        return executed
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Override to add momentum-specific position management
        """
        current_time = datetime.now()
        
        for symbol, position in list(self.positions.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.current_price = current_price
                
                # Calculate holding time
                holding_time = (current_time - position.entry_time).total_seconds()
                
                # Calculate current PnL
                if position.side == "LONG":
                    position.pnl = (current_price - position.entry_price) * position.size
                else:
                    position.pnl = (position.entry_price - current_price) * position.size
                
                # Check stop loss (wider for momentum)
                if ((position.side == "LONG" and current_price <= position.stop_loss) or
                    (position.side == "SHORT" and current_price >= position.stop_loss)):
                    
                    self.close_position(symbol, current_price, current_time)
                    self.false_breakouts += 1
                    continue
                
                # Check take profit (momentum targets larger moves)
                elif ((position.side == "LONG" and current_price >= position.take_profit) or
                      (position.side == "SHORT" and current_price <= position.take_profit)):
                    
                    self.close_position(symbol, current_price, current_time)
                    self.trend_captures += 1
                    
                    # Update average trend duration
                    if self.avg_trend_duration == 0:
                        self.avg_trend_duration = holding_time
                    else:
                        self.avg_trend_duration = (self.avg_trend_duration + holding_time) / 2
                    continue
    
    def get_momentum_metrics(self) -> Dict:
        """
        Get momentum-specific performance metrics
        """
        base_metrics = self.get_performance_metrics()
        
        return {
            **base_metrics,
            'trend_captures': self.trend_captures,
            'false_breakouts': self.false_breakouts,
            'breakout_success_rate': self.trend_captures / (self.trend_captures + self.false_breakouts) if (self.trend_captures + self.false_breakouts) > 0 else 0.0,
            'avg_trend_duration_hours': self.avg_trend_duration / 3600 if self.avg_trend_duration > 0 else 0.0
        }
    
    def log_momentum_status(self):
        """
        Log momentum-specific status
        """
        metrics = self.get_momentum_metrics()
        logger.info(f"Momentum Strategy Status:")
        logger.info(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        logger.info(f"  Daily PnL: ${metrics['daily_pnl']:.2f}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"  Trend Captures: {metrics['trend_captures']}")
        logger.info(f"  False Breakouts: {metrics['false_breakouts']}")
        logger.info(f"  Breakout Success Rate: {metrics['breakout_success_rate']:.2%}")
        logger.info(f"  Avg Trend Duration: {metrics['avg_trend_duration_hours']:.1f} hours")
        logger.info(f"  Open Positions: {len(self.positions)}") 