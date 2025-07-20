"""
High-Frequency Scalping Strategy
Exploits market microstructure for quick profits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

class ScalpingStrategy(BaseStrategy):
    """
    High-frequency scalping strategy that exploits:
    - Order book imbalances
    - Spread capture
    - Microsecond-level timing
    - Liquidity detection
    """
    
    def __init__(self, 
                 symbols: List[str],
                 capital: float = 100000,
                 min_spread_threshold: float = 0.0001,  # Minimum spread to trade
                 max_holding_time: int = 300,  # Max holding time in seconds
                 order_book_depth: int = 5,    # Levels to analyze
                 imbalance_threshold: float = 0.6,  # Order book imbalance threshold
                 volume_threshold: float = 1000,    # Minimum volume for signal
                 **kwargs):
        
        super().__init__("Scalping", symbols, capital, **kwargs)
        
        # Scalping-specific parameters
        self.min_spread_threshold = min_spread_threshold
        self.max_holding_time = max_holding_time
        self.order_book_depth = order_book_depth
        self.imbalance_threshold = imbalance_threshold
        self.volume_threshold = volume_threshold
        
        # Order book state
        self.order_book_data: Dict[str, Dict] = {}
        self.last_trade_prices: Dict[str, float] = {}
        self.volume_profile: Dict[str, pd.Series] = {}
        
        # Performance tracking
        self.spread_captured = 0.0
        self.avg_holding_time = 0.0
        self.fast_trades = 0  # Trades under 30 seconds
        
        logger.info(f"Initialized Scalping Strategy with {len(symbols)} symbols")
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate scalping signals based on:
        1. Order book imbalance
        2. Spread analysis
        3. Volume surge detection
        4. Price momentum
        5. Liquidity analysis
        """
        if len(data) < 50:  # Need minimum data
            return None
        
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        
        # 1. Calculate spread
        spread = self.calculate_spread(symbol, current_price)
        if spread < self.min_spread_threshold:
            return None
        
        # 2. Analyze order book imbalance
        imbalance = self.calculate_order_book_imbalance(symbol)
        
        # 3. Detect volume surge
        volume_surge = self.detect_volume_surge(symbol, data)
        
        # 4. Calculate momentum
        momentum = self.calculate_momentum(data)
        
        # 5. Check liquidity
        liquidity_score = self.assess_liquidity(symbol)
        
        # Generate signal based on combined factors
        signal_strength = self.calculate_signal_strength(
            spread, imbalance, volume_surge, momentum, liquidity_score
        )
        
        if signal_strength > 0.7:  # Strong signal threshold
            signal_type = self.determine_signal_direction(
                imbalance, momentum, volume_surge
            )
            
            confidence = min(signal_strength, 0.95)  # Cap confidence
            
            return Signal(
                timestamp=current_time,
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                price=current_price,
                confidence=confidence,
                metadata={
                    'spread': spread,
                    'imbalance': imbalance,
                    'volume_surge': volume_surge,
                    'momentum': momentum,
                    'liquidity_score': liquidity_score,
                    'strategy': 'scalping'
                }
            )
        
        return None
    
    def calculate_spread(self, symbol: str, current_price: float) -> float:
        """
        Calculate current bid-ask spread
        """
        if symbol not in self.order_book_data:
            return 0.0
        
        order_book = self.order_book_data[symbol]
        
        if 'bids' in order_book and 'asks' in order_book:
            best_bid = max(order_book['bids'].keys())
            best_ask = min(order_book['asks'].keys())
            return best_ask - best_bid
        
        return 0.0
    
    def calculate_order_book_imbalance(self, symbol: str) -> float:
        """
        Calculate order book imbalance (0.0 to 1.0)
        Positive = more bids (bullish)
        Negative = more asks (bearish)
        """
        if symbol not in self.order_book_data:
            return 0.0
        
        order_book = self.order_book_data[symbol]
        
        if 'bids' not in order_book or 'asks' not in order_book:
            return 0.0
        
        # Sum volume at top levels
        bid_volume = sum(list(order_book['bids'].values())[:self.order_book_depth])
        ask_volume = sum(list(order_book['asks'].values())[:self.order_book_depth])
        
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0.0
        
        # Calculate imbalance (-1 to 1)
        imbalance = (bid_volume - ask_volume) / total_volume
        
        return imbalance
    
    def detect_volume_surge(self, symbol: str, data: pd.DataFrame) -> float:
        """
        Detect volume surge compared to recent average
        """
        if len(data) < 20:
            return 0.0
        
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        
        if avg_volume == 0:
            return 0.0
        
        surge_ratio = current_volume / avg_volume
        
        # Normalize to 0-1 scale
        return min(surge_ratio / 3.0, 1.0)  # Cap at 3x average
    
    def calculate_momentum(self, data: pd.DataFrame) -> float:
        """
        Calculate short-term price momentum
        """
        if len(data) < 10:
            return 0.0
        
        # Calculate rate of change over last 10 periods
        roc = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]
        
        # Normalize to -1 to 1 scale
        return np.tanh(roc * 100)  # Scale and bound
    
    def assess_liquidity(self, symbol: str) -> float:
        """
        Assess current market liquidity
        """
        if symbol not in self.order_book_data:
            return 0.5  # Default medium liquidity
        
        order_book = self.order_book_data[symbol]
        
        if 'bids' not in order_book or 'asks' not in order_book:
            return 0.5
        
        # Calculate total volume at top levels
        bid_volume = sum(list(order_book['bids'].values())[:3])
        ask_volume = sum(list(order_book['asks'].values())[:3])
        
        total_volume = bid_volume + ask_volume
        
        # Normalize based on typical volume for this symbol
        typical_volume = self.volume_threshold
        liquidity_score = min(total_volume / typical_volume, 1.0)
        
        return liquidity_score
    
    def calculate_signal_strength(self, 
                                spread: float, 
                                imbalance: float, 
                                volume_surge: float, 
                                momentum: float, 
                                liquidity_score: float) -> float:
        """
        Calculate overall signal strength (0.0 to 1.0)
        """
        # Weighted combination of factors
        weights = {
            'spread': 0.25,      # Spread is important for scalping
            'imbalance': 0.30,   # Order book imbalance is key
            'volume_surge': 0.20, # Volume confirms moves
            'momentum': 0.15,    # Price momentum
            'liquidity': 0.10    # Liquidity for execution
        }
        
        # Normalize spread (higher spread = better opportunity)
        spread_score = min(spread / 0.001, 1.0)  # Normalize to 1% spread
        
        # Calculate weighted score
        signal_strength = (
            weights['spread'] * spread_score +
            weights['imbalance'] * abs(imbalance) +
            weights['volume_surge'] * volume_surge +
            weights['momentum'] * abs(momentum) +
            weights['liquidity'] * liquidity_score
        )
        
        return signal_strength
    
    def determine_signal_direction(self, 
                                 imbalance: float, 
                                 momentum: float, 
                                 volume_surge: float) -> SignalType:
        """
        Determine signal direction based on imbalance and momentum
        """
        # Strong imbalance in one direction
        if abs(imbalance) > self.imbalance_threshold:
            if imbalance > 0:  # More bids = bullish
                return SignalType.BUY
            else:  # More asks = bearish
                return SignalType.SELL
        
        # Momentum-based signal
        if abs(momentum) > 0.3:  # Strong momentum
            if momentum > 0:
                return SignalType.BUY
            else:
                return SignalType.SELL
        
        # Volume surge with slight imbalance
        if volume_surge > 0.5 and abs(imbalance) > 0.2:
            if imbalance > 0:
                return SignalType.BUY
            else:
                return SignalType.SELL
        
        return SignalType.HOLD
    
    def update_order_book(self, symbol: str, order_book: Dict):
        """
        Update order book data for signal generation
        """
        self.order_book_data[symbol] = order_book
    
    def update_trade_data(self, symbol: str, trade_price: float, trade_volume: float):
        """
        Update trade data for analysis
        """
        self.last_trade_prices[symbol] = trade_price
        
        if symbol not in self.volume_profile:
            self.volume_profile[symbol] = pd.Series(dtype=float)
        
        # Add to volume profile
        self.volume_profile[symbol] = self.volume_profile[symbol].append(
            pd.Series([trade_volume], index=[datetime.now()])
        )
        
        # Keep only recent data
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.volume_profile[symbol] = self.volume_profile[symbol][
            self.volume_profile[symbol].index > cutoff_time
        ]
    
    def process_signal(self, signal: Signal) -> bool:
        """
        Override to add scalping-specific logic
        """
        # Check if we can capture enough spread
        spread = signal.metadata.get('spread', 0)
        if spread < self.min_spread_threshold:
            return False
        
        # Check liquidity for quick execution
        liquidity_score = signal.metadata.get('liquidity_score', 0)
        if liquidity_score < 0.3:  # Need decent liquidity
            return False
        
        # Process signal using base logic
        executed = super().process_signal(signal)
        
        if executed:
            # Track scalping metrics
            self.spread_captured += spread
            logger.info(f"Scalping signal executed: {signal.symbol} @ {signal.price}, spread: {spread:.6f}")
        
        return executed
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Override to add scalping-specific position management
        """
        current_time = datetime.now()
        
        for symbol, position in list(self.positions.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.current_price = current_price
                
                # Calculate holding time
                holding_time = (current_time - position.entry_time).total_seconds()
                
                # Check max holding time (scalping specific)
                if holding_time > self.max_holding_time:
                    self.close_position(symbol, current_price, current_time)
                    logger.info(f"Scalping position closed due to max holding time: {symbol}")
                    continue
                
                # Calculate current PnL
                if position.side == "LONG":
                    position.pnl = (current_price - position.entry_price) * position.size
                else:
                    position.pnl = (position.entry_price - current_price) * position.size
                
                # Check stop loss (tighter for scalping)
                if ((position.side == "LONG" and current_price <= position.stop_loss) or
                    (position.side == "SHORT" and current_price >= position.stop_loss)):
                    
                    self.close_position(symbol, current_price, current_time)
                    continue
                
                # Check take profit (scalping targets smaller profits)
                elif ((position.side == "LONG" and current_price >= position.take_profit) or
                      (position.side == "SHORT" and current_price <= position.take_profit)):
                    
                    self.close_position(symbol, current_price, current_time)
                    
                    # Track fast trades
                    if holding_time < 30:
                        self.fast_trades += 1
                    continue
    
    def get_scalping_metrics(self) -> Dict:
        """
        Get scalping-specific performance metrics
        """
        base_metrics = self.get_performance_metrics()
        
        # Calculate average holding time
        if self.trade_history:
            holding_times = [(trade['exit_time'] - trade['entry_time']).total_seconds() 
                           for trade in self.trade_history]
            avg_holding_time = np.mean(holding_times)
        else:
            avg_holding_time = 0.0
        
        # Calculate spread capture efficiency
        total_trades = len(self.trade_history)
        avg_spread_captured = self.spread_captured / total_trades if total_trades > 0 else 0.0
        
        return {
            **base_metrics,
            'avg_holding_time_seconds': avg_holding_time,
            'total_spread_captured': self.spread_captured,
            'avg_spread_captured': avg_spread_captured,
            'fast_trades_under_30s': self.fast_trades,
            'fast_trade_ratio': self.fast_trades / total_trades if total_trades > 0 else 0.0
        }
    
    def log_scalping_status(self):
        """
        Log scalping-specific status
        """
        metrics = self.get_scalping_metrics()
        logger.info(f"Scalping Strategy Status:")
        logger.info(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        logger.info(f"  Daily PnL: ${metrics['daily_pnl']:.2f}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"  Avg Holding Time: {metrics['avg_holding_time_seconds']:.1f}s")
        logger.info(f"  Total Spread Captured: {metrics['total_spread_captured']:.6f}")
        logger.info(f"  Fast Trades (<30s): {metrics['fast_trades_under_30s']}")
        logger.info(f"  Open Positions: {len(self.positions)}") 