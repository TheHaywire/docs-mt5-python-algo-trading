"""
Base Strategy Class for MT5 Python Algo Trading System
Provides foundation for all profit-generating strategies
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    price: float
    confidence: float  # 0.0 to 1.0
    metadata: Dict

@dataclass
class Position:
    symbol: str
    side: str  # "LONG" or "SHORT"
    size: float
    entry_price: float
    entry_time: datetime
    current_price: float
    pnl: float
    stop_loss: float
    take_profit: float

class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    Implements core profit-generating logic
    """
    
    def __init__(self, 
                 name: str,
                 symbols: List[str],
                 capital: float = 100000,
                 max_position_size: float = 0.02,  # 2% of capital
                 max_daily_loss: float = 0.05,     # 5% daily loss limit
                 risk_per_trade: float = 0.01,     # 1% risk per trade
                 volatility_lookback: int = 20):
        
        self.name = name
        self.symbols = symbols
        self.capital = capital
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.risk_per_trade = risk_per_trade
        self.volatility_lookback = volatility_lookback
        
        # State management
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.total_pnl = 0.0
        
        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.volatility_data: Dict[str, pd.Series] = {}
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.signals_generated: List[Signal] = []
        
        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate trading signal based on strategy logic
        Must be implemented by each strategy
        """
        pass
    
    def calculate_position_size(self, signal: Signal, current_price: float) -> float:
        """
        Kelly Criterion + Volatility-adjusted position sizing
        """
        # Kelly Criterion
        win_rate = self.get_win_rate()
        avg_win = self.get_average_win()
        avg_loss = self.get_average_loss()
        
        if avg_loss == 0:
            kelly_fraction = 0.1  # Conservative default
        else:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Volatility adjustment
        volatility = self.get_volatility(signal.symbol)
        volatility_factor = 1.0 / (1.0 + volatility)
        
        # Risk-based sizing
        risk_amount = self.capital * self.risk_per_trade
        stop_distance = abs(signal.price - self.get_stop_loss_price(signal))
        
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = self.capital * self.max_position_size * 0.1
        
        # Apply Kelly and volatility adjustments
        final_size = position_size * kelly_fraction * volatility_factor
        
        # Cap at maximum position size
        max_size = self.capital * self.max_position_size
        return min(final_size, max_size)
    
    def get_stop_loss_price(self, signal: Signal) -> float:
        """
        Calculate stop loss price based on volatility
        """
        volatility = self.get_volatility(signal.symbol)
        atr_multiplier = 2.0  # 2x ATR for stop loss
        
        if signal.signal_type == SignalType.BUY:
            return signal.price - (volatility * atr_multiplier)
        else:
            return signal.price + (volatility * atr_multiplier)
    
    def get_take_profit_price(self, signal: Signal) -> float:
        """
        Calculate take profit price (2:1 risk-reward ratio)
        """
        stop_loss = self.get_stop_loss_price(signal)
        risk = abs(signal.price - stop_loss)
        
        if signal.signal_type == SignalType.BUY:
            return signal.price + (risk * 2)
        else:
            return signal.price - (risk * 2)
    
    def get_volatility(self, symbol: str) -> float:
        """
        Calculate current volatility using ATR
        """
        if symbol not in self.volatility_data:
            return 0.001  # Default low volatility
        
        return self.volatility_data[symbol].iloc[-1] if len(self.volatility_data[symbol]) > 0 else 0.001
    
    def get_win_rate(self) -> float:
        """
        Calculate win rate from trade history
        """
        if not self.trade_history:
            return 0.5  # Default 50% win rate
        
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        return winning_trades / len(self.trade_history)
    
    def get_average_win(self) -> float:
        """
        Calculate average winning trade
        """
        winning_trades = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
        return np.mean(winning_trades) if winning_trades else 0.01
    
    def get_average_loss(self) -> float:
        """
        Calculate average losing trade
        """
        losing_trades = [abs(trade['pnl']) for trade in self.trade_history if trade['pnl'] < 0]
        return np.mean(losing_trades) if losing_trades else 0.01
    
    def update_data(self, symbol: str, new_data: pd.DataFrame):
        """
        Update price and volatility data
        """
        if symbol not in self.price_data:
            self.price_data[symbol] = new_data
        else:
            self.price_data[symbol] = pd.concat([self.price_data[symbol], new_data])
        
        # Calculate volatility (ATR)
        high = new_data['high']
        low = new_data['low']
        close = new_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.volatility_lookback).mean()
        
        self.volatility_data[symbol] = atr
    
    def process_signal(self, signal: Signal) -> bool:
        """
        Process trading signal and execute if conditions are met
        """
        # Check daily loss limit
        if self.daily_pnl < -(self.capital * self.max_daily_loss):
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
            return False
        
        # Check if we already have a position in this symbol
        if signal.symbol in self.positions:
            current_position = self.positions[signal.symbol]
            
            # Close position if signal is opposite
            if ((signal.signal_type == SignalType.SELL and current_position.side == "LONG") or
                (signal.signal_type == SignalType.BUY and current_position.side == "SHORT")):
                
                self.close_position(signal.symbol, signal.price, signal.timestamp)
                return True
        
        # Open new position
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            position_size = self.calculate_position_size(signal, signal.price)
            
            if position_size > 0:
                self.open_position(signal, position_size)
                return True
        
        return False
    
    def open_position(self, signal: Signal, size: float):
        """
        Open a new trading position
        """
        side = "LONG" if signal.signal_type == SignalType.BUY else "SHORT"
        stop_loss = self.get_stop_loss_price(signal)
        take_profit = self.get_take_profit_price(signal)
        
        position = Position(
            symbol=signal.symbol,
            side=side,
            size=size,
            entry_price=signal.price,
            entry_time=signal.timestamp,
            current_price=signal.price,
            pnl=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[signal.symbol] = position
        logger.info(f"Opened {side} position: {signal.symbol} @ {signal.price}, size: {size}")
    
    def close_position(self, symbol: str, price: float, timestamp: datetime):
        """
        Close an existing position
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate PnL
        if position.side == "LONG":
            pnl = (price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - price) * position.size
        
        # Update tracking
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.daily_trades += 1
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': price,
            'size': position.size,
            'pnl': pnl,
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'duration': (timestamp - position.entry_time).total_seconds() / 3600  # hours
        }
        
        self.trade_history.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} @ {price}, PnL: {pnl:.2f}")
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update position PnL and check stop loss/take profit
        """
        for symbol, position in list(self.positions.items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.current_price = current_price
                
                # Calculate current PnL
                if position.side == "LONG":
                    position.pnl = (current_price - position.entry_price) * position.size
                else:
                    position.pnl = (position.entry_price - current_price) * position.size
                
                # Check stop loss
                if ((position.side == "LONG" and current_price <= position.stop_loss) or
                    (position.side == "SHORT" and current_price >= position.stop_loss)):
                    
                    self.close_position(symbol, current_price, datetime.now())
                
                # Check take profit
                elif ((position.side == "LONG" and current_price >= position.take_profit) or
                      (position.side == "SHORT" and current_price <= position.take_profit)):
                    
                    self.close_position(symbol, current_price, datetime.now())
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.trade_history:
            return {
                'total_pnl': 0.0,
                'daily_pnl': self.daily_pnl,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': 0
            }
        
        pnls = [trade['pnl'] for trade in self.trade_history]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [abs(p) for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        win_rate = len(winning_trades) / len(pnls) if pnls else 0
        
        profit_factor = (sum(winning_trades) / sum(losing_trades) 
                        if losing_trades and sum(losing_trades) > 0 else float('inf'))
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(cumulative_pnl)
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252) 
                       if len(returns) > 1 and np.std(returns) > 0 else 0)
        
        return {
            'total_pnl': total_pnl,
            'daily_pnl': self.daily_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trade_history),
            'avg_trade_duration': np.mean([t['duration'] for t in self.trade_history])
        }
    
    def reset_daily_metrics(self):
        """
        Reset daily metrics (call at start of new trading day)
        """
        self.daily_pnl = 0.0
        self.daily_trades = 0
        logger.info("Daily metrics reset")
    
    def log_status(self):
        """
        Log current strategy status
        """
        metrics = self.get_performance_metrics()
        logger.info(f"Strategy {self.name} Status:")
        logger.info(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        logger.info(f"  Daily PnL: ${metrics['daily_pnl']:.2f}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"  Open Positions: {len(self.positions)}")
        logger.info(f"  Total Trades: {metrics['total_trades']}") 