"""
Backtesting Engine for MT5 Python Algo Trading System
Supports walk-forward analysis, parameter optimization, and performance attribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class BacktestResult:
    def __init__(self, trades: List[Dict], equity_curve: pd.Series, metrics: Dict):
        self.trades = trades
        self.equity_curve = equity_curve
        self.metrics = metrics

class BacktestEngine:
    """
    Backtesting engine for trading strategies
    """
    def __init__(self, strategy_class: Callable, data: pd.DataFrame, params: Dict, initial_capital: float = 100000):
        self.strategy_class = strategy_class
        self.data = data
        self.params = params
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = None
        self.metrics = {}

    def run(self) -> BacktestResult:
        """
        Run the backtest
        """
        strategy = self.strategy_class(symbols=[self.params.get('symbol', 'TEST')], capital=self.initial_capital, **self.params)
        cash = self.initial_capital
        equity = [cash]
        position = None
        entry_price = 0
        entry_time = None
        trade_log = []
        
        for i in range(1, len(self.data)):
            row = self.data.iloc[:i+1]
            signal = strategy.generate_signal(self.params.get('symbol', 'TEST'), row)
            price = row['close'].iloc[-1]
            timestamp = row['timestamp'].iloc[-1] if 'timestamp' in row else None
            
            # Process signal
            if signal and signal.signal_type in ['BUY', 'SELL'] and not position:
                size = strategy.calculate_position_size(signal, price)
                position = {
                    'side': signal.signal_type,
                    'size': size,
                    'entry_price': price,
                    'entry_time': timestamp
                }
                entry_price = price
                entry_time = timestamp
                trade_log.append({'action': 'OPEN', 'side': signal.signal_type, 'price': price, 'size': size, 'timestamp': timestamp})
            elif position:
                # Check for exit
                if (position['side'] == 'BUY' and price <= strategy.get_stop_loss_price(signal)) or \
                   (position['side'] == 'SELL' and price >= strategy.get_stop_loss_price(signal)) or \
                   (position['side'] == 'BUY' and price >= strategy.get_take_profit_price(signal)) or \
                   (position['side'] == 'SELL' and price <= strategy.get_take_profit_price(signal)):
                    pnl = (price - entry_price) * position['size'] if position['side'] == 'BUY' else (entry_price - price) * position['size']
                    cash += pnl
                    trade_log.append({'action': 'CLOSE', 'side': position['side'], 'price': price, 'size': position['size'], 'pnl': pnl, 'timestamp': timestamp})
                    position = None
            equity.append(cash)
        
        self.trades = trade_log
        self.equity_curve = pd.Series(equity, index=self.data.index[:len(equity)])
        self.metrics = self.calculate_metrics()
        return BacktestResult(self.trades, self.equity_curve, self.metrics)

    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics
        """
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return {}
        returns = self.equity_curve.pct_change().dropna()
        total_return = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        max_drawdown = ((self.equity_curve.cummax() - self.equity_curve) / self.equity_curve.cummax()).max()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        win_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        loss_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        win_rate = len(win_trades) / (len(win_trades) + len(loss_trades)) if (len(win_trades) + len(loss_trades)) > 0 else 0
        profit_factor = sum(t.get('pnl', 0) for t in win_trades) / abs(sum(t.get('pnl', 0) for t in loss_trades)) if loss_trades else float('inf')
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(self.trades)
        }

    def walk_forward(self, window: int = 252, step: int = 21) -> List[BacktestResult]:
        """
        Walk-forward analysis
        """
        results = []
        for start in range(0, len(self.data) - window, step):
            end = start + window
            sub_data = self.data.iloc[start:end]
            if len(sub_data) < window:
                continue
            result = self.run()
            results.append(result)
        return results

    def optimize_parameters(self, param_grid: Dict[str, List[Any]]) -> Dict:
        """
        Parameter optimization (grid search)
        """
        best_params = None
        best_metric = -np.inf
        for param, values in param_grid.items():
            for value in values:
                params = self.params.copy()
                params[param] = value
                self.params = params
                result = self.run()
                metric = result.metrics.get('sharpe_ratio', 0)
                if metric > best_metric:
                    best_metric = metric
                    best_params = params.copy()
        return best_params 