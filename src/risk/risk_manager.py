"""
Risk Management System for MT5 Python Algo Trading System
Real-time risk monitoring, VaR calculation, and kill switches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    portfolio_beta: float
    correlation_matrix: pd.DataFrame
    concentration_risk: float
    leverage_ratio: float
    margin_usage: float

class RiskManager:
    """
    Comprehensive risk management system with:
    - Real-time VaR calculation
    - Portfolio risk aggregation
    - Kill switch logic
    - Stress testing
    - Position limits
    """
    
    def __init__(self,
                 capital: float = 100000,
                 max_var_95: float = 0.02,      # 2% VaR limit
                 max_drawdown: float = 0.15,    # 15% max drawdown
                 max_leverage: float = 2.0,     # 2x leverage limit
                 max_concentration: float = 0.1, # 10% max concentration
                 correlation_threshold: float = 0.7, # High correlation threshold
                 lookback_period: int = 252):   # 1 year for calculations
        
        self.capital = capital
        self.max_var_95 = max_var_95
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.max_concentration = max_concentration
        self.correlation_threshold = correlation_threshold
        self.lookback_period = lookback_period
        
        # Risk state
        self.positions: Dict[str, Dict] = {}
        self.portfolio_value = capital
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
        # Historical data for calculations
        self.price_history: Dict[str, pd.Series] = {}
        self.returns_history: Dict[str, pd.Series] = {}
        self.portfolio_returns: List[float] = []
        
        # Risk metrics
        self.current_risk_metrics: Optional[RiskMetrics] = None
        self.risk_level = RiskLevel.LOW
        
        # Kill switch state
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.kill_switch_time = None
        
        # Alerts
        self.risk_alerts: List[Dict] = []
        
        logger.info(f"Initialized Risk Manager with capital: ${capital:,.2f}")
    
    def update_position(self, symbol: str, position_data: Dict):
        """
        Update position data for risk calculations
        """
        self.positions[symbol] = position_data
        
        # Update portfolio value
        self.update_portfolio_value()
    
    def update_portfolio_value(self):
        """
        Calculate current portfolio value
        """
        total_value = self.capital
        
        for symbol, position in self.positions.items():
            if position['side'] == 'LONG':
                total_value += position['pnl']
            else:
                total_value += position['pnl']
        
        self.portfolio_value = total_value
    
    def update_price_data(self, symbol: str, price_data: pd.DataFrame):
        """
        Update price data for risk calculations
        """
        if len(price_data) > 0:
            self.price_history[symbol] = price_data['close']
            
            # Calculate returns
            returns = price_data['close'].pct_change().dropna()
            self.returns_history[symbol] = returns
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk using historical simulation
        """
        if not self.portfolio_returns:
            return 0.0
        
        returns = np.array(self.portfolio_returns)
        
        # Historical VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        # Convert to dollar amount
        var_dollars = abs(var) * self.portfolio_value
        
        return var_dollars
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        """
        if not self.portfolio_returns:
            return 0.0
        
        returns = np.array(self.portfolio_returns)
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)
        
        # Calculate expected shortfall
        tail_returns = returns[returns <= var_threshold]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else 0.0
        
        return abs(expected_shortfall) * self.portfolio_value
    
    def calculate_portfolio_beta(self) -> float:
        """
        Calculate portfolio beta relative to market
        """
        if not self.returns_history:
            return 1.0
        
        # Use first symbol as market proxy (simplified)
        market_symbol = list(self.returns_history.keys())[0]
        market_returns = self.returns_history[market_symbol]
        
        if len(self.portfolio_returns) < 2 or len(market_returns) < 2:
            return 1.0
        
        # Calculate beta
        portfolio_returns = np.array(self.portfolio_returns[-len(market_returns):])
        market_returns_array = market_returns.values
        
        if len(portfolio_returns) != len(market_returns_array):
            min_len = min(len(portfolio_returns), len(market_returns_array))
            portfolio_returns = portfolio_returns[-min_len:]
            market_returns_array = market_returns_array[-min_len:]
        
        if len(portfolio_returns) < 2:
            return 1.0
        
        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, market_returns_array)[0, 1]
        market_variance = np.var(market_returns_array)
        
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        return beta
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for all positions
        """
        if len(self.returns_history) < 2:
            return pd.DataFrame()
        
        # Create returns dataframe
        returns_df = pd.DataFrame(self.returns_history)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def calculate_concentration_risk(self) -> float:
        """
        Calculate concentration risk (largest position as % of portfolio)
        """
        if not self.positions:
            return 0.0
        
        position_values = []
        for symbol, position in self.positions.items():
            position_value = abs(position['size'] * position['current_price'])
            position_values.append(position_value)
        
        if not position_values:
            return 0.0
        
        max_position_value = max(position_values)
        concentration_risk = max_position_value / self.portfolio_value
        
        return concentration_risk
    
    def calculate_leverage_ratio(self) -> float:
        """
        Calculate current leverage ratio
        """
        if not self.positions:
            return 1.0
        
        total_position_value = 0.0
        for symbol, position in self.positions.items():
            position_value = abs(position['size'] * position['current_price'])
            total_position_value += position_value
        
        leverage_ratio = total_position_value / self.portfolio_value
        
        return leverage_ratio
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown
        """
        if not self.portfolio_returns:
            return 0.0
        
        cumulative_returns = np.cumsum(self.portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        return max_drawdown
    
    def update_risk_metrics(self):
        """
        Update all risk metrics
        """
        # Calculate VaR
        var_95 = self.calculate_var(0.95)
        var_99 = self.calculate_var(0.99)
        
        # Calculate Expected Shortfall
        expected_shortfall = self.calculate_expected_shortfall(0.95)
        
        # Calculate other metrics
        max_drawdown = self.calculate_max_drawdown()
        portfolio_beta = self.calculate_portfolio_beta()
        correlation_matrix = self.calculate_correlation_matrix()
        concentration_risk = self.calculate_concentration_risk()
        leverage_ratio = self.calculate_leverage_ratio()
        
        # Calculate margin usage (simplified)
        margin_usage = leverage_ratio / self.max_leverage
        
        # Create risk metrics object
        self.current_risk_metrics = RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            portfolio_beta=portfolio_beta,
            correlation_matrix=correlation_matrix,
            concentration_risk=concentration_risk,
            leverage_ratio=leverage_ratio,
            margin_usage=margin_usage
        )
        
        # Update risk level
        self.update_risk_level()
    
    def update_risk_level(self):
        """
        Update risk level based on current metrics
        """
        if not self.current_risk_metrics:
            self.risk_level = RiskLevel.LOW
            return
        
        metrics = self.current_risk_metrics
        
        # Check for critical risk
        if (metrics.var_95 > self.capital * self.max_var_95 * 1.5 or
            metrics.max_drawdown > self.max_drawdown * 1.2 or
            metrics.leverage_ratio > self.max_leverage * 1.1):
            self.risk_level = RiskLevel.CRITICAL
        
        # Check for high risk
        elif (metrics.var_95 > self.capital * self.max_var_95 or
              metrics.max_drawdown > self.max_drawdown or
              metrics.leverage_ratio > self.max_leverage):
            self.risk_level = RiskLevel.HIGH
        
        # Check for medium risk
        elif (metrics.var_95 > self.capital * self.max_var_95 * 0.7 or
              metrics.max_drawdown > self.max_drawdown * 0.7 or
              metrics.leverage_ratio > self.max_leverage * 0.7):
            self.risk_level = RiskLevel.MEDIUM
        
        else:
            self.risk_level = RiskLevel.LOW
    
    def check_kill_switch(self) -> bool:
        """
        Check if kill switch should be activated
        """
        if self.kill_switch_active:
            return True
        
        if not self.current_risk_metrics:
            return False
        
        metrics = self.current_risk_metrics
        
        # Critical conditions for kill switch
        kill_conditions = [
            (metrics.var_95 > self.capital * self.max_var_95 * 2.0, "VaR exceeded 200% of limit"),
            (metrics.max_drawdown > self.max_drawdown * 1.5, "Drawdown exceeded 150% of limit"),
            (metrics.leverage_ratio > self.max_leverage * 1.5, "Leverage exceeded 150% of limit"),
            (metrics.concentration_risk > self.max_concentration * 2.0, "Concentration risk exceeded 200% of limit"),
            (self.daily_pnl < -self.capital * 0.1, "Daily loss exceeded 10% of capital")
        ]
        
        for condition, reason in kill_conditions:
            if condition:
                self.activate_kill_switch(reason)
                return True
        
        return False
    
    def activate_kill_switch(self, reason: str):
        """
        Activate kill switch
        """
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_time = datetime.now()
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        
        # Create alert
        alert = {
            'timestamp': self.kill_switch_time,
            'level': 'CRITICAL',
            'message': f'Kill switch activated: {reason}',
            'action_required': 'Immediate position closure required'
        }
        self.risk_alerts.append(alert)
    
    def deactivate_kill_switch(self):
        """
        Deactivate kill switch
        """
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.kill_switch_time = None
        
        logger.info("Kill switch deactivated")
    
    def check_position_limits(self, symbol: str, size: float, price: float) -> bool:
        """
        Check if new position meets risk limits
        """
        if self.kill_switch_active:
            return False
        
        # Check concentration limit
        position_value = size * price
        new_concentration = position_value / self.portfolio_value
        
        if new_concentration > self.max_concentration:
            logger.warning(f"Position rejected: concentration limit exceeded for {symbol}")
            return False
        
        # Check leverage limit
        current_leverage = self.calculate_leverage_ratio()
        new_leverage = current_leverage + (position_value / self.portfolio_value)
        
        if new_leverage > self.max_leverage:
            logger.warning(f"Position rejected: leverage limit exceeded for {symbol}")
            return False
        
        return True
    
    def stress_test(self, scenario: str) -> Dict:
        """
        Perform stress testing under different scenarios
        """
        if not self.current_risk_metrics:
            return {}
        
        base_metrics = self.current_risk_metrics
        
        if scenario == "market_crash":
            # Simulate 20% market crash
            stress_factor = 0.8
            stress_pnl = self.daily_pnl * stress_factor
            stress_var = base_metrics.var_95 * 2.0  # VaR doubles in crash
            
        elif scenario == "volatility_spike":
            # Simulate 3x volatility increase
            stress_factor = 3.0
            stress_pnl = self.daily_pnl * stress_factor
            stress_var = base_metrics.var_95 * stress_factor
            
        elif scenario == "correlation_breakdown":
            # Simulate correlation breakdown (diversification fails)
            stress_factor = 1.5
            stress_pnl = self.daily_pnl * stress_factor
            stress_var = base_metrics.var_95 * stress_factor
            
        else:
            return {}
        
        stress_result = {
            'scenario': scenario,
            'stress_pnl': stress_pnl,
            'stress_var': stress_var,
            'capital_impact': stress_pnl / self.capital,
            'var_impact': stress_var / self.capital
        }
        
        return stress_result
    
    def update_portfolio_return(self, pnl: float):
        """
        Update portfolio return for risk calculations
        """
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        # Calculate return as percentage
        if self.portfolio_value > 0:
            return_pct = pnl / self.portfolio_value
            self.portfolio_returns.append(return_pct)
        
        # Keep only recent returns
        if len(self.portfolio_returns) > self.lookback_period:
            self.portfolio_returns = self.portfolio_returns[-self.lookback_period:]
    
    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report
        """
        if not self.current_risk_metrics:
            return {}
        
        metrics = self.current_risk_metrics
        
        report = {
            'timestamp': datetime.now(),
            'risk_level': self.risk_level.value,
            'portfolio_value': self.portfolio_value,
            'capital': self.capital,
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'var_95': metrics.var_95,
            'var_99': metrics.var_99,
            'expected_shortfall': metrics.expected_shortfall,
            'max_drawdown': metrics.max_drawdown,
            'portfolio_beta': metrics.portfolio_beta,
            'concentration_risk': metrics.concentration_risk,
            'leverage_ratio': metrics.leverage_ratio,
            'margin_usage': metrics.margin_usage,
            'kill_switch_active': self.kill_switch_active,
            'kill_switch_reason': self.kill_switch_reason,
            'open_positions': len(self.positions),
            'risk_alerts': len(self.risk_alerts)
        }
        
        return report
    
    def log_risk_status(self):
        """
        Log current risk status
        """
        if not self.current_risk_metrics:
            logger.info("Risk metrics not available")
            return
        
        metrics = self.current_risk_metrics
        report = self.get_risk_report()
        
        logger.info(f"Risk Manager Status:")
        logger.info(f"  Risk Level: {report['risk_level']}")
        logger.info(f"  Portfolio Value: ${report['portfolio_value']:,.2f}")
        logger.info(f"  Daily PnL: ${report['daily_pnl']:,.2f}")
        logger.info(f"  VaR (95%): ${metrics.var_95:,.2f}")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
        logger.info(f"  Leverage Ratio: {metrics.leverage_ratio:.2f}")
        logger.info(f"  Concentration Risk: {metrics.concentration_risk:.2%}")
        logger.info(f"  Kill Switch: {'ACTIVE' if self.kill_switch_active else 'Inactive'}")
        logger.info(f"  Open Positions: {len(self.positions)}") 