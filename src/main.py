"""
Main Trading System for MT5 Python Algo Trading System
Orchestrates all components for automated trading
"""

import asyncio
import logging
import yaml
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Import our components
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from strategies.scalping_strategy import ScalpingStrategy
from strategies.momentum_strategy import MomentumStrategy
from risk.risk_manager import RiskManager, RiskLevel
from execution.execution_engine import ExecutionEngine, OrderType
from data.data_feed import DataFeed
from monitoring.monitor import TradingMonitor
from utils.logger import setup_logger

# Configure logging
setup_logger()
logger = logging.getLogger(__name__)

class MT5TradingSystem:
    """
    Main trading system that orchestrates all components
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # System state
        self.is_running = False
        self.is_shutdown = False
        
        # Components
        self.data_feed = None
        self.risk_manager = None
        self.execution_engine = None
        self.monitor = None
        
        # Strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        
        # Performance tracking
        self.start_time = None
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.trade_count = 0
        
        # Market data
        self.current_prices: Dict[str, float] = {}
        self.order_books: Dict[str, Dict] = {}
        
        logger.info("Initializing MT5 Trading System")
    
    def load_config(self) -> Dict:
        """
        Load configuration from YAML file
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """
        Get default configuration
        """
        return {
            'mt5': {
                'login': 12345,
                'password': 'password',
                'server': 'MetaQuotes-Demo',
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY']
            },
            'trading': {
                'capital': 100000,
                'max_position_size': 0.02,
                'max_daily_loss': 0.05,
                'risk_per_trade': 0.01
            },
            'strategies': {
                'scalping': {
                    'enabled': True,
                    'symbols': ['EURUSD'],
                    'parameters': {
                        'min_spread_threshold': 0.0001,
                        'max_holding_time': 300,
                        'imbalance_threshold': 0.6
                    }
                },
                'momentum': {
                    'enabled': True,
                    'symbols': ['GBPUSD', 'USDJPY'],
                    'parameters': {
                        'momentum_lookback': 20,
                        'breakout_threshold': 0.02,
                        'trend_strength_threshold': 0.6
                    }
                }
            },
            'risk': {
                'max_var_95': 0.02,
                'max_drawdown': 0.15,
                'max_leverage': 2.0,
                'max_concentration': 0.1
            },
            'execution': {
                'use_smart_routing': True,
                'use_iceberg': True,
                'market_impact_model': True,
                'order_timeout': 30.0
            },
            'monitoring': {
                'enabled': True,
                'alert_threshold': 0.05,
                'dashboard_port': 8080
            }
        }
    
    async def initialize(self):
        """
        Initialize all system components
        """
        logger.info("Initializing trading system components...")
        
        try:
            # Initialize MT5 connection (simulated)
            mt5_connection = self.initialize_mt5()
            
            # Initialize data feed
            self.data_feed = DataFeed(
                symbols=self.config['mt5']['symbols'],
                mt5_connection=mt5_connection
            )
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                capital=self.config['trading']['capital'],
                max_var_95=self.config['risk']['max_var_95'],
                max_drawdown=self.config['risk']['max_drawdown'],
                max_leverage=self.config['risk']['max_leverage'],
                max_concentration=self.config['risk']['max_concentration']
            )
            
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(
                mt5_connection=mt5_connection,
                use_smart_routing=self.config['execution']['use_smart_routing'],
                use_iceberg=self.config['execution']['use_iceberg'],
                market_impact_model=self.config['execution']['market_impact_model'],
                order_timeout=self.config['execution']['order_timeout']
            )
            
            # Initialize strategies
            await self.initialize_strategies()
            
            # Initialize monitor
            if self.config['monitoring']['enabled']:
                self.monitor = TradingMonitor(
                    port=self.config['monitoring']['dashboard_port'],
                    alert_threshold=self.config['monitoring']['alert_threshold']
                )
                await self.monitor.start()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    def initialize_mt5(self):
        """
        Initialize MT5 connection (simulated)
        """
        # In real implementation, this would connect to MT5
        logger.info("Initializing MT5 connection (simulated)")
        return {"connected": True, "account_info": {"balance": 100000}}
    
    async def initialize_strategies(self):
        """
        Initialize trading strategies
        """
        logger.info("Initializing trading strategies...")
        
        # Initialize Scalping Strategy
        if self.config['strategies']['scalping']['enabled']:
            scalping_config = self.config['strategies']['scalping']
            scalping_strategy = ScalpingStrategy(
                symbols=scalping_config['symbols'],
                capital=self.config['trading']['capital'],
                **scalping_config['parameters']
            )
            self.strategies['scalping'] = scalping_strategy
            logger.info("Scalping strategy initialized")
        
        # Initialize Momentum Strategy
        if self.config['strategies']['momentum']['enabled']:
            momentum_config = self.config['strategies']['momentum']
            momentum_strategy = MomentumStrategy(
                symbols=momentum_config['symbols'],
                capital=self.config['trading']['capital'],
                **momentum_config['parameters']
            )
            self.strategies['momentum'] = momentum_strategy
            logger.info("Momentum strategy initialized")
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    async def start(self):
        """
        Start the trading system
        """
        if self.is_running:
            logger.warning("Trading system is already running")
            return
        
        logger.info("Starting MT5 Trading System...")
        
        try:
            # Initialize components
            await self.initialize()
            
            # Set up signal handlers
            self.setup_signal_handlers()
            
            # Start data feed
            await self.data_feed.start()
            
            # Mark as running
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("Trading system started successfully")
            
            # Main trading loop
            await self.trading_loop()
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            await self.shutdown()
    
    async def trading_loop(self):
        """
        Main trading loop
        """
        logger.info("Starting main trading loop...")
        
        while self.is_running and not self.is_shutdown:
            try:
                # Get latest market data
                await self.update_market_data()
                
                # Check risk limits
                if not await self.check_risk_limits():
                    logger.warning("Risk limits exceeded, pausing trading")
                    await asyncio.sleep(5)
                    continue
                
                # Generate and process signals
                await self.process_strategies()
                
                # Update positions
                await self.update_positions()
                
                # Update monitoring
                await self.update_monitoring()
                
                # Log status periodically
                await self.log_status()
                
                # Sleep between iterations
                await asyncio.sleep(1)  # 1 second cycle
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def update_market_data(self):
        """
        Update market data from data feed
        """
        try:
            # Get latest data from data feed
            market_data = await self.data_feed.get_latest_data()
            
            for symbol, data in market_data.items():
                # Update current prices
                if 'close' in data and len(data['close']) > 0:
                    self.current_prices[symbol] = data['close'].iloc[-1]
                
                # Update order books
                if 'order_book' in data:
                    self.order_books[symbol] = data['order_book']
                
                # Update strategies with new data
                for strategy in self.strategies.values():
                    if symbol in strategy.symbols:
                        strategy.update_data(symbol, data)
                
                # Update risk manager
                self.risk_manager.update_price_data(symbol, data)
                
                # Update execution engine
                if 'order_book' in data:
                    self.execution_engine.update_order_book(symbol, data['order_book'])
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def check_risk_limits(self) -> bool:
        """
        Check if risk limits are within bounds
        """
        try:
            # Update risk metrics
            self.risk_manager.update_risk_metrics()
            
            # Check kill switch
            if self.risk_manager.check_kill_switch():
                logger.critical("Kill switch activated!")
                return False
            
            # Check risk level
            if self.risk_manager.risk_level == RiskLevel.CRITICAL:
                logger.warning("Critical risk level detected")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    async def process_strategies(self):
        """
        Process all strategies and execute signals
        """
        try:
            for strategy_name, strategy in self.strategies.items():
                for symbol in strategy.symbols:
                    if symbol in self.current_prices:
                        # Get strategy data
                        if symbol in strategy.price_data:
                            data = strategy.price_data[symbol]
                            
                            # Generate signal
                            signal = strategy.generate_signal(symbol, data)
                            
                            if signal:
                                logger.info(f"Signal generated: {strategy_name} - {symbol} {signal.signal_type.value}")
                                
                                # Process signal
                                if strategy.process_signal(signal):
                                    # Execute order through execution engine
                                    await self.execute_signal(signal, strategy)
        
        except Exception as e:
            logger.error(f"Error processing strategies: {e}")
    
    async def execute_signal(self, signal: Signal, strategy: BaseStrategy):
        """
        Execute trading signal through execution engine
        """
        try:
            # Check risk limits before execution
            if not self.risk_manager.check_position_limits(
                signal.symbol, 
                strategy.positions[signal.symbol].size if signal.symbol in strategy.positions else 0,
                signal.price
            ):
                logger.warning(f"Position rejected by risk manager: {signal.symbol}")
                return
            
            # Determine order type and parameters
            order_type = OrderType.MARKET if signal.signal_type in [SignalType.BUY, SignalType.SELL] else OrderType.LIMIT
            
            # Get position size from strategy
            position_size = strategy.calculate_position_size(signal, signal.price)
            
            if position_size <= 0:
                logger.warning(f"Invalid position size: {position_size}")
                return
            
            # Place order
            order = await self.execution_engine.place_order(
                symbol=signal.symbol,
                side=signal.signal_type.value,
                size=position_size,
                order_type=order_type,
                price=signal.price if order_type == OrderType.LIMIT else None,
                urgency="HIGH" if signal.confidence > 0.8 else "NORMAL"
            )
            
            if order:
                logger.info(f"Order placed: {signal.symbol} {signal.signal_type.value} {position_size}")
                
                # Update risk manager
                self.risk_manager.update_portfolio_return(0)  # Will be updated when position closes
        
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    async def update_positions(self):
        """
        Update all strategy positions
        """
        try:
            for strategy in self.strategies.values():
                # Update positions with current prices
                strategy.update_positions(self.current_prices)
                
                # Update risk manager with PnL changes
                for symbol, position in strategy.positions.items():
                    if symbol in self.current_prices:
                        current_price = self.current_prices[symbol]
                        
                        # Calculate PnL change
                        if position.side == "LONG":
                            pnl_change = (current_price - position.entry_price) * position.size
                        else:
                            pnl_change = (position.entry_price - current_price) * position.size
                        
                        # Update risk manager
                        self.risk_manager.update_portfolio_return(pnl_change - position.pnl)
                        position.pnl = pnl_change
        
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def update_monitoring(self):
        """
        Update monitoring dashboard
        """
        if self.monitor:
            try:
                # Collect system metrics
                metrics = {
                    'timestamp': datetime.now(),
                    'total_pnl': self.total_pnl,
                    'daily_pnl': self.daily_pnl,
                    'trade_count': self.trade_count,
                    'open_positions': sum(len(s.positions) for s in self.strategies.values()),
                    'risk_level': self.risk_manager.risk_level.value if self.risk_manager else 'UNKNOWN',
                    'kill_switch_active': self.risk_manager.kill_switch_active if self.risk_manager else False
                }
                
                # Add strategy metrics
                for name, strategy in self.strategies.items():
                    strategy_metrics = strategy.get_performance_metrics()
                    metrics[f'{name}_pnl'] = strategy_metrics.get('total_pnl', 0)
                    metrics[f'{name}_win_rate'] = strategy_metrics.get('win_rate', 0)
                
                # Add execution metrics
                if self.execution_engine:
                    exec_metrics = self.execution_engine.get_execution_statistics()
                    metrics.update(exec_metrics)
                
                # Update monitor
                await self.monitor.update_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Error updating monitoring: {e}")
    
    async def log_status(self):
        """
        Log system status periodically
        """
        # Log every 5 minutes
        if datetime.now().minute % 5 == 0 and datetime.now().second < 10:
            logger.info("=== System Status ===")
            
            # Overall status
            logger.info(f"System running: {self.is_running}")
            logger.info(f"Total PnL: ${self.total_pnl:.2f}")
            logger.info(f"Daily PnL: ${self.daily_pnl:.2f}")
            logger.info(f"Trade count: {self.trade_count}")
            
            # Strategy status
            for name, strategy in self.strategies.items():
                strategy.log_status()
            
            # Risk status
            if self.risk_manager:
                self.risk_manager.log_risk_status()
            
            # Execution status
            if self.execution_engine:
                self.execution_engine.log_execution_status()
            
            logger.info("===================")
    
    def setup_signal_handlers(self):
        """
        Set up signal handlers for graceful shutdown
        """
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """
        Graceful shutdown of the trading system
        """
        if self.is_shutdown:
            return
        
        logger.info("Initiating system shutdown...")
        self.is_shutdown = True
        self.is_running = False
        
        try:
            # Close all positions
            await self.close_all_positions()
            
            # Stop data feed
            if self.data_feed:
                await self.data_feed.stop()
            
            # Stop monitor
            if self.monitor:
                await self.monitor.stop()
            
            # Log final statistics
            self.log_final_statistics()
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        finally:
            sys.exit(0)
    
    async def close_all_positions(self):
        """
        Close all open positions
        """
        logger.info("Closing all open positions...")
        
        try:
            for strategy in self.strategies.values():
                for symbol, position in list(strategy.positions.items()):
                    if symbol in self.current_prices:
                        current_price = self.current_prices[symbol]
                        strategy.close_position(symbol, current_price, datetime.now())
                        logger.info(f"Closed position: {symbol}")
        
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def log_final_statistics(self):
        """
        Log final trading statistics
        """
        logger.info("=== Final Trading Statistics ===")
        
        if self.start_time:
            runtime = datetime.now() - self.start_time
            logger.info(f"Total runtime: {runtime}")
        
        logger.info(f"Total PnL: ${self.total_pnl:.2f}")
        logger.info(f"Daily PnL: ${self.daily_pnl:.2f}")
        logger.info(f"Total trades: {self.trade_count}")
        
        # Strategy statistics
        for name, strategy in self.strategies.items():
            metrics = strategy.get_performance_metrics()
            logger.info(f"{name} Strategy:")
            logger.info(f"  Total PnL: ${metrics.get('total_pnl', 0):.2f}")
            logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        
        # Risk statistics
        if self.risk_manager:
            risk_report = self.risk_manager.get_risk_report()
            logger.info("Risk Statistics:")
            logger.info(f"  Max Drawdown: {risk_report.get('max_drawdown', 0):.2%}")
            logger.info(f"  VaR (95%): ${risk_report.get('var_95', 0):.2f}")
            logger.info(f"  Leverage Ratio: {risk_report.get('leverage_ratio', 0):.2f}")
        
        # Execution statistics
        if self.execution_engine:
            exec_stats = self.execution_engine.get_execution_statistics()
            logger.info("Execution Statistics:")
            logger.info(f"  Fill Rate: {exec_stats.get('fill_rate', 0):.2%}")
            logger.info(f"  Avg Latency: {exec_stats.get('avg_latency_ms', 0):.2f}ms")
            logger.info(f"  Total Volume: {exec_stats.get('total_volume', 0):,.0f}")
        
        logger.info("================================")

async def main():
    """
    Main entry point
    """
    # Create and start trading system
    trading_system = MT5TradingSystem()
    
    try:
        await trading_system.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await trading_system.shutdown()

if __name__ == "__main__":
    # Run the trading system
    asyncio.run(main())
