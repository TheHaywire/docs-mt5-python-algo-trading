"""
Monitoring System for MT5 Python Algo Trading System
Real-time dashboard, alerts, and performance tracking
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    timestamp: datetime
    level: AlertLevel
    message: str
    source: str
    data: Dict

class TradingMonitor:
    """
    Real-time monitoring system for trading operations
    """
    
    def __init__(self, 
                 port: int = 8080,
                 alert_threshold: float = 0.05,
                 update_interval: float = 1.0):
        
        self.port = port
        self.alert_threshold = alert_threshold
        self.update_interval = update_interval
        
        # State
        self.is_running = False
        self.server = None
        
        # Data storage
        self.metrics_history: List[Dict] = []
        self.alerts: List[Alert] = []
        self.performance_data: Dict = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            'daily_loss': -0.05,  # 5% daily loss
            'drawdown': -0.15,    # 15% drawdown
            'var_95': 0.02,       # 2% VaR
            'leverage': 2.0,      # 2x leverage
            'concentration': 0.1   # 10% concentration
        }
        
        # Performance tracking
        self.start_time = None
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        
        logger.info(f"Initialized TradingMonitor on port {port}")
    
    async def start(self):
        """
        Start the monitoring system
        """
        if self.is_running:
            logger.warning("TradingMonitor is already running")
            return
        
        logger.info("Starting TradingMonitor...")
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start monitoring loop
        asyncio.create_task(self.monitoring_loop())
        
        # Start web server (simulated)
        asyncio.create_task(self.start_web_server())
        
        logger.info("TradingMonitor started successfully")
    
    async def stop(self):
        """
        Stop the monitoring system
        """
        logger.info("Stopping TradingMonitor...")
        self.is_running = False
        
        if self.server:
            await self.server.close()
    
    async def monitoring_loop(self):
        """
        Main monitoring loop
        """
        logger.info("Starting monitoring loop...")
        
        while self.is_running:
            try:
                # Check for alerts
                await self.check_alerts()
                
                # Update performance metrics
                await self.update_performance_metrics()
                
                # Clean up old data
                await self.cleanup_old_data()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def start_web_server(self):
        """
        Start web server for dashboard (simulated)
        """
        logger.info(f"Starting web server on port {self.port}")
        
        # Simulate web server
        while self.is_running:
            await asyncio.sleep(1)
    
    async def update_metrics(self, metrics: Dict):
        """
        Update system metrics
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in metrics:
                metrics['timestamp'] = datetime.now()
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Keep only recent metrics (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m['timestamp'] > cutoff_time
            ]
            
            # Update performance tracking
            self.update_performance_tracking(metrics)
            
            # Check for alerts
            await self.check_metric_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def update_performance_tracking(self, metrics: Dict):
        """
        Update performance tracking data
        """
        try:
            # Update total PnL
            if 'total_pnl' in metrics:
                self.total_pnl = metrics['total_pnl']
            
            # Update peak value and drawdown
            if 'total_pnl' in metrics:
                current_value = self.total_pnl
                
                if current_value > self.peak_value:
                    self.peak_value = current_value
                
                # Calculate drawdown
                if self.peak_value > 0:
                    drawdown = (current_value - self.peak_value) / self.peak_value
                    self.max_drawdown = min(self.max_drawdown, drawdown)
            
            # Store performance data
            self.performance_data = {
                'total_pnl': self.total_pnl,
                'max_drawdown': self.max_drawdown,
                'peak_value': self.peak_value,
                'runtime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
        
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    async def check_metric_alerts(self, metrics: Dict):
        """
        Check metrics for alert conditions
        """
        try:
            # Check daily loss
            if 'daily_pnl' in metrics:
                daily_pnl = metrics['daily_pnl']
                if daily_pnl < self.alert_thresholds['daily_loss']:
                    await self.create_alert(
                        AlertLevel.WARNING,
                        f"Daily loss threshold exceeded: {daily_pnl:.2%}",
                        "risk_manager",
                        {'daily_pnl': daily_pnl, 'threshold': self.alert_thresholds['daily_loss']}
                    )
            
            # Check drawdown
            if 'max_drawdown' in metrics:
                drawdown = metrics['max_drawdown']
                if drawdown < self.alert_thresholds['drawdown']:
                    await self.create_alert(
                        AlertLevel.WARNING,
                        f"Drawdown threshold exceeded: {drawdown:.2%}",
                        "risk_manager",
                        {'drawdown': drawdown, 'threshold': self.alert_thresholds['drawdown']}
                    )
            
            # Check VaR
            if 'var_95' in metrics:
                var_95 = metrics['var_95']
                if var_95 > self.alert_thresholds['var_95']:
                    await self.create_alert(
                        AlertLevel.ERROR,
                        f"VaR threshold exceeded: {var_95:.2%}",
                        "risk_manager",
                        {'var_95': var_95, 'threshold': self.alert_thresholds['var_95']}
                    )
            
            # Check leverage
            if 'leverage_ratio' in metrics:
                leverage = metrics['leverage_ratio']
                if leverage > self.alert_thresholds['leverage']:
                    await self.create_alert(
                        AlertLevel.WARNING,
                        f"Leverage threshold exceeded: {leverage:.2f}x",
                        "risk_manager",
                        {'leverage': leverage, 'threshold': self.alert_thresholds['leverage']}
                    )
            
            # Check kill switch
            if 'kill_switch_active' in metrics and metrics['kill_switch_active']:
                await self.create_alert(
                    AlertLevel.CRITICAL,
                    "Kill switch activated - trading stopped",
                    "risk_manager",
                    {'kill_switch_active': True}
                )
        
        except Exception as e:
            logger.error(f"Error checking metric alerts: {e}")
    
    async def create_alert(self, level: AlertLevel, message: str, source: str, data: Dict):
        """
        Create and store an alert
        """
        try:
            alert = Alert(
                timestamp=datetime.now(),
                level=level,
                message=message,
                source=source,
                data=data
            )
            
            # Store alert
            self.alerts.append(alert)
            
            # Keep only recent alerts (last 1000)
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]
            
            # Log alert
            log_level = getattr(logging, level.value)
            logger.log(log_level, f"ALERT [{source}]: {message}")
            
            # Send notification (simulated)
            await self.send_notification(alert)
        
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    async def send_notification(self, alert: Alert):
        """
        Send notification for alert (simulated)
        """
        try:
            # Simulate notification sending
            if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                logger.critical(f"NOTIFICATION: {alert.message}")
            elif alert.level == AlertLevel.WARNING:
                logger.warning(f"NOTIFICATION: {alert.message}")
            else:
                logger.info(f"NOTIFICATION: {alert.message}")
        
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def check_alerts(self):
        """
        Check for system-wide alerts
        """
        try:
            # Check system health
            if not self.is_running:
                await self.create_alert(
                    AlertLevel.ERROR,
                    "Monitoring system not running",
                    "monitor",
                    {'is_running': False}
                )
            
            # Check data freshness
            if self.metrics_history:
                last_metric = self.metrics_history[-1]
                time_since_update = (datetime.now() - last_metric['timestamp']).total_seconds()
                
                if time_since_update > 60:  # No updates for 1 minute
                    await self.create_alert(
                        AlertLevel.WARNING,
                        f"No metrics update for {time_since_update:.0f} seconds",
                        "monitor",
                        {'time_since_update': time_since_update}
                    )
        
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def update_performance_metrics(self):
        """
        Update performance metrics
        """
        try:
            # Calculate additional metrics
            if self.metrics_history:
                recent_metrics = self.metrics_history[-100:]  # Last 100 metrics
                
                # Calculate average PnL
                pnls = [m.get('total_pnl', 0) for m in recent_metrics]
                avg_pnl = sum(pnls) / len(pnls) if pnls else 0
                
                # Calculate volatility
                if len(pnls) > 1:
                    volatility = (sum((p - avg_pnl) ** 2 for p in pnls) / (len(pnls) - 1)) ** 0.5
                else:
                    volatility = 0
                
                # Update performance data
                self.performance_data.update({
                    'avg_pnl': avg_pnl,
                    'volatility': volatility,
                    'metric_count': len(self.metrics_history),
                    'alert_count': len(self.alerts)
                })
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def cleanup_old_data(self):
        """
        Clean up old data
        """
        try:
            # Clean up old alerts (older than 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def get_dashboard_data(self) -> Dict:
        """
        Get data for dashboard
        """
        try:
            # Get latest metrics
            latest_metrics = self.metrics_history[-1] if self.metrics_history else {}
            
            # Get recent alerts
            recent_alerts = self.alerts[-10:]  # Last 10 alerts
            
            # Get performance summary
            performance_summary = {
                'total_pnl': self.total_pnl,
                'max_drawdown': self.max_drawdown,
                'peak_value': self.peak_value,
                'runtime_hours': self.performance_data.get('runtime', 0) / 3600,
                'avg_pnl': self.performance_data.get('avg_pnl', 0),
                'volatility': self.performance_data.get('volatility', 0)
            }
            
            # Get system status
            system_status = {
                'is_running': self.is_running,
                'metric_count': len(self.metrics_history),
                'alert_count': len(self.alerts),
                'last_update': latest_metrics.get('timestamp'),
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
            
            return {
                'performance': performance_summary,
                'system_status': system_status,
                'latest_metrics': latest_metrics,
                'recent_alerts': [asdict(alert) for alert in recent_alerts],
                'alert_thresholds': self.alert_thresholds
            }
        
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def get_alert_summary(self) -> Dict:
        """
        Get alert summary
        """
        try:
            alert_counts = {
                AlertLevel.INFO: 0,
                AlertLevel.WARNING: 0,
                AlertLevel.ERROR: 0,
                AlertLevel.CRITICAL: 0
            }
            
            for alert in self.alerts:
                alert_counts[alert.level] += 1
            
            return {
                'total_alerts': len(self.alerts),
                'alert_counts': {level.value: count for level, count in alert_counts.items()},
                'recent_alerts': [asdict(alert) for alert in self.alerts[-20:]]  # Last 20 alerts
            }
        
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {}
    
    def get_performance_report(self) -> Dict:
        """
        Get detailed performance report
        """
        try:
            if not self.metrics_history:
                return {}
            
            # Calculate performance metrics
            pnls = [m.get('total_pnl', 0) for m in self.metrics_history]
            
            if len(pnls) > 1:
                returns = [pnls[i] - pnls[i-1] for i in range(1, len(pnls))]
                avg_return = sum(returns) / len(returns)
                volatility = (sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            else:
                avg_return = 0
                volatility = 0
                sharpe_ratio = 0
            
            return {
                'total_pnl': self.total_pnl,
                'max_drawdown': self.max_drawdown,
                'peak_value': self.peak_value,
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'total_metrics': len(self.metrics_history),
                'runtime_hours': self.performance_data.get('runtime', 0) / 3600
            }
        
        except Exception as e:
            logger.error(f"Error getting performance report: {e}")
            return {}
    
    def log_monitor_status(self):
        """
        Log monitoring system status
        """
        dashboard_data = self.get_dashboard_data()
        alert_summary = self.get_alert_summary()
        
        logger.info("=== Trading Monitor Status ===")
        logger.info(f"Running: {self.is_running}")
        logger.info(f"Total PnL: ${dashboard_data.get('performance', {}).get('total_pnl', 0):.2f}")
        logger.info(f"Max Drawdown: {dashboard_data.get('performance', {}).get('max_drawdown', 0):.2%}")
        logger.info(f"Runtime: {dashboard_data.get('performance', {}).get('runtime_hours', 0):.1f} hours")
        logger.info(f"Metrics Count: {dashboard_data.get('system_status', {}).get('metric_count', 0)}")
        logger.info(f"Total Alerts: {alert_summary.get('total_alerts', 0)}")
        logger.info("==============================") 