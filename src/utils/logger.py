"""
Logging utilities for MT5 Python Algo Trading System
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional
import json

def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging for the trading system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    json_formatter = JsonFormatter()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Error file handler
    if log_file:
        error_log_file = log_file.replace('.log', '_error.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    # Trading-specific log file
    trading_log_file = "logs/trading.log"
    if not os.path.exists(os.path.dirname(trading_log_file)):
        os.makedirs(os.path.dirname(trading_log_file), exist_ok=True)
    
    trading_handler = logging.handlers.RotatingFileHandler(
        trading_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(json_formatter)
    logger.addHandler(trading_handler)
    
    return logger

class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    """
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

class TradingLogger:
    """
    Specialized logger for trading events
    """
    
    def __init__(self, name: str = "trading"):
        self.logger = logging.getLogger(name)
    
    def log_signal(self, signal_data: dict):
        """
        Log trading signal
        """
        self.logger.info("Trading Signal", extra={
            'extra_fields': {
                'event_type': 'signal',
                'signal_data': signal_data
            }
        })
    
    def log_order(self, order_data: dict):
        """
        Log order placement
        """
        self.logger.info("Order Placed", extra={
            'extra_fields': {
                'event_type': 'order',
                'order_data': order_data
            }
        })
    
    def log_fill(self, fill_data: dict):
        """
        Log order fill
        """
        self.logger.info("Order Filled", extra={
            'extra_fields': {
                'event_type': 'fill',
                'fill_data': fill_data
            }
        })
    
    def log_position(self, position_data: dict):
        """
        Log position update
        """
        self.logger.info("Position Update", extra={
            'extra_fields': {
                'event_type': 'position',
                'position_data': position_data
            }
        })
    
    def log_risk_alert(self, risk_data: dict):
        """
        Log risk alert
        """
        self.logger.warning("Risk Alert", extra={
            'extra_fields': {
                'event_type': 'risk_alert',
                'risk_data': risk_data
            }
        })
    
    def log_performance(self, performance_data: dict):
        """
        Log performance metrics
        """
        self.logger.info("Performance Update", extra={
            'extra_fields': {
                'event_type': 'performance',
                'performance_data': performance_data
            }
        })

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name
    """
    return logging.getLogger(name)

def log_function_call(func):
    """
    Decorator to log function calls
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    return wrapper

def log_execution_time(func):
    """
    Decorator to log function execution time
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    return wrapper 