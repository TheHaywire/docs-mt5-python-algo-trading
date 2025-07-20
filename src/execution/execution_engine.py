"""
Execution Engine for MT5 Python Algo Trading System
High-performance order execution with smart routing and market impact modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"

@dataclass
class Order:
    id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    order_type: OrderType
    size: float
    price: Optional[float]
    stop_price: Optional[float]
    timestamp: datetime
    status: OrderStatus
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    fills: List[Dict] = None
    
    def __post_init__(self):
        if self.fills is None:
            self.fills = []

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    timestamp: datetime
    fees: float = 0.0

class ExecutionEngine:
    """
    High-performance execution engine with:
    - Smart order routing
    - Market impact modeling
    - Latency optimization
    - Order book analysis
    - Slippage prediction
    """
    
    def __init__(self,
                 mt5_connection,
                 max_retries: int = 3,
                 retry_delay: float = 0.1,
                 order_timeout: float = 30.0,
                 use_iceberg: bool = True,
                 use_smart_routing: bool = True,
                 market_impact_model: bool = True):
        
        self.mt5_connection = mt5_connection
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.order_timeout = order_timeout
        self.use_iceberg = use_iceberg
        self.use_smart_routing = use_smart_routing
        self.market_impact_model = market_impact_model
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_counter = 0
        
        # Performance tracking
        self.execution_latency: List[float] = []
        self.slippage_data: Dict[str, List[float]] = {}
        self.fill_rates: Dict[str, float] = {}
        
        # Market data
        self.order_books: Dict[str, Dict] = {}
        self.market_impact_params: Dict[str, Dict] = {}
        
        # Execution statistics
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.total_volume = 0.0
        self.total_fees = 0.0
        
        logger.info("Initialized Execution Engine")
    
    def generate_order_id(self) -> str:
        """
        Generate unique order ID
        """
        self.order_counter += 1
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        return f"ORDER_{timestamp}_{self.order_counter}"
    
    async def place_order(self, 
                         symbol: str, 
                         side: str, 
                         size: float, 
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         urgency: str = "NORMAL") -> Optional[Order]:
        """
        Place order with smart execution logic
        """
        order_id = self.generate_order_id()
        timestamp = datetime.now()
        
        # Create order object
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            stop_price=stop_price,
            timestamp=timestamp,
            status=OrderStatus.PENDING
        )
        
        self.orders[order_id] = order
        self.total_orders += 1
        
        # Apply smart execution logic
        if self.use_smart_routing:
            order = await self.apply_smart_routing(order, urgency)
        
        # Apply market impact modeling
        if self.market_impact_model:
            order = self.apply_market_impact_model(order)
        
        # Execute order
        success = await self.execute_order(order)
        
        if success:
            self.successful_orders += 1
            logger.info(f"Order placed successfully: {order_id} - {symbol} {side} {size}")
            return order
        else:
            self.failed_orders += 1
            order.status = OrderStatus.REJECTED
            logger.error(f"Order failed: {order_id} - {symbol} {side} {size}")
            return None
    
    async def apply_smart_routing(self, order: Order, urgency: str) -> Order:
        """
        Apply smart order routing based on urgency and market conditions
        """
        symbol = order.symbol
        
        # Get market conditions
        spread = self.calculate_spread(symbol)
        liquidity = self.assess_liquidity(symbol)
        volatility = self.calculate_volatility(symbol)
        
        # Route based on urgency
        if urgency == "HIGH":
            # High urgency: use market orders or aggressive limits
            if order.order_type == OrderType.LIMIT:
                # Adjust limit price to be more aggressive
                if order.side == "BUY":
                    order.price = order.price * 1.001  # 0.1% more aggressive
                else:
                    order.price = order.price * 0.999  # 0.1% more aggressive
        
        elif urgency == "LOW":
            # Low urgency: use passive orders
            if order.order_type == OrderType.LIMIT:
                # Adjust limit price to be more passive
                if order.side == "BUY":
                    order.price = order.price * 0.999  # 0.1% more passive
                else:
                    order.price = order.price * 1.001  # 0.1% more passive
        
        # Use iceberg orders for large sizes
        if self.use_iceberg and order.size > self.get_iceberg_threshold(symbol):
            order.order_type = OrderType.ICEBERG
        
        return order
    
    def apply_market_impact_model(self, order: Order) -> Order:
        """
        Apply market impact model to adjust order parameters
        """
        symbol = order.symbol
        
        if symbol not in self.market_impact_params:
            return order
        
        params = self.market_impact_params[symbol]
        
        # Calculate expected market impact
        impact = self.calculate_market_impact(order.size, params)
        
        # Adjust order price based on impact
        if order.order_type == OrderType.LIMIT and order.price:
            if order.side == "BUY":
                order.price = order.price * (1 + impact)
            else:
                order.price = order.price * (1 - impact)
        
        return order
    
    def calculate_market_impact(self, size: float, params: Dict) -> float:
        """
        Calculate expected market impact
        """
        # Linear impact model: impact = alpha * size
        alpha = params.get('alpha', 0.0001)  # Impact per unit
        beta = params.get('beta', 0.5)       # Non-linear factor
        
        impact = alpha * (size ** beta)
        
        return min(impact, 0.01)  # Cap at 1%
    
    async def execute_order(self, order: Order) -> bool:
        """
        Execute order through MT5
        """
        start_time = time.time()
        
        try:
            # Prepare order parameters for MT5
            order_params = self.prepare_mt5_order(order)
            
            # Submit order to MT5
            result = await self.submit_mt5_order(order_params)
            
            if result:
                order.status = OrderStatus.SUBMITTED
                
                # Record execution latency
                latency = time.time() - start_time
                self.execution_latency.append(latency)
                
                # Monitor order status
                asyncio.create_task(self.monitor_order(order))
                
                return True
            else:
                order.status = OrderStatus.REJECTED
                return False
                
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    def prepare_mt5_order(self, order: Order) -> Dict:
        """
        Prepare order parameters for MT5
        """
        order_params = {
            'symbol': order.symbol,
            'type': self.map_order_type(order.order_type),
            'volume': order.size,
            'price': order.price if order.price else 0.0,
            'deviation': 10,  # Slippage tolerance
            'magic': 123456,  # Magic number for identification
            'comment': f"Python_Algo_{order.id}",
            'type_filling': 2,  # Filling type
            'type_time': 0     # Time type
        }
        
        # Add stop loss/take profit if specified
        if order.stop_price:
            if order.side == "BUY":
                order_params['sl'] = order.stop_price
            else:
                order_params['tp'] = order.stop_price
        
        return order_params
    
    def map_order_type(self, order_type: OrderType) -> int:
        """
        Map order type to MT5 constants
        """
        mapping = {
            OrderType.MARKET: 0,      # ORDER_TYPE_BUY/SELL
            OrderType.LIMIT: 2,       # ORDER_TYPE_BUY_LIMIT/SELL_LIMIT
            OrderType.STOP: 3,        # ORDER_TYPE_BUY_STOP/SELL_STOP
            OrderType.STOP_LIMIT: 4,  # ORDER_TYPE_BUY_STOP_LIMIT/SELL_STOP_LIMIT
            OrderType.ICEBERG: 2      # Use limit orders for iceberg
        }
        
        return mapping.get(order_type, 0)
    
    async def submit_mt5_order(self, order_params: Dict) -> bool:
        """
        Submit order to MT5 (simulated for now)
        """
        # Simulate MT5 order submission
        await asyncio.sleep(0.001)  # Simulate network latency
        
        # Simulate success/failure based on market conditions
        success_rate = 0.95  # 95% success rate
        success = np.random.random() < success_rate
        
        if success:
            # Simulate order fill
            await asyncio.sleep(0.01)  # Simulate fill time
            self.simulate_order_fill(order_params)
        
        return success
    
    def simulate_order_fill(self, order_params: Dict):
        """
        Simulate order fill (replace with actual MT5 callback)
        """
        order_id = order_params['comment'].split('_')[-1]
        
        if order_id in self.orders:
            order = self.orders[order_id]
            
            # Create fill
            fill = Fill(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                price=order.price or self.get_current_price(order.symbol),
                timestamp=datetime.now(),
                fees=self.calculate_fees(order.size, order.price)
            )
            
            self.fills.append(fill)
            order.fills.append(fill)
            order.filled_size = order.size
            order.avg_fill_price = fill.price
            order.status = OrderStatus.FILLED
            
            # Update statistics
            self.total_volume += fill.size
            self.total_fees += fill.fees
            
            # Calculate slippage
            self.calculate_slippage(order, fill)
            
            logger.info(f"Order filled: {order_id} - {fill.size} @ {fill.price}")
    
    async def monitor_order(self, order: Order):
        """
        Monitor order status and handle partial fills
        """
        timeout = time.time() + self.order_timeout
        
        while time.time() < timeout and order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            await asyncio.sleep(0.1)
            
            # Check for partial fills (simulated)
            if order.status == OrderStatus.SUBMITTED and np.random.random() < 0.1:
                # Simulate partial fill
                partial_size = order.size * 0.5
                fill_price = order.price or self.get_current_price(order.symbol)
                
                fill = Fill(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    size=partial_size,
                    price=fill_price,
                    timestamp=datetime.now(),
                    fees=self.calculate_fees(partial_size, fill_price)
                )
                
                self.fills.append(fill)
                order.fills.append(fill)
                order.filled_size += partial_size
                order.status = OrderStatus.PARTIAL_FILL
                
                logger.info(f"Partial fill: {order.id} - {partial_size} @ {fill_price}")
        
        # Handle timeout
        if order.status == OrderStatus.SUBMITTED:
            await self.cancel_order(order.id)
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        # Simulate order cancellation
        await asyncio.sleep(0.001)
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Order cancelled: {order_id}")
        
        return True
    
    def update_order_book(self, symbol: str, order_book: Dict):
        """
        Update order book data
        """
        self.order_books[symbol] = order_book
    
    def calculate_spread(self, symbol: str) -> float:
        """
        Calculate current bid-ask spread
        """
        if symbol not in self.order_books:
            return 0.0
        
        order_book = self.order_books[symbol]
        
        if 'bids' in order_book and 'asks' in order_book:
            best_bid = max(order_book['bids'].keys())
            best_ask = min(order_book['asks'].keys())
            return best_ask - best_bid
        
        return 0.0
    
    def assess_liquidity(self, symbol: str) -> float:
        """
        Assess market liquidity
        """
        if symbol not in self.order_books:
            return 0.5
        
        order_book = self.order_books[symbol]
        
        if 'bids' not in order_book or 'asks' not in order_book:
            return 0.5
        
        # Calculate total volume at top levels
        bid_volume = sum(list(order_book['bids'].values())[:3])
        ask_volume = sum(list(order_book['asks'].values())[:3])
        
        total_volume = bid_volume + ask_volume
        
        # Normalize to 0-1 scale
        liquidity_score = min(total_volume / 10000, 1.0)  # Normalize to 10k volume
        
        return liquidity_score
    
    def calculate_volatility(self, symbol: str) -> float:
        """
        Calculate current volatility
        """
        # Simplified volatility calculation
        return 0.001  # Default 0.1% volatility
    
    def get_iceberg_threshold(self, symbol: str) -> float:
        """
        Get iceberg order threshold for symbol
        """
        # Symbol-specific thresholds
        thresholds = {
            'EURUSD': 100000,
            'GBPUSD': 50000,
            'USDJPY': 100000,
            'default': 50000
        }
        
        return thresholds.get(symbol, thresholds['default'])
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price
        """
        # Simplified price retrieval
        return 1.1000  # Default EURUSD price
    
    def calculate_fees(self, size: float, price: float) -> float:
        """
        Calculate trading fees
        """
        # Simplified fee calculation
        fee_rate = 0.0001  # 1 pip per lot
        return size * fee_rate
    
    def calculate_slippage(self, order: Order, fill: Fill):
        """
        Calculate and record slippage
        """
        if order.price and order.price > 0:
            slippage = abs(fill.price - order.price) / order.price
            
            if order.symbol not in self.slippage_data:
                self.slippage_data[order.symbol] = []
            
            self.slippage_data[order.symbol].append(slippage)
    
    def get_execution_statistics(self) -> Dict:
        """
        Get execution performance statistics
        """
        if not self.execution_latency:
            return {}
        
        avg_latency = np.mean(self.execution_latency)
        p95_latency = np.percentile(self.execution_latency, 95)
        p99_latency = np.percentile(self.execution_latency, 99)
        
        # Calculate average slippage
        avg_slippage = {}
        for symbol, slippages in self.slippage_data.items():
            if slippages:
                avg_slippage[symbol] = np.mean(slippages)
        
        # Calculate fill rates
        fill_rate = self.successful_orders / self.total_orders if self.total_orders > 0 else 0.0
        
        return {
            'total_orders': self.total_orders,
            'successful_orders': self.successful_orders,
            'failed_orders': self.failed_orders,
            'fill_rate': fill_rate,
            'total_volume': self.total_volume,
            'total_fees': self.total_fees,
            'avg_latency_ms': avg_latency * 1000,
            'p95_latency_ms': p95_latency * 1000,
            'p99_latency_ms': p99_latency * 1000,
            'avg_slippage': avg_slippage,
            'open_orders': len([o for o in self.orders.values() if o.status == OrderStatus.SUBMITTED])
        }
    
    def log_execution_status(self):
        """
        Log execution engine status
        """
        stats = self.get_execution_statistics()
        
        logger.info(f"Execution Engine Status:")
        logger.info(f"  Total Orders: {stats.get('total_orders', 0)}")
        logger.info(f"  Fill Rate: {stats.get('fill_rate', 0):.2%}")
        logger.info(f"  Avg Latency: {stats.get('avg_latency_ms', 0):.2f}ms")
        logger.info(f"  Total Volume: {stats.get('total_volume', 0):,.0f}")
        logger.info(f"  Total Fees: ${stats.get('total_fees', 0):.2f}")
        logger.info(f"  Open Orders: {stats.get('open_orders', 0)}") 