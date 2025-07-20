# MT5 Python Algo Trading System

A professional, modular, and extensible algorithmic trading system for MetaTrader 5 (MT5) using Python. Designed for high-frequency, multi-strategy, risk-aware trading with robust monitoring, logging, and risk management.

## Features
- Multi-strategy, plug-and-play architecture
- Real-time and historical data ingestion
- Advanced risk management and kill switches
- Backtesting and walk-forward analysis
- Modular execution and order management
- Comprehensive monitoring, logging, and alerting
- Secure credential and configuration management
- Extensible for new strategies, data sources, and brokers

## Folder Structure
```
/docs         # Documentation (vision, architecture, risk, etc.)
/src          # Source code (modular components)
/config       # Configuration files (YAML)
/tests        # Unit and integration tests
README.md     # Project overview and instructions
requirements.txt # Python dependencies
```

## Getting Started
1. **Install MetaTrader 5 (MT5)** on Windows and set up your broker account.
2. **Clone this repository** and set up a Python 3.9+ environment.
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure your credentials and strategies** in `config/config.yaml`.
5. **Run the main engine:**
   ```bash
   python src/main.py
   ```

## Documentation
See the `/docs` directory for detailed design, architecture, and operational guides.

## Disclaimer
This system is for educational and research purposes. Live trading involves significant risk. Use at your own discretion.
