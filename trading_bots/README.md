# Trading Bots

This directory contains various trading bot implementations for the trading model system.

## Files

### Virtual Trading Bots
- **`virtual_trading_bot.py`** - Basic virtual trading bot with simple portfolio management
- **`virtual_realtime_trading_bot.py`** - Real-time virtual trading bot with Alpaca integration
- **`virtual_realtime_trading_bot_fixed.py`** - Fixed version of the real-time trading bot with improved error handling

### Paper Trading
- **`alpaca_paper_trading_bot.py`** - Paper trading bot using Alpaca's paper trading environment

## Usage

### Basic Virtual Trading
```python
from trading_bots.virtual_trading_bot import VirtualTradingBot
bot = VirtualTradingBot(initial_cash=3500)
bot.run()
```

### Real-Time Virtual Trading
```python
from trading_bots.virtual_realtime_trading_bot_fixed import VirtualTradingBot
bot = VirtualTradingBot(
    initial_cash=3500,
    model_path='../models/ensemble_model.pkl',
    buy_threshold=0.65,
    sell_threshold=0.35
)
bot.run()
```

### Paper Trading
```python
from trading_bots.alpaca_paper_trading_bot import PaperTradingBot
bot = PaperTradingBot(api_key='your_key', secret_key='your_secret')
bot.run()
```

## Features

- Virtual portfolio simulation
- Real-time data integration
- Model-based decision making
- Risk management
- Performance tracking
- Error handling and recovery

## Configuration

Each bot can be configured with:
- Initial capital
- Trading thresholds
- Model parameters
- Risk limits
- Trading symbols 