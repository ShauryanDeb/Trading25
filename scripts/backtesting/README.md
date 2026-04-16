# Backtesting Scripts

This directory contains scripts for backtesting trading strategies and performance analysis with various levels of sophistication.

## Core Files

### Basic Backtesting
- **`backtest.py`** - Basic walk-forward backtesting framework
- **`simple_backtest.py`** - Simple strategy backtesting
- **`random_backtest.py`** - Random strategy testing and baseline comparison

### Advanced Backtesting
- **`enhanced_backtest.py`** - Advanced backtesting with backtrader integration
- **`advanced_backtest.py`** - Sophisticated backtesting with realistic market constraints
- **`backtrader_backtest.py`** - Backtrader-based backtesting framework
- **`realistic_backtest.py`** - Realistic market simulation with slippage and fees

### Risk Management
- **`advanced_risk_manager.py`** - Risk management and position sizing algorithms

## Usage Examples

### Basic Backtesting
```python
from backtesting.backtest import backtest
cumulative_return = backtest('AAPL.csv', options_path='AAPL_options.csv')
```

### Enhanced Backtesting with Backtrader
```python
from backtesting.enhanced_backtest import run_backtest
run_backtest(
    model_path='model.pkl',
    price_file='AAPL.csv',
    options_file='AAPL_options.csv',
    threshold=0.55
)
```

### Advanced Backtesting with Realistic Constraints
```python
from backtesting.advanced_backtest import AdvancedBacktester
backtester = AdvancedBacktester(
    initial_capital=100000,
    commission_rate=0.001,
    slippage_rate=0.0005
)
results = backtester.run_backtest(df, predictions, threshold=0.6)
```

### Risk Management
```python
from backtesting.advanced_risk_manager import RiskManager
risk_manager = RiskManager(max_position_size=0.2, stop_loss=0.05)
position_size = risk_manager.calculate_position_size(portfolio_value, volatility)
```

## Backtesting Features

### Market Realism
- Transaction costs (commissions, slippage)
- Market impact modeling
- Realistic order execution
- Position sizing constraints
- Portfolio-level risk limits

### Performance Metrics
- Total return and annualized return
- Sharpe ratio and Sortino ratio
- Maximum drawdown
- Win rate and profit factor
- Calmar ratio and information ratio

### Risk Management
- Position sizing algorithms
- Stop-loss and take-profit orders
- Portfolio-level risk limits
- Drawdown protection
- Volatility-based position sizing

### Strategy Analysis
- Trade-by-trade analysis
- Entry/exit timing analysis
- Sector/asset allocation analysis
- Correlation analysis
- Stress testing

## Backtesting Frameworks

### Custom Framework
- Lightweight and fast
- Easy to customize
- Good for simple strategies
- Limited market realism

### Backtrader Integration
- Professional backtesting framework
- Rich feature set
- Good for complex strategies
- Extensive analysis capabilities

### Advanced Custom Framework
- Maximum realism
- Sophisticated risk management
- Custom market impact models
- Portfolio-level optimization

## Best Practices

1. **Start Simple**: Use basic backtesting for initial strategy validation
2. **Add Realism**: Progress to enhanced versions for more realistic results
3. **Risk Management**: Always implement proper risk controls
4. **Multiple Timeframes**: Test strategies across different time periods
5. **Out-of-Sample Testing**: Use walk-forward analysis for robust validation
6. **Stress Testing**: Test under various market conditions
7. **Transaction Costs**: Include realistic fees and slippage

## Performance Metrics

### Return Metrics
- Total Return
- Annualized Return
- Risk-Adjusted Return

### Risk Metrics
- Volatility
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional VaR

### Ratio Metrics
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Information Ratio

### Trade Metrics
- Win Rate
- Profit Factor
- Average Win/Loss
- Maximum Consecutive Losses 