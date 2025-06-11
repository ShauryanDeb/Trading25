# Trading25

This project provides a simple starting point for experimenting with stock trading models. It includes scripts for downloading historical data, generating technical indicators, and training a basic machine learning model.

## Setup

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Download historical data. For example, to download the full history for Apple:

```bash
python scripts/data.py AAPL --start none --output apple.csv
```

Passing `--start none` uses the maximum history available from Yahoo Finance. You can also specify a custom start date if you only want recent data.

3. Train a model using the downloaded data:

```bash
python scripts/train_model.py apple.csv
```

The training script adds a number of technical indicators, including moving averages,
exponential moving averages, Bollinger Bands, MACD and RSI. These features are used
as inputs to a RandomForest classifier.

4. Run a simple walk-forward backtest:

```bash
python scripts/backtest.py apple.csv
```

The training script will print classification metrics evaluating how well the model predicts the next-day price movement.

This code is intended as a minimal example for experimentation only. Use caution and thoroughly backtest any trading strategy before using it with real funds.
