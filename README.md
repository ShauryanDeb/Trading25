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

3. Train a model using the downloaded data and save it for later use:

```bash
python scripts/train_model.py apple.csv --model-out apple_model.pkl
```

The training script adds a number of technical indicators, including moving averages,
exponential moving averages, Bollinger Bands, MACD and RSI. It also computes
Stochastic Oscillator values, Average True Range, Commodity Channel Index and
On-Balance Volume. These features are used as inputs to a RandomForest classifier.

4. Run a simple walk-forward backtest:

```bash
python scripts/backtest.py apple.csv
```

5. Run a configurable backtest using the `backtrader` library:

```bash
python scripts/backtrader_backtest.py apple_model.pkl apple.csv --threshold 0.6 --stake 1 --commission 0.001
```

This backtest uses the trained model's predicted probabilities to trade. The `threshold`, `stake`, and `commission` options can be tuned to search for the best return.

6. Generate live predictions using the saved model:

```bash
python scripts/realtime.py apple_model.pkl AAPL --interval 60
```

This script polls Yahoo Finance for the most recent minute data and prints the model's prediction each cycle.

The training script will print classification metrics evaluating how well the model predicts the next-day price movement.

7. Evaluate a saved model on historical data:

```bash
python scripts/evaluate_model.py apple_model.pkl apple.csv
```

This script loads the saved model and CSV file and prints accuracy metrics on the known data so you can verify how well the model performs.

This code is intended as a minimal example for experimentation only. Use caution and thoroughly backtest any trading strategy before using it with real funds.
