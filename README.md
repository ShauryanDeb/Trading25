# Trading25

This project provides a simple starting point for experimenting with stock trading models. It includes scripts for downloading historical data, generating technical indicators, and training a basic machine learning model.

## Setup

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Download historical data. For example, to download daily data for Apple:

```bash
python scripts/data.py AAPL --start 2020-01-01 --output apple.csv
```

3. Train a model using the downloaded data:

```bash
python scripts/train_model.py apple.csv
```

4. Run a simple walk-forward backtest:

```bash
python scripts/backtest.py apple.csv
```

The training script will print classification metrics evaluating how well the model predicts the next-day price movement.

This code is intended as a minimal example for experimentation only. Use caution and thoroughly backtest any trading strategy before using it with real funds.
