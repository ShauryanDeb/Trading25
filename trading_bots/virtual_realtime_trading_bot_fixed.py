import os
import time
import pytz
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv
from scripts.enhanced_features import add_technical_indicators
from scripts.ensemble_learning import EnsembleModel
from scripts.advanced_ensemble_model import AdvancedEnsembleModel

# Load Alpaca API keys from .env
load_dotenv()
API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

# Trading parameters
STARTING_CASH = 3500
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35
STOCK_LIST = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'TSLA', 'JPM', 'V', 'MA',
    'UNH', 'HD', 'COST', 'LLY', 'AVGO', 'PEP', 'MCD', 'BAC', 'WMT', 'DIS'
]
BAR_TIMEFRAME = '1Min'
WINDOW_SIZE = 50  # Number of bars for rolling feature calculation
TRADES_CSV = 'virtual_realtime_trades_log.csv'
PORTFOLIO_CSV = 'virtual_realtime_portfolio_log.csv'

# Load model
print("Loading ensemble model...")
model_data = joblib.load('new_advanced_ensemble_model.pkl')
ensemble = model_data['ensemble']
selected_features = model_data['selected_features']

# Initialize Alpaca API
api = REST(API_KEY, API_SECRET, BASE_URL)

# Initialize virtual portfolio
cash = STARTING_CASH
holdings = {symbol: 0 for symbol in STOCK_LIST}
trades = []
portfolio_history = []

# Helper: Retry logic for network calls
import functools
import random

def retry_on_exception(max_attempts=5, initial_wait=5, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            wait = initial_wait
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    print(f"[Network Error] {e}. Attempt {attempts}/{max_attempts}. Retrying in {wait}s...")
                    time.sleep(wait + random.uniform(0, 2))
                    wait *= backoff
            print(f"[Network Error] Max attempts reached for {func.__name__}. Skipping this cycle.")
            return None
        return wrapper
    return decorator

# Helper: Check if market is open
@retry_on_exception(max_attempts=10, initial_wait=5, backoff=2, exceptions=(Exception,))
def is_market_open():
    clock = api.get_clock()
    return clock.is_open

# Helper: Get current US Eastern time
def get_eastern_now():
    return datetime.now(pytz.timezone('US/Eastern'))

# Helper: Get most recent 1-minute bars for all stocks
@retry_on_exception(max_attempts=10, initial_wait=5, backoff=2, exceptions=(Exception,))
def get_latest_bars(symbols, window):
    end = datetime.utcnow()
    start = end - timedelta(minutes=window+5)  # buffer for missing bars
    bars = api.get_bars(symbols, BAR_TIMEFRAME, start=start, end=end, adjustment='raw').df
    # Ensure we have a DataFrame for each symbol
    bar_dict = {}
    for symbol in symbols:
        df = bars[bars['symbol'] == symbol].copy()
        df = df.tail(window)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            bar_dict[symbol] = df
    return bar_dict

# Main trading loop
print("Starting real-time virtual trading bot...")
while True:
    now = get_eastern_now()
    # Only run during market hours (9:30am-4:00pm ET, Mon-Fri)
    try:
        market_open = is_market_open()
    except Exception as e:
        print(f"[Error] Could not check market status: {e}. Sleeping 60s...")
        time.sleep(60)
        continue
    if now.weekday() >= 5 or not market_open:
        print(f"Market closed ({now.strftime('%Y-%m-%d %H:%M:%S')}). Sleeping 60s...")
        time.sleep(60)
        continue

    # Get latest bars for all stocks
    bar_dict = get_latest_bars(STOCK_LIST, WINDOW_SIZE)
    if bar_dict is None or len(bar_dict) == 0:
        print(f"[Warning] No data received from Alpaca. Skipping this cycle.")
        time.sleep(10)
        continue
    for symbol, df in bar_dict.items():
        if len(df) < WINDOW_SIZE:
            print(f"[Warning] Not enough data for {symbol}. Skipping.")
            continue  # Not enough data for features
        # Feature engineering
        try:
            feature_df = add_technical_indicators(df.copy())
        except Exception as e:
            print(f"[Feature Error] {symbol}: {e}. Skipping.")
            continue
        # Only use selected features
        if not all(f in feature_df.columns for f in selected_features):
            print(f"[Warning] Missing features for {symbol}. Skipping.")
            continue
        X_current = feature_df[selected_features].iloc[[-1]]
        # Model prediction
        try:
            pred, proba, *_ = ensemble.predict_ensemble(X_current)
            probability = proba[0]
        except Exception as e:
            print(f"[Model Error] {symbol}: {e}. Skipping.")
            continue
        current_price = df['close'].iloc[-1]
        # Trading logic
        # Buy
        if holdings[symbol] == 0 and probability > BUY_THRESHOLD and cash >= current_price:
            shares_to_buy = int(cash / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                cash -= cost
                holdings[symbol] += shares_to_buy
                trades.append({
                    'datetime': now,
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'probability': probability,
                    'cash': cash
                })
                print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} BUY {shares_to_buy} {symbol} @ ${current_price:.2f} (prob: {probability:.3f})")
        # Sell
        elif holdings[symbol] > 0 and probability < SELL_THRESHOLD:
            shares_to_sell = holdings[symbol]
            proceeds = shares_to_sell * current_price
            cash += proceeds
            trades.append({
                'datetime': now,
                'symbol': symbol,
                'action': 'SELL',
                'price': current_price,
                'shares': shares_to_sell,
                'probability': probability,
                'cash': cash
            })
            print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} SELL {shares_to_sell} {symbol} @ ${current_price:.2f} (prob: {probability:.3f})")
            holdings[symbol] = 0
        # Hold: do nothing
        # Record portfolio value
        try:
            total_value = cash + sum(holdings[s] * bar_dict[s]['close'].iloc[-1] for s in STOCK_LIST if s in bar_dict and not bar_dict[s].empty)
        except Exception as e:
            print(f"[Portfolio Error] {e}. Skipping portfolio value update.")
            total_value = cash
        portfolio_history.append({
            'datetime': now,
            'cash': cash,
            'holdings': holdings.copy(),
            'portfolio_value': total_value
        })
    # Save logs
    try:
        pd.DataFrame(trades).to_csv(TRADES_CSV, index=False)
        pd.DataFrame(portfolio_history).to_csv(PORTFOLIO_CSV, index=False)
    except Exception as e:
        print(f"[Logging Error] {e}. Skipping log save this cycle.")
    # Sleep until next minute
    time_to_next_minute = 60 - datetime.now().second
    time.sleep(max(1, time_to_next_minute)) 