import os
import time
import pytz
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv
import matplotlib.pyplot as plt
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
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'TSLA'
]
BAR_TIMEFRAME = '1Min'
WINDOW_SIZE = 50  # Number of bars for rolling feature calculation

# Load model
print("Loading ensemble model...")
model_data = joblib.load('new_advanced_ensemble_model.pkl')
ensemble = model_data['ensemble']
selected_features = model_data['selected_features']

# Initialize Alpaca API
api = REST(API_KEY, API_SECRET, BASE_URL)

# 1. Download 1-minute bar data for the last month for all stocks
def get_last_month_bars(symbols, window=WINDOW_SIZE):
    end = datetime.utcnow()
    start = end - timedelta(days=31)
    # Format as RFC3339/ISO8601 for Alpaca
    start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')
    bars = api.get_bars(symbols, BAR_TIMEFRAME, start=start_str, end=end_str, adjustment='raw', feed='iex').df
    bar_dict = {}
    for symbol in symbols:
        df = bars[bars['symbol'] == symbol].copy()
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            bar_dict[symbol] = df
    return bar_dict

print("Downloading 1-minute bars for last month...")
bar_dict = get_last_month_bars(STOCK_LIST)

# Align all stocks to the same timestamps (inner join)
all_indices = set.intersection(*(set(df.index) for df in bar_dict.values() if not df.empty))
all_indices = sorted(all_indices)

# 2. Initialize portfolios for both bots
class VirtualPortfolio:
    def __init__(self, starting_cash, stock_list):
        self.cash = starting_cash
        self.holdings = {symbol: 0 for symbol in stock_list}
        self.history = []
        self.trades = []
    def value(self, prices):
        return self.cash + sum(self.holdings[s] * prices.get(s, 0) for s in self.holdings)
    def record(self, dt, prices):
        self.history.append({
            'datetime': dt,
            'cash': self.cash,
            'holdings': self.holdings.copy(),
            'portfolio_value': self.value(prices)
        })
    def log_trade(self, dt, symbol, action, price, shares):
        self.trades.append({
            'datetime': dt,
            'symbol': symbol,
            'action': action,
            'price': price,
            'shares': shares,
            'cash': self.cash
        })

# 3. Run both bots in parallel
def run_bots(bar_dict, all_indices):
    base_portfolio = VirtualPortfolio(STARTING_CASH, STOCK_LIST)
    model_portfolio = VirtualPortfolio(STARTING_CASH, STOCK_LIST)
    # For base bot: alternate buy/sell per stock, start with buy
    base_state = {symbol: 'buy' for symbol in STOCK_LIST}
    # For model bot: need rolling window for features
    rolling_windows = {symbol: bar_dict[symbol].copy() for symbol in STOCK_LIST}
    log_count = 0
    for i, dt in enumerate(all_indices):
        prices = {s: bar_dict[s].loc[dt]['close'] for s in STOCK_LIST if dt in bar_dict[s].index}
        # --- BASE BOT (true 50/50 rebalance) ---
        base_portfolio_value = base_portfolio.value(prices)
        if len(prices) > 0:
            target_stock_value = base_portfolio_value / 2 / len(STOCK_LIST)
            for symbol in STOCK_LIST:
                price = prices.get(symbol)
                if price is None:
                    continue
                target_shares = int(target_stock_value / price)
                current_shares = base_portfolio.holdings[symbol]
                diff = target_shares - current_shares
                if diff > 0:
                    # Buy up to target
                    cost = diff * price
                    if cost > base_portfolio.cash:
                        diff = int(base_portfolio.cash / price)
                        cost = diff * price
                    if diff > 0:
                        base_portfolio.cash -= cost
                        base_portfolio.holdings[symbol] += diff
                        base_portfolio.log_trade(dt, symbol, 'BUY', price, diff)
                elif diff < 0:
                    # Sell down to target
                    shares_to_sell = -diff
                    base_portfolio.cash += shares_to_sell * price
                    base_portfolio.holdings[symbol] -= shares_to_sell
                    base_portfolio.log_trade(dt, symbol, 'SELL', price, shares_to_sell)
        base_portfolio.record(dt, prices)
        # --- MODEL BOT ---
        for symbol in STOCK_LIST:
            df = bar_dict[symbol]
            if dt not in df.index:
                continue
            idx = df.index.get_loc(dt)
            if idx < WINDOW_SIZE:
                continue  # Not enough data for features
            window_df = df.iloc[idx-WINDOW_SIZE+1:idx+1].copy()
            try:
                feature_df = add_technical_indicators(window_df)
            except Exception as e:
                if log_count < 1000:
                    print(f"{dt} {symbol} Feature error: {e}")
                    log_count += 1
                continue
            missing = [f for f in selected_features if f not in feature_df.columns]
            if missing:
                if log_count < 1000:
                    print(f"{dt} {symbol} Missing features: {missing}")
                    log_count += 1
                continue
            X_current = feature_df[selected_features].iloc[[-1]]
            try:
                pred, proba, *_ = ensemble.predict_ensemble(X_current)
                probability = proba[0]
            except Exception as e:
                if log_count < 1000:
                    print(f"{dt} {symbol} Model error: {e}")
                    log_count += 1
                continue
            action = 'HOLD'
            if model_portfolio.holdings[symbol] == 0 and probability > BUY_THRESHOLD and model_portfolio.cash >= price:
                action = 'BUY'
            elif model_portfolio.holdings[symbol] > 0 and probability < SELL_THRESHOLD:
                action = 'SELL'
            if log_count < 1000:
                print(f"{dt} {symbol} prob={probability:.3f} action={action}")
                log_count += 1
            # Buy
            if action == 'BUY':
                shares = int(model_portfolio.cash / price)
                if shares > 0:
                    model_portfolio.cash -= shares * price
                    model_portfolio.holdings[symbol] += shares
                    model_portfolio.log_trade(dt, symbol, 'BUY', price, shares)
            # Sell
            elif action == 'SELL':
                shares = model_portfolio.holdings[symbol]
                model_portfolio.cash += shares * price
                model_portfolio.log_trade(dt, symbol, 'SELL', price, shares)
                model_portfolio.holdings[symbol] = 0
        model_portfolio.record(dt, prices)
    return base_portfolio, model_portfolio

print("Running both bots on historical data...")
base_portfolio, model_portfolio = run_bots(bar_dict, all_indices)

# 4. Save logs
pd.DataFrame(base_portfolio.trades).to_csv('base_bot_trades.csv', index=False)
pd.DataFrame(base_portfolio.history).to_csv('base_bot_portfolio.csv', index=False)
pd.DataFrame(model_portfolio.trades).to_csv('model_bot_trades.csv', index=False)
pd.DataFrame(model_portfolio.history).to_csv('model_bot_portfolio.csv', index=False)

# 5. Plot results
base_df = pd.DataFrame(base_portfolio.history)
model_df = pd.DataFrame(model_portfolio.history)
plt.figure(figsize=(12,6))
plt.plot(base_df['datetime'], base_df['portfolio_value'], label='Base Bot')
plt.plot(model_df['datetime'], model_df['portfolio_value'], label='Model Bot')
plt.xlabel('Time')
plt.ylabel('Portfolio Value ($)')
plt.title('Portfolio Value Over Time: Base Bot vs Model Bot')
plt.legend()
plt.tight_layout()
plt.savefig('bot_comparison.png')
plt.show()

# Print summary stats
def print_stats(df, name):
    print(f"\n{name}:")
    print(f"  Final Value: ${df['portfolio_value'].iloc[-1]:.2f}")
    returns = df['portfolio_value'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252*6.5*60) if returns.std() > 0 else 0
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {((df['portfolio_value'] / df['portfolio_value'].cummax()) - 1).min():.2%}")

print_stats(base_df, 'Base Bot')
print_stats(model_df, 'Model Bot') 