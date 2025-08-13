import joblib
import pandas as pd
import numpy as np
from scripts.enhanced_features import load_and_engineer_features

# Parameters
STARTING_CASH = 3500
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35
MODEL_PATH = 'new_advanced_ensemble_model.pkl'
PRICE_FILE = 'apple_recent2y.csv'
OPTIONS_FILE = 'comprehensive_options_30d.csv'
USE_COMPREHENSIVE = True
TRADES_CSV = 'virtual_trades_log.csv'

# Load the ensemble model
print("Loading ensemble model...")
model_data = joblib.load(MODEL_PATH)
ensemble = model_data['ensemble']
selected_features = model_data['selected_features']

print(f"Ensemble model loaded with {len(ensemble.models)} models")
print(f"Selected features: {len(selected_features)}")

# Load and prepare data
print("Loading data...")
X, y, feature_names = load_and_engineer_features(
    price_path=PRICE_FILE,
    options_path=OPTIONS_FILE,
    use_comprehensive_options=USE_COMPREHENSIVE
)

# Load price data
price_df = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)

print(f"Data shape: {X.shape}")

# Virtual trading simulation
print("\nRunning virtual trading simulation...")
cash = STARTING_CASH
shares = 0
trades = []
portfolio_history = []

for i in range(len(X)):
    current_date = X.index[i]
    current_price = price_df.loc[current_date, 'Close']
    X_current = X.iloc[i:i+1][selected_features]
    pred, proba, individual_preds, individual_probas = ensemble.predict_ensemble(X_current)
    probability = proba[0]

    # Buy signal
    if shares == 0 and probability > BUY_THRESHOLD:
        shares_to_buy = int(cash / current_price)
        if shares_to_buy > 0:
            cost = shares_to_buy * current_price
            cash -= cost
            shares += shares_to_buy
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': current_price,
                'shares': shares_to_buy,
                'probability': probability,
                'cash': cash
            })

    # Sell signal
    elif shares > 0 and probability < SELL_THRESHOLD:
        proceeds = shares * current_price
        cash += proceeds
        trades.append({
            'date': current_date,
            'action': 'SELL',
            'price': current_price,
            'shares': shares,
            'probability': probability,
            'cash': cash
        })
        shares = 0

    # Record portfolio value
    current_portfolio_value = cash + (shares * current_price)
    portfolio_history.append({
        'date': current_date,
        'portfolio_value': current_portfolio_value,
        'cash': cash,
        'shares': shares,
        'price': current_price,
        'probability': probability
    })

# Close any remaining position
if shares > 0:
    final_price = price_df.iloc[-1]['Close']
    final_date = price_df.index[-1]
    proceeds = shares * final_price
    cash += proceeds
    trades.append({
        'date': final_date,
        'action': 'SELL',
        'price': final_price,
        'shares': shares,
        'probability': 0,
        'cash': cash
    })
    shares = 0

# Save trades to CSV
trades_df = pd.DataFrame(trades)
trades_df.to_csv(TRADES_CSV, index=False)
print(f"\nAll trades logged to {TRADES_CSV}")

# Calculate results
initial_value = STARTING_CASH
final_value = cash
buy_hold_return = (price_df['Close'].iloc[-1] / price_df['Close'].iloc[0] - 1) * 100
strategy_return = (final_value / initial_value - 1) * 100

print(f"\n=== VIRTUAL TRADING BOT RESULTS ===")
print(f"Initial Capital: ${initial_value:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Strategy Return: {strategy_return:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Number of Trades: {len(trades)}")

# Show some trade details
if not trades_df.empty:
    print(f"\nSample Trades:")
    for i, trade in trades_df.head(5).iterrows():
        print(f"  {trade['date'][:10]} - {trade['action']} {trade['shares']} shares at ${trade['price']:.2f} (prob: {trade['probability']:.3f})") 