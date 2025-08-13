import joblib
import pandas as pd
import numpy as np
from scripts.enhanced_features import load_and_engineer_features

# Load the ensemble model
print("Loading ensemble model...")
model_data = joblib.load('new_advanced_ensemble_model.pkl')
ensemble = model_data['ensemble']
selected_features = model_data['selected_features']

print(f"Ensemble model loaded with {len(ensemble.models)} models")
print(f"Selected features: {len(selected_features)}")

# Load and prepare data
print("Loading data...")
X, y, feature_names = load_and_engineer_features(
    price_path='apple_recent2y.csv',
    options_path='comprehensive_options_30d.csv',
    use_comprehensive_options=True
)

# Load price data
price_df = pd.read_csv('apple_recent2y.csv', index_col=0, parse_dates=True)
price_df['returns'] = price_df['Close'].pct_change()

print(f"Data shape: {X.shape}")

# Simple backtest
print("\nRunning simple backtest...")
portfolio_value = 100000
cash = 100000
shares = 0
trades = []
portfolio_history = []

for i in range(len(X)):
    current_date = X.index[i]
    current_price = price_df.loc[current_date, 'Close']
    
    # Get model prediction
    X_current = X.iloc[i:i+1][selected_features]
    pred, proba, individual_preds, individual_probas = ensemble.predict_ensemble(X_current)
    probability = proba[0]
    
    # Simple trading logic
    if shares == 0 and probability > 0.65:  # Buy signal
        shares_to_buy = int((cash * 0.1) / current_price)  # Use 10% of cash
        if shares_to_buy > 0:
            cost = shares_to_buy * current_price
            cash -= cost
            shares += shares_to_buy
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': current_price,
                'shares': shares_to_buy,
                'probability': probability
            })
    
    elif shares > 0 and probability < 0.35:  # Sell signal
        proceeds = shares * current_price
        cash += proceeds
        trades.append({
            'date': current_date,
            'action': 'SELL',
            'price': current_price,
            'shares': shares,
            'probability': probability
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
        'probability': 0
    })

# Calculate results
initial_value = 100000
final_value = cash
total_return = (final_value / initial_value - 1) * 100
buy_hold_return = (price_df['Close'].iloc[-1] / price_df['Close'].iloc[0] - 1) * 100

print(f"\n=== ENSEMBLE MODEL BACKTEST RESULTS ===")
print(f"Initial Capital: ${initial_value:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Excess Return: {total_return - buy_hold_return:.2f}%")
print(f"Number of Trades: {len(trades)}")

# Calculate accuracy
predictions = []
for i in range(len(X)):
    X_current = X.iloc[i:i+1][selected_features]
    pred, proba, individual_preds, individual_probas = ensemble.predict_ensemble(X_current)
    predictions.append(pred[0])

accuracy = np.mean(predictions == y.values)
print(f"Model Accuracy: {accuracy:.4f}")

# Show some trade details
if trades:
    print(f"\nSample Trades:")
    for i, trade in enumerate(trades[:5]):
        print(f"  {i+1}. {trade['date'].strftime('%Y-%m-%d')} - {trade['action']} {trade['shares']} shares at ${trade['price']:.2f} (prob: {trade['probability']:.3f})") 