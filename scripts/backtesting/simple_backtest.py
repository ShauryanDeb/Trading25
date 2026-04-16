import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

def simple_backtest(model_path, price_file, options_file=None, use_comprehensive=False, threshold=0.55):
    """Simple backtest that shows predictions and simulates trading."""
    
    print("Loading model and data...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    
    # Use the same feature engineering logic as training
    X, y, feature_names = load_and_engineer_features(
        price_path=price_file,
        options_path=options_file,
        use_comprehensive_options=use_comprehensive
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict_proba(X)[:, 1]
    
    # Show prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean prediction: {predictions.mean():.4f}")
    print(f"  Std prediction: {predictions.std():.4f}")
    print(f"  Min prediction: {predictions.min():.4f}")
    print(f"  Max prediction: {predictions.max():.4f}")
    print(f"  Predictions > {threshold}: {np.sum(predictions > threshold)}")
    print(f"  Predictions < {1-threshold}: {np.sum(predictions < (1-threshold))}")
    
    # Show some sample predictions
    print(f"\nSample Predictions (first 10):")
    for i in range(min(10, len(predictions))):
        print(f"  Day {i+1}: {predictions[i]:.4f}")
    
    # Load price data for returns calculation
    price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    
    # Align predictions with price data
    aligned_data = pd.DataFrame({
        'Close': price_df['Close'],
        'prediction': predictions
    }, index=X.index)
    
    # Simple trading simulation
    print(f"\nRunning simple trading simulation with threshold {threshold}...")
    
    cash = 10000.0
    shares = 0
    trades = []
    
    for i in range(len(aligned_data)):
        current_price = aligned_data['Close'].iloc[i]
        pred = aligned_data['prediction'].iloc[i]
        
        # Buy signal
        if shares == 0 and pred > threshold:
            shares = cash / current_price
            cash = 0
            trades.append({
                'date': aligned_data.index[i],
                'action': 'BUY',
                'price': current_price,
                'prediction': pred,
                'portfolio_value': shares * current_price
            })
        
        # Sell signal
        elif shares > 0 and pred < (1 - threshold):
            cash = shares * current_price
            shares = 0
            trades.append({
                'date': aligned_data.index[i],
                'action': 'SELL',
                'price': current_price,
                'prediction': pred,
                'portfolio_value': cash
            })
    
    # Final portfolio value
    if shares > 0:
        final_value = shares * aligned_data['Close'].iloc[-1]
    else:
        final_value = cash
    
    # Calculate returns
    buy_hold_return = (aligned_data['Close'].iloc[-1] / aligned_data['Close'].iloc[0] - 1) * 100
    strategy_return = (final_value / 10000 - 1) * 100
    
    print(f"\n--- Simple Backtest Results ---")
    print(f"Starting Portfolio Value: $10,000.00")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Strategy Return:         {strategy_return:.2f}%")
    print(f"Buy & Hold Return:       {buy_hold_return:.2f}%")
    print(f"Total Trades:            {len(trades)}")
    
    if len(trades) > 0:
        print(f"\nSample Trades (first 5):")
        for i, trade in enumerate(trades[:5]):
            print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} @ ${trade['price']:.2f} (pred: {trade['prediction']:.3f})")
    
    print("------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Simple backtesting with prediction analysis.")
    parser.add_argument("model_path", help="Path to the saved .pkl model file.")
    parser.add_argument("price_file", help="Path to the historical price data (e.g., apple.csv).")
    parser.add_argument("--options", help="Path to the comprehensive options data CSV.")
    parser.add_argument("--use-comprehensive", action='store_true',
                        help="Signal that the model uses comprehensive options features.")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Probability threshold for entering a trade.")
    
    args = parser.parse_args()
    
    simple_backtest(
        model_path=args.model_path,
        price_file=args.price_file,
        options_file=args.options,
        use_comprehensive=args.use_comprehensive,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main() 