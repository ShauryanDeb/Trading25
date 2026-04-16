import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime, timedelta

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

def realistic_backtest(model_path, price_file, options_file=None, use_comprehensive=False, 
                      threshold=0.55, years_back=5):
    """Realistic backtest using recent data to avoid historical scaling issues."""
    
    print("Loading model and data...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    
    # Load price data and filter to recent years
    price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    
    # Calculate the cutoff date (years_back from the most recent date)
    end_date = price_df.index.max()
    start_date = end_date - timedelta(days=365 * years_back)
    
    # Filter to recent data
    recent_price_df = price_df[price_df.index >= start_date].copy()
    
    print(f"Using data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Recent data shape: {recent_price_df.shape}")
    
    # Save filtered data temporarily for feature engineering
    temp_price_file = "temp_recent_apple.csv"
    recent_price_df.to_csv(temp_price_file)
    
    try:
        # Use the same feature engineering logic as training
        X, y, feature_names = load_and_engineer_features(
            price_path=temp_price_file,
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
        
        # Align predictions with price data
        aligned_data = pd.DataFrame({
            'Close': recent_price_df['Close'],
            'prediction': predictions
        }, index=X.index)
        
        # Simple trading simulation
        print(f"\nRunning trading simulation with threshold {threshold}...")
        
        cash = 10000.0
        shares = 0
        trades = []
        portfolio_values = []
        
        for i in range(len(aligned_data)):
            current_price = aligned_data['Close'].iloc[i]
            pred = aligned_data['prediction'].iloc[i]
            
            # Calculate current portfolio value
            if shares > 0:
                current_value = shares * current_price
            else:
                current_value = cash
            portfolio_values.append(current_value)
            
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
        
        # Calculate additional metrics
        portfolio_series = pd.Series(portfolio_values, index=aligned_data.index)
        max_drawdown = calculate_max_drawdown(portfolio_series)
        
        print(f"\n--- Realistic Backtest Results ({years_back} years) ---")
        print(f"Starting Portfolio Value: $10,000.00")
        print(f"Final Portfolio Value:   ${final_value:,.2f}")
        print(f"Strategy Return:         {strategy_return:.2f}%")
        print(f"Buy & Hold Return:       {buy_hold_return:.2f}%")
        print(f"Max Drawdown:            {max_drawdown:.2f}%")
        print(f"Total Trades:            {len(trades)}")
        
        if len(trades) > 0:
            winning_trades = sum(1 for i in range(1, len(trades), 2) 
                               if i < len(trades) and trades[i]['portfolio_value'] > trades[i-1]['portfolio_value'])
            win_rate = (winning_trades / (len(trades) // 2)) * 100 if len(trades) > 1 else 0
            print(f"Win Rate:                {win_rate:.1f}%")
            
            print(f"\nSample Trades (first 5):")
            for i, trade in enumerate(trades[:5]):
                print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} @ ${trade['price']:.2f} (pred: {trade['prediction']:.3f})")
        
        print("----------------------------------------")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_price_file):
            os.remove(temp_price_file)

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from peak."""
    peak = portfolio_values.expanding().max()
    drawdown = (portfolio_values - peak) / peak * 100
    return drawdown.min()

def main():
    parser = argparse.ArgumentParser(description="Realistic backtesting with recent data.")
    parser.add_argument("model_path", help="Path to the saved .pkl model file.")
    parser.add_argument("price_file", help="Path to the historical price data (e.g., apple.csv).")
    parser.add_argument("--options", help="Path to the comprehensive options data CSV.")
    parser.add_argument("--use-comprehensive", action='store_true',
                        help="Signal that the model uses comprehensive options features.")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Probability threshold for entering a trade.")
    parser.add_argument("--years", type=int, default=5,
                        help="Number of years of recent data to use for backtesting.")
    
    args = parser.parse_args()
    
    realistic_backtest(
        model_path=args.model_path,
        price_file=args.price_file,
        options_file=args.options,
        use_comprehensive=args.use_comprehensive,
        threshold=args.threshold,
        years_back=args.years
    )

if __name__ == "__main__":
    main() 