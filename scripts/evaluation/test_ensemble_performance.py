import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import sys
import os

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

warnings.filterwarnings('ignore')

def test_ensemble_walk_forward(model_file, price_file, options_file=None, use_comprehensive=False,
                              train_window=120, test_window=30, step_size=10):
    """Test ensemble model performance with walk-forward validation."""
    
    print("Loading ensemble model...")
    model_data = joblib.load(model_file)
    ensemble = model_data['ensemble']
    selected_features = model_data['selected_features']
    
    print(f"Ensemble model loaded with {len(ensemble.models)} models")
    print(f"Selected features: {len(selected_features)}")
    
    # Load and prepare data
    print("Loading and preparing data...")
    X, y, feature_names = load_and_engineer_features(
        price_path=price_file,
        options_path=options_file,
        use_comprehensive_options=use_comprehensive
    )
    
    # Load price data for returns calculation
    price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    price_df['returns'] = price_df['Close'].pct_change()
    
    print(f"Data shape: {X.shape}")
    print(f"Date range: {X.index[0]} to {X.index[-1]}")
    
    # Run walk-forward testing
    results = []
    
    for i in range(0, len(X) - train_window - test_window + 1, step_size):
        # Split data
        train_start = i
        train_end = i + train_window
        test_start = train_end
        test_end = test_start + test_window
        
        train_data = X.iloc[train_start:train_end]
        train_targets = y.iloc[train_start:train_end]
        test_data = X.iloc[test_start:test_end]
        test_targets = y.iloc[test_start:test_end]
        
        if len(train_data) < 10 or len(test_data) < 5:
            continue
        
        # Get ensemble predictions
        try:
            pred, proba, individual_preds, individual_probas = ensemble.predict_ensemble(test_data)
            accuracy = np.mean(pred == test_targets.values)
        except Exception as e:
            print(f"Error in period {len(results)+1}: {e}")
            continue
        
        # Calculate returns
        test_dates = test_data.index
        test_returns = price_df.loc[test_dates, 'returns'].values
        
        strategy_returns = []
        for j, pred_val in enumerate(pred):
            if pred_val == 1:  # Buy signal
                strategy_returns.append(test_returns[j])
            else:  # Hold signal
                strategy_returns.append(0)
        
        strategy_return = np.sum(strategy_returns) * 100
        buy_hold_return = np.sum(test_returns) * 100
        excess_return = strategy_return - buy_hold_return
        
        # Calculate Sharpe ratio
        if len(strategy_returns) > 1:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        else:
            sharpe = 0
        
        # Win rate
        winning_periods = sum(1 for r in strategy_returns if r > 0)
        total_periods = len([r for r in strategy_returns if r != 0])
        win_rate = (winning_periods / total_periods * 100) if total_periods > 0 else 0
        
        results.append({
            'period': len(results) + 1,
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'test_accuracy': accuracy,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate
        })
        
        print(f"Period {len(results)}: Test Acc={accuracy:.3f}, Strategy={strategy_return:.2f}%, B&H={buy_hold_return:.2f}%, Excess={excess_return:.2f}%")
    
    return results

def analyze_results(results):
    """Analyze walk-forward results."""
    if not results:
        print("No results to analyze.")
        return
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL WALK-FORWARD RESULTS")
    print("="*60)
    
    # Accuracy analysis
    print(f"\nAccuracy Analysis:")
    print(f"  Average Test Accuracy: {results_df['test_accuracy'].mean():.3f}")
    print(f"  Accuracy Std Dev: {results_df['test_accuracy'].std():.3f}")
    print(f"  Best Accuracy: {results_df['test_accuracy'].max():.3f}")
    print(f"  Worst Accuracy: {results_df['test_accuracy'].min():.3f}")
    
    # Trading performance
    print(f"\nTrading Performance:")
    print(f"  Average Strategy Return: {results_df['strategy_return'].mean():.2f}%")
    print(f"  Average Buy & Hold Return: {results_df['buy_hold_return'].mean():.2f}%")
    print(f"  Average Excess Return: {results_df['excess_return'].mean():.2f}%")
    print(f"  Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.3f}")
    print(f"  Average Win Rate: {results_df['win_rate'].mean():.1f}%")
    
    # Consistency analysis
    print(f"\nConsistency Analysis:")
    print(f"  Periods with Positive Returns: {(results_df['strategy_return'] > 0).sum()}")
    print(f"  Periods Outperforming Buy & Hold: {(results_df['excess_return'] > 0).sum()}")
    print(f"  Win Rate Consistency: {(results_df['win_rate'] > 50).sum()}/{len(results_df)} periods")
    
    # Risk analysis
    print(f"\nRisk Analysis:")
    print(f"  Worst Period Return: {results_df['strategy_return'].min():.2f}%")
    print(f"  Best Period Return: {results_df['strategy_return'].max():.2f}%")
    print(f"  Return Volatility: {results_df['strategy_return'].std():.2f}%")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Test ensemble model with walk-forward validation")
    parser.add_argument("model_file", help="Path to the ensemble model file")
    parser.add_argument("price_file", help="Path to the price data CSV file")
    parser.add_argument("--options", help="Path to the options data CSV file")
    parser.add_argument("--use-comprehensive", action='store_true',
                        help="Use comprehensive options features")
    parser.add_argument("--train-window", type=int, default=120,
                        help="Training window size in days")
    parser.add_argument("--test-window", type=int, default=30,
                        help="Testing window size in days")
    parser.add_argument("--step-size", type=int, default=10,
                        help="Step size between periods")
    parser.add_argument("--output", default="ensemble_walk_forward_results.csv",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    try:
        # Run walk-forward testing
        results = test_ensemble_walk_forward(
            model_file=args.model_file,
            price_file=args.price_file,
            options_file=args.options,
            use_comprehensive=args.use_comprehensive,
            train_window=args.train_window,
            test_window=args.test_window,
            step_size=args.step_size
        )
        
        # Analyze results
        results_df = analyze_results(results)
        
        # Save results
        if results_df is not None:
            results_df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 