import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import warnings

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

warnings.filterwarnings('ignore')

def load_expanded_model(model_file="expanded_multi_stock_model.pkl"):
    """Load the expanded multi-stock model."""
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found!")
    
    print(f"Loading expanded multi-stock model: {model_file}")
    model_data = joblib.load(model_file)
    return model_data

def prepare_walk_forward_data(stock_symbols, test_start_date='2024-01-01', test_end_date='2024-12-31'):
    """Prepare data for walk-forward testing."""
    print(f"Preparing walk-forward data for {len(stock_symbols)} stocks...")
    
    all_features = []
    all_targets = []
    stock_labels = []
    successful_stocks = []
    
    for symbol in stock_symbols:
        try:
            filename = f"{symbol.lower()}_data.csv"
            if not os.path.exists(filename):
                print(f"  ❌ {symbol}: Data file not found")
                continue
            
            # Load and engineer features
            X, y, feature_names = load_and_engineer_features(
                price_path=filename,
                options_path=None,  # No options for walk-forward
                use_comprehensive_options=False
            )
            
            if len(X) > 50:
                # Filter for test period
                X['stock_symbol'] = symbol
                all_features.append(X)
                all_targets.append(y)
                stock_labels.extend([symbol] * len(X))
                successful_stocks.append(symbol)
                print(f"  ✅ {symbol}: {len(X)} samples")
            else:
                print(f"  ❌ {symbol}: Insufficient samples ({len(X)})")
                
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
    
    if not all_features:
        raise ValueError("No valid stock data found")
    
    # Combine all stocks
    combined_features = pd.concat(all_features, axis=0, ignore_index=False)
    combined_targets = pd.concat(all_targets, axis=0, ignore_index=False)
    
    print(f"Combined dataset: {len(combined_features)} samples from {len(successful_stocks)} stocks")
    return combined_features, combined_targets, stock_labels, successful_stocks

def walk_forward_test(model_data, X, y, stock_labels, window_size=252, step_size=63):
    """Perform walk-forward testing on the expanded model."""
    print(f"Starting walk-forward testing...")
    print(f"Window size: {window_size} days, Step size: {step_size} days")
    
    # Get model components
    models = model_data['models']
    scaler = model_data['scaler']
    selected_features = model_data['selected_features']
    
    # Sort by date
    X_sorted = X.sort_index()
    y_sorted = y.sort_index()
    
    # Initialize results
    results = []
    total_samples = len(X_sorted)
    
    # Walk-forward testing
    for start_idx in range(0, total_samples - window_size, step_size):
        end_idx = start_idx + window_size
        
        # Training window
        X_train = X_sorted.iloc[start_idx:end_idx]
        y_train = y_sorted.iloc[start_idx:end_idx]
        
        # Test window (next step_size days)
        test_end = min(end_idx + step_size, total_samples)
        X_test = X_sorted.iloc[end_idx:test_end]
        y_test = y_sorted.iloc[end_idx:test_end]
        
        if len(X_test) < 10:  # Need minimum test samples
            continue
        
        try:
            # Prepare features - ensure we have the selected features
            available_features = [f for f in selected_features if f in X_train.columns]
            if len(available_features) != len(selected_features):
                print(f"  Warning: Missing features. Expected {len(selected_features)}, got {len(available_features)}")
                continue
            
            X_train_numeric = X_train[selected_features]
            X_test_numeric = X_test[selected_features]
            
            # Scale features
            X_train_scaled = scaler.transform(X_train_numeric)
            X_test_scaled = scaler.transform(X_test_numeric)
            
            # Get ensemble predictions
            predictions = []
            for name, model in models.items():
                pred = model.predict(X_test_scaled)
                predictions.append(pred)
            
            # Ensemble prediction
            ensemble_pred = np.mean(predictions, axis=0) > 0.5
            
            # Calculate metrics
            accuracy = np.mean(ensemble_pred == y_test)
            
            # Calculate returns (simplified)
            test_returns = []
            for i, (idx, row) in enumerate(X_test.iterrows()):
                if i < len(ensemble_pred):
                    if ensemble_pred[i]:
                        # Buy signal - assume 1% return if correct, -1% if wrong
                        if ensemble_pred[i] == y_test.iloc[i]:
                            test_returns.append(0.01)
                        else:
                            test_returns.append(-0.01)
                    else:
                        # Hold - assume 0% return
                        test_returns.append(0.0)
            
            avg_return = np.mean(test_returns) if test_returns else 0.0
            
            # Store results
            window_result = {
                'start_date': X_train.index[0],
                'end_date': X_train.index[-1],
                'test_start': X_test.index[0],
                'test_end': X_test.index[-1],
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': accuracy,
                'avg_return': avg_return,
                'predictions': ensemble_pred,
                'actual': y_test.values
            }
            
            results.append(window_result)
            
            print(f"  Window {len(results)}: {X_train.index[0].date()} - {X_test.index[-1].date()}, "
                  f"Accuracy: {accuracy:.4f}, Return: {avg_return:.4f}")
        
        except Exception as e:
            print(f"  Error in window {len(results)+1}: {e}")
            continue
    
    return results

def analyze_walk_forward_results(results):
    """Analyze walk-forward testing results."""
    print("\n" + "="*60)
    print("WALK-FORWARD TESTING RESULTS")
    print("="*60)
    
    if not results:
        print("No results to analyze")
        return
    
    # Extract metrics
    accuracies = [r['accuracy'] for r in results]
    returns = [r['avg_return'] for r in results]
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total Windows: {len(results)}")
    print(f"  Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"  Accuracy Std Dev: {np.std(accuracies):.4f}")
    print(f"  Average Return: {np.mean(returns):.4f}")
    print(f"  Return Std Dev: {np.std(returns):.4f}")
    
    # Performance over time
    print(f"\n📈 Performance Trends:")
    early_acc = np.mean(accuracies[:len(accuracies)//3])
    mid_acc = np.mean(accuracies[len(accuracies)//3:2*len(accuracies)//3])
    late_acc = np.mean(accuracies[2*len(accuracies)//3:])
    
    print(f"  Early Period Accuracy: {early_acc:.4f}")
    print(f"  Mid Period Accuracy: {mid_acc:.4f}")
    print(f"  Late Period Accuracy: {late_acc:.4f}")
    
    # Best and worst windows
    best_window = max(results, key=lambda x: x['accuracy'])
    worst_window = min(results, key=lambda x: x['accuracy'])
    
    print(f"\n🏆 Best Window:")
    print(f"  Period: {best_window['test_start'].date()} - {best_window['test_end'].date()}")
    print(f"  Accuracy: {best_window['accuracy']:.4f}")
    print(f"  Return: {best_window['avg_return']:.4f}")
    
    print(f"\n📉 Worst Window:")
    print(f"  Period: {worst_window['test_start'].date()} - {worst_window['test_end'].date()}")
    print(f"  Accuracy: {worst_window['accuracy']:.4f}")
    print(f"  Return: {worst_window['avg_return']:.4f}")
    
    # Consistency analysis
    above_50 = sum(1 for acc in accuracies if acc > 0.5)
    print(f"\n📊 Consistency Analysis:")
    print(f"  Windows above 50% accuracy: {above_50}/{len(accuracies)} ({above_50/len(accuracies)*100:.1f}%)")
    print(f"  Windows above 55% accuracy: {sum(1 for acc in accuracies if acc > 0.55)}/{len(accuracies)}")
    print(f"  Windows above 60% accuracy: {sum(1 for acc in accuracies if acc > 0.6)}/{len(accuracies)}")
    
    return {
        'accuracies': accuracies,
        'returns': returns,
        'results': results
    }

def main():
    """Main function for walk-forward testing."""
    try:
        # Load expanded model
        model_data = load_expanded_model()
        
        # Get successful stocks from model
        successful_stocks = model_data.get('successful_stocks', [])
        if not successful_stocks:
            # Fallback to common stocks
            successful_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        print(f"Using {len(successful_stocks)} stocks for walk-forward testing")
        
        # Prepare data
        X, y, stock_labels, final_stocks = prepare_walk_forward_data(successful_stocks)
        
        # Perform walk-forward testing
        results = walk_forward_test(model_data, X, y, stock_labels)
        
        # Analyze results
        analysis = analyze_walk_forward_results(results)
        
        # Save results
        results_df = pd.DataFrame([
            {
                'start_date': r['start_date'],
                'end_date': r['end_date'],
                'test_start': r['test_start'],
                'test_end': r['test_end'],
                'train_samples': r['train_samples'],
                'test_samples': r['test_samples'],
                'accuracy': r['accuracy'],
                'avg_return': r['avg_return']
            }
            for r in results
        ])
        
        results_df.to_csv('walk_forward_expanded_results.csv', index=False)
        print(f"\n✅ Walk-forward results saved to: walk_forward_expanded_results.csv")
        
    except Exception as e:
        print(f"❌ Error in walk-forward testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 