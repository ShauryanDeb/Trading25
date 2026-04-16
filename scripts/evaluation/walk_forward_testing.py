import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Add XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost is not installed. XGBoost model type will not be available.")

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

class WalkForwardTester:
    """
    Walk-forward testing system to prevent overfitting and validate model performance.
    """
    
    def __init__(self, train_window_days=252, test_window_days=63, step_days=21):
        """
        Initialize walk-forward tester.
        
        Args:
            train_window_days: Number of days for training (default: 1 year)
            test_window_days: Number of days for testing (default: 3 months)
            step_days: Number of days to step forward (default: 1 month)
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.results = []
        
    def prepare_data(self, price_file, options_file=None, use_comprehensive=False):
        """Prepare data for walk-forward testing."""
        print("Loading and preparing data for walk-forward testing...")
        
        # Load price data
        price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
        
        # Use the same feature engineering logic as training
        X, y, feature_names = load_and_engineer_features(
            price_path=price_file,
            options_path=options_file,
            use_comprehensive_options=use_comprehensive
        )
        
        # Align all data
        aligned_data = pd.DataFrame({
            'Close': price_df['Close'],
            'Target': y
        }, index=X.index)
        
        # Add features
        for i, feature in enumerate(feature_names):
            aligned_data[feature] = X.iloc[:, i]
        
        print(f"Prepared data shape: {aligned_data.shape}")
        print(f"Date range: {aligned_data.index.min()} to {aligned_data.index.max()}")
        
        return aligned_data, feature_names
    
    def run_walk_forward_test(self, data, feature_names, model_params=None, model_type='random_forest', ensemble_model=None):
        """Run walk-forward testing with optional ensemble model."""
        results = []
        
        for i in range(0, len(data) - self.train_window_days - self.test_window_days + 1, self.step_days):
            # Split data
            train_start = i
            train_end = i + self.train_window_days
            test_start = train_end
            test_end = test_start + self.test_window_days
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(train_data) < 10 or len(test_data) < 5:
                continue
            
            # Prepare features and target
            X_train = train_data[feature_names]
            y_train = train_data['Target']
            X_test = test_data[feature_names]
            y_test = test_data['Target']
            
            # Use ensemble model if provided, otherwise train new model
            if ensemble_model is not None:
                # Use pre-trained ensemble model
                pred, proba, individual_preds, individual_probas = ensemble_model.predict_ensemble(X_test)
                y_pred = pred
            else:
                # Train new model
                if model_type == 'random_forest':
                    model = RandomForestClassifier(**model_params) if model_params else RandomForestClassifier(random_state=42)
                elif model_type == 'xgboost':
                    if not XGBOOST_AVAILABLE:
                        print("XGBoost not available, using RandomForest")
                        model = RandomForestClassifier(**model_params) if model_params else RandomForestClassifier(random_state=42)
                    else:
                        model = xgb.XGBClassifier(**model_params) if model_params else xgb.XGBClassifier(random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate returns
            test_returns = test_data['Close'].values
            strategy_returns = []
            
            for j, pred in enumerate(y_pred):
                if pred == 1:  # Buy signal
                    strategy_returns.append(test_returns[j])
                else:  # Hold signal
                    strategy_returns.append(0)
            
            strategy_return = np.sum(strategy_returns)
            buy_hold_return = np.sum(test_returns)
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
                'train_accuracy': 1.0,  # Placeholder for ensemble
                'test_accuracy': accuracy,
                'strategy_return': strategy_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': excess_return,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate
            })
            
            print(f"Period {len(results)}: Test Acc={accuracy:.3f}, Strategy={strategy_return:.3f}%, B&H={buy_hold_return:.3f}%, Excess={excess_return:.3f}%")
        
        return results
    
    def calculate_trading_returns(self, test_data, predictions, threshold=0.55):
        """Calculate trading returns for a test period."""
        cash = 10000.0
        shares = 0
        trades = []
        portfolio_values = []
        
        for i in range(len(test_data)):
            current_price = test_data['Close'].iloc[i]
            pred = predictions[i]
            
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
                    'action': 'BUY',
                    'price': current_price,
                    'portfolio_value': shares * current_price
                })
            
            # Sell signal
            elif shares > 0 and pred < (1 - threshold):
                cash = shares * current_price
                shares = 0
                trades.append({
                    'action': 'SELL',
                    'price': current_price,
                    'portfolio_value': cash
                })
        
        # Final portfolio value
        if shares > 0:
            final_value = shares * test_data['Close'].iloc[-1]
        else:
            final_value = cash
        
        # Calculate metrics
        strategy_return = (final_value / 10000 - 1) * 100
        buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
        
        # Calculate max drawdown
        portfolio_series = pd.Series(portfolio_values)
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified)
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate win rate
        if len(trades) >= 2:
            winning_trades = sum(1 for i in range(1, len(trades), 2) 
                               if i < len(trades) and trades[i]['portfolio_value'] > trades[i-1]['portfolio_value'])
            win_rate = (winning_trades / (len(trades) // 2)) * 100 if len(trades) > 1 else 0
        else:
            win_rate = 0
        
        return {
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(trades),
            'win_rate': win_rate
        }
    
    def analyze_results(self):
        """Analyze walk-forward testing results."""
        if not self.results:
            print("No results to analyze. Run walk-forward testing first.")
            return
        
        results_df = pd.DataFrame(self.results)
        
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD TESTING ANALYSIS")
        print(f"{'='*60}")
        
        # Overall performance
        print(f"\nOverall Performance Summary:")
        print(f"  Total Periods: {len(results_df)}")
        print(f"  Average Test Accuracy: {results_df['test_accuracy'].mean():.3f}")
        print(f"  Average Train Accuracy: {results_df['train_accuracy'].mean():.3f}")
        print(f"  Average Accuracy Drop: {results_df['accuracy_drop'].mean():.3f}")
        
        # Overfitting analysis
        print(f"\nOverfitting Analysis:")
        print(f"  Periods with Accuracy Drop > 0.1: {(results_df['accuracy_drop'] > 0.1).sum()}")
        print(f"  Periods with Accuracy Drop > 0.05: {(results_df['accuracy_drop'] > 0.05).sum()}")
        print(f"  Average Accuracy Drop: {results_df['accuracy_drop'].mean():.3f}")
        
        # Trading performance
        print(f"\nTrading Performance:")
        print(f"  Average Strategy Return: {results_df['strategy_return'].mean():.2f}%")
        print(f"  Average Buy & Hold Return: {results_df['buy_hold_return'].mean():.2f}%")
        print(f"  Average Excess Return: {results_df['excess_return'].mean():.2f}%")
        print(f"  Average Max Drawdown: {results_df['max_drawdown'].mean():.2f}%")
        print(f"  Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.3f}")
        print(f"  Average Win Rate: {results_df['win_rate'].mean():.1f}%")
        print(f"  Average Number of Trades: {results_df['num_trades'].mean():.1f}")
        
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
        print(f"  Worst Max Drawdown: {results_df['max_drawdown'].min():.2f}%")
        
        return results_df
    
    def save_results(self, output_file):
        """Save results to CSV file."""
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Walk-forward testing to prevent overfitting.")
    parser.add_argument("price_file", help="Path to the historical price data (e.g., apple.csv).")
    parser.add_argument("--options", help="Path to the comprehensive options data CSV.")
    parser.add_argument("--use-comprehensive", action='store_true',
                        help="Signal that the model uses comprehensive options features.")
    parser.add_argument("--train-window", type=int, default=252,
                        help="Training window in days (default: 252 = 1 year).")
    parser.add_argument("--test-window", type=int, default=63,
                        help="Testing window in days (default: 63 = 3 months).")
    parser.add_argument("--step-size", type=int, default=21,
                        help="Step size in days (default: 21 = 1 month).")
    parser.add_argument("--output", default="walk_forward_results.csv",
                        help="Output CSV file for results.")
    parser.add_argument("--model-type", choices=['random_forest', 'xgboost'], default='random_forest',
                        help="Type of model to use")
    parser.add_argument("--ensemble-model", help="Path to pre-trained ensemble model file")
    
    args = parser.parse_args()
    
    # Initialize walk-forward tester
    tester = WalkForwardTester(
        train_window_days=args.train_window,
        test_window_days=args.test_window,
        step_days=args.step_size
    )
    
    # Prepare data
    aligned_data, feature_names = tester.prepare_data(
        price_file=args.price_file,
        options_file=args.options,
        use_comprehensive=args.use_comprehensive
    )
    
    # Load ensemble model if specified
    ensemble_model = None
    if args.ensemble_model:
        print(f"Loading ensemble model from {args.ensemble_model}...")
        try:
            import joblib
            model_data = joblib.load(args.ensemble_model)
            ensemble_model = model_data['ensemble']
            print(f"Ensemble model loaded with {len(ensemble_model.models)} models")
        except Exception as e:
            print(f"Error loading ensemble model: {e}")
            print("Falling back to individual model training")
    
    # Run walk-forward testing
    results = tester.run_walk_forward_test(
        data=aligned_data,
        feature_names=feature_names,
        model_type=args.model_type,
        ensemble_model=ensemble_model
    )
    
    # Analyze results
    results_df = tester.analyze_results()
    
    # Save results
    tester.save_results(args.output)

if __name__ == "__main__":
    main() 