import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import sys
import os
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. Using alternative models.")

from enhanced_features import load_and_engineer_features

warnings.filterwarnings('ignore')

def download_stock_data(symbols, start_date='2022-01-01', end_date='2024-12-31'):
    """Download data for multiple stocks."""
    print(f"Downloading data for {len(symbols)} stocks...")
    
    stock_data = {}
    for symbol in symbols:
        try:
            print(f"Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if len(data) > 100:  # Ensure sufficient data
                # Save to CSV for consistency
                filename = f"{symbol.lower()}_data.csv"
                data.to_csv(filename)
                stock_data[symbol] = filename
                print(f"  {symbol}: {len(data)} days of data")
            else:
                print(f"  {symbol}: Insufficient data ({len(data)} days)")
                
        except Exception as e:
            print(f"  Error downloading {symbol}: {e}")
    
    return stock_data

def prepare_multi_stock_features(stock_files, options_file=None, use_comprehensive=False):
    """Prepare features for multiple stocks."""
    print("Preparing multi-stock features...")
    
    all_features = []
    all_targets = []
    stock_labels = []
    
    for symbol, file_path in stock_files.items():
        try:
            print(f"Processing {symbol}...")
            
            # Check if macro-enriched file exists
            macro_file = file_path.replace('_data.csv', '_data_full_macro.csv')
            if os.path.exists(macro_file):
                print(f"  Using macro-enriched file: {macro_file}")
                actual_file = macro_file
            else:
                print(f"  Using regular file: {file_path}")
                actual_file = file_path
            
            # Load and engineer features
            X, y, feature_names = load_and_engineer_features(
                price_path=actual_file,
                options_path=options_file,
                use_comprehensive_options=use_comprehensive
            )
            
            if len(X) > 50:  # Ensure sufficient data
                # Add stock identifier
                X['stock_symbol'] = symbol
                
                all_features.append(X)
                all_targets.append(y)
                stock_labels.extend([symbol] * len(X))
                
                print(f"  {symbol}: {len(X)} samples, {len(feature_names)} features")
            else:
                print(f"  {symbol}: Insufficient samples ({len(X)})")
                
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
    
    if not all_features:
        raise ValueError("No valid stock data found")
    
    # Combine all stocks
    combined_features = pd.concat(all_features, axis=0, ignore_index=False)
    combined_targets = pd.concat(all_targets, axis=0, ignore_index=False)
    
    print(f"Combined dataset: {len(combined_features)} samples from {len(stock_files)} stocks")
    
    return combined_features, combined_targets, stock_labels, feature_names

def create_regularized_models():
    """Create ensemble models with regularization to reduce overfitting."""
    models = {}
    
    # 1. Random Forest with regularization
    models['random_forest'] = RandomForestClassifier(
        n_estimators=100,  # Reduced from 200
        max_depth=6,       # Reduced from 10
        min_samples_split=20,  # Increased from 10
        min_samples_leaf=10,   # Increased from 5
        max_features='sqrt',   # Limit features per split
        random_state=42
    )
    
    # 2. Gradient Boosting with regularization
    models['gradient_boosting'] = GradientBoostingClassifier(
        n_estimators=100,  # Reduced from 150
        learning_rate=0.05, # Reduced from 0.1
        max_depth=4,       # Reduced from 6
        subsample=0.8,     # Add subsampling
        random_state=42
    )
    
    # 3. XGBoost with regularization
    if XGBOOST_AVAILABLE:
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,  # Reduced from 200
            max_depth=4,       # Reduced from 6
            learning_rate=0.05, # Reduced from 0.1
            subsample=0.8,     # Subsampling
            colsample_bytree=0.8, # Feature subsampling
            reg_alpha=0.1,     # L1 regularization
            reg_lambda=0.1,    # L2 regularization
            random_state=42,
            verbosity=0
        )
    
    return models

def train_multi_stock_ensemble(stock_files, options_file=None, use_comprehensive=False, n_features=30):
    """Train multi-stock ensemble with overfitting reduction."""
    print("Training multi-stock ensemble model...")
    
    # Prepare multi-stock data
    X, y, stock_labels, feature_names = prepare_multi_stock_features(
        stock_files, options_file, use_comprehensive
    )
    
    # Feature selection
    print(f"Selecting top {n_features} features...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X_numeric, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = X_numeric.columns[selected_indices].tolist()
    
    print(f"Selected features: {len(selected_features)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Create and train models with cross-validation
    models = create_regularized_models()
    cv_scores = {}
    trained_models = {}
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    for name, model in models.items():
        print(f"\nTraining {name} with cross-validation...")
        
        # Cross-validation
        cv_scores[name] = cross_val_score(
            model, X_scaled, y, 
            cv=tscv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"  CV Accuracy: {cv_scores[name].mean():.4f} (+/- {cv_scores[name].std() * 2:.4f})")
        
        # Train final model
        model.fit(X_scaled, y)
        trained_models[name] = model
    
    # Evaluate by stock
    print("\nEvaluating performance by stock...")
    stock_performance = {}
    
    unique_stocks = list(set(stock_labels))
    for stock in unique_stocks:
        stock_mask = [s == stock for s in stock_labels]
        X_stock = X_scaled[stock_mask]
        y_stock = y[stock_mask]
        
        if len(X_stock) > 10:
            # Get ensemble predictions
            predictions = []
            for name, model in trained_models.items():
                pred = model.predict(X_stock)
                predictions.append(pred)
            
            # Ensemble prediction
            ensemble_pred = np.mean(predictions, axis=0) > 0.5
            accuracy = accuracy_score(y_stock, ensemble_pred)
            
            stock_performance[stock] = {
                'accuracy': accuracy,
                'samples': len(X_stock)
            }
            
            print(f"  {stock}: {accuracy:.4f} ({len(X_stock)} samples)")
    
    return {
        'models': trained_models,
        'scaler': scaler,
        'selected_features': selected_features,
        'cv_scores': cv_scores,
        'stock_performance': stock_performance,
        'feature_names': feature_names
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-stock ensemble training with overfitting reduction")
    parser.add_argument("--symbols", nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                        help="Stock symbols to train on")
    parser.add_argument("--options", help="Options data CSV file (optional)")
    parser.add_argument("--model-out", default="multi_stock_ensemble_model.pkl", 
                        help="Output model file")
    parser.add_argument("--use-comprehensive", action='store_true',
                        help="Use comprehensive options features")
    parser.add_argument("--n-features", type=int, default=30, 
                        help="Number of features to select")
    parser.add_argument("--start-date", default='2022-01-01',
                        help="Start date for data download")
    parser.add_argument("--end-date", default='2024-12-31',
                        help="End date for data download")
    parser.add_argument("--download-data", action='store_true',
                        help="Download fresh data for symbols")
    
    args = parser.parse_args()
    
    try:
        # Download data if requested
        if args.download_data:
            stock_files = download_stock_data(
                args.symbols, args.start_date, args.end_date
            )
        else:
            # Use existing files if available
            stock_files = {}
            for symbol in args.symbols:
                filename = f"{symbol.lower()}_data.csv"
                if os.path.exists(filename):
                    stock_files[symbol] = filename
                else:
                    print(f"Warning: {filename} not found. Run with --download-data to get fresh data.")
            
            if not stock_files:
                print("No stock data files found. Downloading fresh data...")
                stock_files = download_stock_data(
                    args.symbols, args.start_date, args.end_date
                )
        
        if not stock_files:
            raise ValueError("No stock data available")
        
        # Train ensemble
        results = train_multi_stock_ensemble(
            stock_files, args.options, args.use_comprehensive, args.n_features
        )
        
        # Analyze results
        print("\n" + "="*60)
        print("MULTI-STOCK ENSEMBLE RESULTS")
        print("="*60)
        
        print(f"\nCross-Validation Results:")
        for name, scores in results['cv_scores'].items():
            print(f"  {name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        if results['stock_performance']:
            accuracies = [perf['accuracy'] for perf in results['stock_performance'].values()]
            print(f"\nStock Performance Analysis:")
            print(f"  Average Stock Accuracy: {np.mean(accuracies):.4f}")
            print(f"  Stock Accuracy Std Dev: {np.std(accuracies):.4f}")
            print(f"  Best Stock: {max(results['stock_performance'].items(), key=lambda x: x[1]['accuracy'])[0]}")
            print(f"  Worst Stock: {min(results['stock_performance'].items(), key=lambda x: x[1]['accuracy'])[0]}")
        
        # Save model
        model_data = {
            'models': results['models'],
            'scaler': results['scaler'],
            'selected_features': results['selected_features'],
            'cv_scores': results['cv_scores'],
            'stock_performance': results['stock_performance'],
            'feature_names': results['feature_names'],
            'training_date': datetime.now(),
            'symbols_trained': list(stock_files.keys()),
            'model_config': {
                'n_features_select': args.n_features,
                'use_comprehensive_options': args.use_comprehensive
            }
        }
        
        joblib.dump(model_data, args.model_out)
        print(f"\nMulti-stock ensemble model saved to: {args.model_out}")
        print(f"Trained on {len(stock_files)} stocks: {list(stock_files.keys())}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 