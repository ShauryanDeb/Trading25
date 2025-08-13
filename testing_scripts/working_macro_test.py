import os
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

warnings.filterwarnings('ignore')

def get_macro_files():
    """Get macro-enriched stock files."""
    files = [f for f in os.listdir('.') if f.endswith('_data_full_macro_fixed.csv')]
    symbols = [f.replace('_data_full_macro_fixed.csv', '').upper() for f in files]
    return dict(zip(symbols, files))

def prepare_features_manual(stock_files):
    """Prepare features manually to avoid NaN issues."""
    print("Preparing features manually...")
    
    all_features = []
    all_targets = []
    stock_labels = []
    
    for symbol, file_path in stock_files.items():
        try:
            print(f"Processing {symbol}...")
            
            # Load data
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Remove problematic columns
            if 'VIX' in df.columns:
                df = df.drop('VIX', axis=1)
            if 'SP500' in df.columns:
                df = df.drop('SP500', axis=1)
            
            # Basic technical indicators
            df['MA_20'] = df['Close'].rolling(20).mean()
            df['MA_50'] = df['Close'].rolling(50).mean()
            df['RSI_14'] = 100 - (100 / (1 + df['Close'].rolling(14).apply(lambda x: (x.diff() > 0).sum() / (x.diff() < 0).sum())))
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['ATR_14'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
            
            # Macro features (forward fill to handle NaN)
            macro_features = ['FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI', 
                             'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 
                             'IndustrialProduction', 'RetailSales', 'ConsumerConfidence']
            
            for feature in macro_features:
                if feature in df.columns:
                    df[feature] = df[feature].ffill().bfill()
            
            # Create target
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            # Select features
            feature_cols = ['MA_20', 'MA_50', 'RSI_14', 'MACD', 'ATR_14'] + macro_features
            available_features = [col for col in feature_cols if col in df.columns]
            
            # Drop NaN and get valid data
            valid_data = df[available_features + ['Target']].dropna()
            
            if len(valid_data) > 50:
                X = valid_data[available_features]
                y = valid_data['Target']
                
                # Add stock identifier
                X['stock_symbol'] = symbol
                
                all_features.append(X)
                all_targets.append(y)
                stock_labels.extend([symbol] * len(X))
                
                print(f"  {symbol}: {len(X)} samples, {len(available_features)} features")
                
                # Count macro features
                macro_count = len([f for f in available_features if f in macro_features])
                print(f"    Macro features: {macro_count}")
                
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
    
    if not all_features:
        raise ValueError("No valid stock data found")
    
    # Combine all stocks
    combined_features = pd.concat(all_features, axis=0, ignore_index=False)
    combined_targets = pd.concat(all_targets, axis=0, ignore_index=False)
    
    print(f"\nCombined dataset: {len(combined_features)} samples from {len(stock_files)} stocks")
    
    return combined_features, combined_targets, stock_labels

def train_models(X, y, n_features=30):
    """Train ensemble models."""
    print(f"\nTraining models with {n_features} features...")
    
    # Feature selection
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    selector = SelectKBest(score_func=f_classif, k=min(n_features, len(X_numeric.columns)))
    X_selected = selector.fit_transform(X_numeric, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = X_numeric.columns[selected_indices].tolist()
    
    print(f"Selected features: {len(selected_features)}")
    
    # Show macro features
    macro_features = ['FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI', 'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 'IndustrialProduction', 'RetailSales', 'ConsumerConfidence']
    selected_macro = [f for f in selected_features if f in macro_features]
    if selected_macro:
        print(f"Selected macro features: {selected_macro}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Create models
    models = {}
    models['random_forest'] = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    models['gradient_boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    if XGBOOST_AVAILABLE:
        models['xgboost'] = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    
    # Train and evaluate
    cv_scores = {}
    trained_models = {}
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        cv_scores[name] = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
        print(f"  CV Accuracy: {cv_scores[name].mean():.4f} (+/- {cv_scores[name].std() * 2:.4f})")
        
        model.fit(X_scaled, y)
        trained_models[name] = model
    
    return {
        'models': trained_models,
        'scaler': scaler,
        'selected_features': selected_features,
        'cv_scores': cv_scores,
        'X_scaled': X_scaled,
        'y': y
    }

def main():
    print("="*60)
    print("WORKING MACRO-ENHANCED MODEL TESTING")
    print("="*60)
    
    # Get macro files
    stock_files = get_macro_files()
    print(f"Found {len(stock_files)} macro-enriched stock files")
    
    if len(stock_files) < 5:
        print("Warning: Few macro files found!")
        return
    
    # Use first 10 stocks for testing
    test_stocks = dict(list(stock_files.items())[:10])
    print(f"Testing with {len(test_stocks)} stocks: {list(test_stocks.keys())}")
    
    try:
        # Prepare features
        X, y, stock_labels = prepare_features_manual(test_stocks)
        
        # Train models
        results = train_models(X, y, n_features=40)
        
        # Results summary
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nCross-Validation Results:")
        for name, scores in results['cv_scores'].items():
            print(f"  {name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Save model
        model_data = {
            'models': results['models'],
            'scaler': results['scaler'],
            'selected_features': results['selected_features'],
            'cv_scores': results['cv_scores'],
            'training_date': datetime.now(),
            'symbols_trained': list(test_stocks.keys()),
            'macro_features_included': True,
            'macro_features_used': 'FRED_only'
        }
        
        model_filename = f"working_macro_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(model_data, model_filename)
        print(f"\nModel saved to: {model_filename}")
        print(f"Trained on {len(test_stocks)} stocks with FRED macro features")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 