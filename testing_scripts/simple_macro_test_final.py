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

def test_macro_features():
    """Test macro features directly."""
    print("Testing macro features directly...")
    
    # Load macro features
    macro_df = pd.read_csv('macro_features_full.csv', index_col=0, parse_dates=True)
    print(f"Macro features shape: {macro_df.shape}")
    print(f"Macro features columns: {list(macro_df.columns)}")
    
    # Check data quality
    for col in macro_df.columns:
        nan_count = macro_df[col].isnull().sum()
        print(f"  {col}: {nan_count} NaN out of {len(macro_df)}")
    
    # Load a few stock files and test
    stock_files = [f for f in os.listdir('.') if f.endswith('_data_full_clean.csv')][:5]
    print(f"\nTesting with {len(stock_files)} stock files")
    
    all_data = []
    
    for file in stock_files:
        try:
            symbol = file.replace('_data_full_clean.csv', '').upper()
            print(f"\nProcessing {symbol}...")
            
            # Load stock data
            stock_df = pd.read_csv(file, index_col=0, parse_dates=True)
            print(f"  Stock data shape: {stock_df.shape}")
            
            # Merge with macro features
            merged = stock_df.join(macro_df, how='left')
            print(f"  Merged shape: {merged.shape}")
            
            # Forward-fill macro features
            macro_cols = [col for col in macro_df.columns if col in merged.columns]
            for col in macro_cols:
                merged[col] = merged[col].ffill().bfill()
            
            # Basic features
            merged['MA_20'] = merged['Close'].rolling(20).mean()
            merged['MA_50'] = merged['Close'].rolling(50).mean()
            merged['RSI_14'] = 100 - (100 / (1 + merged['Close'].rolling(14).apply(lambda x: (x.diff() > 0).sum() / (x.diff() < 0).sum())))
            merged['MACD'] = merged['Close'].ewm(span=12).mean() - merged['Close'].ewm(span=26).mean()
            
            # Target
            merged['Target'] = (merged['Close'].shift(-1) > merged['Close']).astype(int)
            
            # Select features
            feature_cols = ['MA_20', 'MA_50', 'RSI_14', 'MACD'] + macro_cols
            available_features = [col for col in feature_cols if col in merged.columns]
            
            # Get valid data
            valid_data = merged[available_features + ['Target']].dropna()
            
            if len(valid_data) > 50:
                X = valid_data[available_features]
                y = valid_data['Target']
                
                X['stock_symbol'] = symbol
                all_data.append((X, y, symbol))
                
                print(f"  Valid samples: {len(X)}")
                print(f"  Features: {len(available_features)}")
                print(f"  Macro features: {len([f for f in available_features if f in macro_cols])}")
            else:
                print(f"  Insufficient data: {len(valid_data)}")
                
        except Exception as e:
            print(f"  Error processing {file}: {e}")
    
    if not all_data:
        print("No valid data found!")
        return None
    
    # Combine all data
    combined_X = pd.concat([data[0] for data in all_data], axis=0, ignore_index=False)
    combined_y = pd.concat([data[1] for data in all_data], axis=0, ignore_index=False)
    
    print(f"\nCombined dataset: {len(combined_X)} samples from {len(all_data)} stocks")
    
    return combined_X, combined_y

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
    print("FINAL MACRO FEATURES TESTING")
    print("="*60)
    
    try:
        # Test macro features
        result = test_macro_features()
        
        if result is None:
            print("No valid data found!")
            return
        
        X, y = result
        
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
            'macro_features_included': True,
            'macro_features_used': 'FRED_direct'
        }
        
        model_filename = f"final_macro_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(model_data, model_filename)
        print(f"\nModel saved to: {model_filename}")
        print(f"Trained with macro features from FRED")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 