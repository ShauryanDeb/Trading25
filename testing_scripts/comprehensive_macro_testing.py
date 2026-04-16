# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import sys
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

from scripts.enhanced_features import load_and_engineer_features

warnings.filterwarnings('ignore')

def get_macro_enriched_files():
    """Get all available macro-enriched stock files."""
    files = [f for f in os.listdir('.') if f.endswith('_data_full_macro.csv')]
    symbols = [f.replace('_data_full_macro.csv', '').upper() for f in files]
    return dict(zip(symbols, files))

def prepare_macro_features(stock_files, options_file=None, use_comprehensive=False):
    """Prepare features for multiple stocks using macro-enriched files."""
    print("Preparing macro-enriched features...")
    
    all_features = []
    all_targets = []
    stock_labels = []
    feature_counts = []
    
    for symbol, file_path in stock_files.items():
        try:
            print(f"Processing {symbol}...")
            
            # Load and engineer features
            X, y, feature_names = load_and_engineer_features(
                price_path=file_path,
                options_path=options_file,
                use_comprehensive_options=use_comprehensive
            )
            
            if len(X) > 50:  # Ensure sufficient data
                # Add stock identifier
                X['stock_symbol'] = symbol
                
                all_features.append(X)
                all_targets.append(y)
                stock_labels.extend([symbol] * len(X))
                feature_counts.append(len(feature_names))
                
                print(f"  {symbol}: {len(X)} samples, {len(feature_names)} features")
                
                # Show macro features if available
                macro_features = [col for col in feature_names if col in ['FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI', 'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 'IndustrialProduction', 'RetailSales', 'ConsumerConfidence', 'ISM_Manufacturing', 'VIX', 'SP500']]
                if macro_features:
                    print(f"    Macro features: {len(macro_features)} - {macro_features[:5]}...")
            else:
                print(f"  {symbol}: Insufficient samples ({len(X)})")
                
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
    
    if not all_features:
        raise ValueError("No valid stock data found")
    
    # Combine all stocks
    combined_features = pd.concat(all_features, axis=0, ignore_index=False)
    combined_targets = pd.concat(all_targets, axis=0, ignore_index=False)
    
    print(f"\nCombined dataset: {len(combined_features)} samples from {len(stock_files)} stocks")
    print(f"Average features per stock: {np.mean(feature_counts):.1f}")
    print(f"Feature range: {min(feature_counts)} - {max(feature_counts)}")
    
    return combined_features, combined_targets, stock_labels

def create_advanced_models():
    """Create advanced ensemble models with macro features."""
    models = {}
    
    # 1. Random Forest with regularization
    models['random_forest'] = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # 2. Gradient Boosting with regularization
    models['gradient_boosting'] = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        random_state=42
    )
    
    # 3. XGBoost with regularization
    if XGBOOST_AVAILABLE:
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
    
    return models

def train_macro_ensemble(stock_files, options_file=None, use_comprehensive=False, n_features=50):
    """Train ensemble with macro features."""
    print("Training macro-enhanced ensemble model...")
    
    # Prepare macro-enriched data
    X, y, stock_labels = prepare_macro_features(
        stock_files, options_file, use_comprehensive
    )
    
    # Feature selection
    print(f"\nSelecting top {n_features} features...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    selector = SelectKBest(score_func=f_classif, k=min(n_features, len(X_numeric.columns)))
    X_selected = selector.fit_transform(X_numeric, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = X_numeric.columns[selected_indices].tolist()
    
    print(f"Selected features: {len(selected_features)}")
    
    # Show top macro features
    macro_features = [f for f in selected_features if f in ['FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI', 'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 'IndustrialProduction', 'RetailSales', 'ConsumerConfidence', 'ISM_Manufacturing', 'VIX', 'SP500']]
    if macro_features:
        print(f"Selected macro features: {macro_features}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Create and train models with cross-validation
    models = create_advanced_models()
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
        'X_scaled': X_scaled,
        'y': y,
        'stock_labels': stock_labels
    }

def walk_forward_analysis(results, n_splits=5):
    """Perform walk-forward analysis."""
    print(f"\n{'='*60}")
    print("WALK-FORWARD ANALYSIS")
    print(f"{'='*60}")
    
    X_scaled = results['X_scaled']
    y = results['y']
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train models on this fold
        models = create_advanced_models()
        fold_predictions = []
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fold_predictions.append(pred)
        
        # Ensemble prediction
        ensemble_pred = np.mean(fold_predictions, axis=0) > 0.5
        accuracy = accuracy_score(y_test, ensemble_pred)
        fold_scores.append(accuracy)
        
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\nWalk-forward results:")
    print(f"  Mean accuracy: {np.mean(fold_scores):.4f}")
    print(f"  Std accuracy: {np.std(fold_scores):.4f}")
    print(f"  Min accuracy: {np.min(fold_scores):.4f}")
    print(f"  Max accuracy: {np.max(fold_scores):.4f}")
    
    return fold_scores

def feature_importance_analysis(results):
    """Analyze feature importance across models."""
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}")
    
    selected_features = results['selected_features']
    models = results['models']
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            print(f"\n{name.upper()} Feature Importance:")
            
            # Get feature importance
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Show top 10 features
            print("Top 10 features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
            
            # Show macro features importance
            macro_features = [f for f in selected_features if f in ['FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI', 'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 'IndustrialProduction', 'RetailSales', 'ConsumerConfidence', 'ISM_Manufacturing', 'VIX', 'SP500']]
            if macro_features:
                macro_importance = feature_importance[feature_importance['feature'].isin(macro_features)]
                if not macro_importance.empty:
                    print(f"\nMacro features importance:")
                    for _, row in macro_importance.iterrows():
                        print(f"  {row['feature']:<25} {row['importance']:.4f}")

def main():
    print("="*80)
    print("COMPREHENSIVE MACRO-ENHANCED MODEL TESTING")
    print("="*80)
    
    # Get macro-enriched files
    stock_files = get_macro_enriched_files()
    print(f"Found {len(stock_files)} macro-enriched stock files")
    
    if len(stock_files) < 5:
        print("Warning: Few macro-enriched files found. Consider running add_macro_features.py first.")
        return
    
    # Select a subset for testing (first 20 stocks)
    test_stocks = dict(list(stock_files.items())[:20])
    print(f"Testing with {len(test_stocks)} stocks: {list(test_stocks.keys())}")
    
    try:
        # Train ensemble with macro features
        results = train_macro_ensemble(
            test_stocks, 
            options_file=None, 
            use_comprehensive=False, 
            n_features=50
        )
        
        # Analyze results
        print(f"\n{'='*60}")
        print("ENSEMBLE RESULTS SUMMARY")
        print(f"{'='*60}")
        
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
        
        # Walk-forward analysis
        walk_forward_scores = walk_forward_analysis(results)
        
        # Feature importance analysis
        feature_importance_analysis(results)
        
        # Save model
        model_data = {
            'models': results['models'],
            'scaler': results['scaler'],
            'selected_features': results['selected_features'],
            'cv_scores': results['cv_scores'],
            'stock_performance': results['stock_performance'],
            'walk_forward_scores': walk_forward_scores,
            'training_date': datetime.now(),
            'symbols_trained': list(test_stocks.keys()),
            'model_config': {
                'n_features_select': 50,
                'use_comprehensive_options': False,
                'macro_features_included': True
            }
        }
        
        model_filename = f"macro_enhanced_ensemble_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(model_data, model_filename)
        print(f"\nMacro-enhanced ensemble model saved to: {model_filename}")
        print(f"Trained on {len(test_stocks)} stocks with macro features")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 