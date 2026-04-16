import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

def quick_overfitting_fix(price_file, options_file=None, use_comprehensive=False, n_features=20):
    """Quick fix for overfitting using basic feature selection and regularization."""
    
    print("="*50)
    print("QUICK OVERFITTING FIX")
    print("="*50)
    
    # Load and prepare data
    print("Loading data...")
    X, y, feature_names = load_and_engineer_features(
        price_path=price_file,
        options_path=options_file,
        use_comprehensive_options=use_comprehensive
    )
    
    print(f"Original features: {len(feature_names)}")
    
    # Quick feature selection using univariate selection
    print(f"\nSelecting top {n_features} features...")
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"Selected features: {selected_features}")
    
    # Split data (use recent 20% for testing)
    split_idx = int(len(X_selected) * 0.8)
    X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train regularized model
    print("\nTraining regularized model...")
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100
        max_depth=6,      # Reduced from default
        min_samples_split=15,  # Increased
        min_samples_leaf=8,    # Increased
        max_features='sqrt',   # Limit features per split
        random_state=42
    )
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for speed
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    overfitting = train_accuracy - test_accuracy
    
    print(f"\nModel Performance:")
    print(f"  Train Accuracy: {train_accuracy:.3f}")
    print(f"  Test Accuracy: {test_accuracy:.3f}")
    print(f"  Overfitting: {overfitting:.3f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Feature Importance:")
    print(importance_df.head(10))
    
    # Save model
    model_data = {
        'model': model,
        'features': selected_features,
        'cv_scores': cv_scores,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'overfitting': overfitting,
        'feature_importance': importance_df,
        'creation_date': datetime.now()
    }
    
    output_file = "quick_fixed_model.pkl"
    joblib.dump(model_data, output_file)
    
    print(f"\nModel saved to: {output_file}")
    print(f"Feature reduction: {((len(feature_names) - len(selected_features)) / len(feature_names) * 100):.1f}%")
    
    if overfitting < 0.1:
        print("✅ Overfitting successfully reduced!")
    else:
        print("⚠️  Overfitting still present but reduced")
    
    return model_data

def main():
    parser = argparse.ArgumentParser(description="Quick overfitting fix using feature selection and regularization.")
    parser.add_argument("price_file", help="Path to the historical price data (e.g., apple.csv).")
    parser.add_argument("--options", help="Path to the comprehensive options data CSV.")
    parser.add_argument("--use-comprehensive", action='store_true',
                        help="Signal that the model uses comprehensive options features.")
    parser.add_argument("--n-features", type=int, default=20,
                        help="Number of features to select (default: 20).")
    
    args = parser.parse_args()
    
    model_data = quick_overfitting_fix(
        price_file=args.price_file,
        options_file=args.options,
        use_comprehensive=args.use_comprehensive,
        n_features=args.n_features
    )

if __name__ == "__main__":
    main() 