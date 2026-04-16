import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
import sys
import os

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

def tune_hyperparameters(X, y, feature_names):
    """
    Performs hyperparameter tuning for a RandomForestClassifier using RandomizedSearchCV.
    """
    print("Setting up hyperparameter tuning...")

    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Use TimeSeriesSplit for cross-validation on time-series data
    tscv = TimeSeriesSplit(n_splits=5)

    # Custom scorer to balance accuracy and AUC
    def combined_scorer(y_true, y_pred_proba):
        y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        return (accuracy + auc) / 2
    
    scoring = make_scorer(combined_scorer, needs_proba=True)


    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings that are sampled
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        scoring=scoring
    )

    print("Starting RandomizedSearchCV...")
    random_search.fit(X, y)

    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best cross-validation score (Combined Accuracy & AUC): {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for the enhanced trading model.")
    parser.add_argument("price_file", help="Path to the historical price data (e.g., apple.csv).")
    parser.add_argument("--options", help="Path to the comprehensive options data CSV.")
    parser.add_argument("--use-comprehensive-options", action='store_true',
                        help="Use all available comprehensive options features for tuning.")
    parser.add_argument("--model-out", default="tuned_model.pkl",
                        help="Path to save the best-tuned model.")
    
    args = parser.parse_args()

    try:
        print("Loading and engineering features for tuning...")
        X, y, feature_names = load_and_engineer_features(
            price_path=args.price_file,
            options_path=args.options,
            use_comprehensive_options=args.use_comprehensive_options
        )

        best_model, best_params = tune_hyperparameters(X, y, feature_names)

        # Save the best model along with its parameters and features
        model_data = {
            'model': best_model,
            'features': feature_names,
            'best_params': best_params,
            'feature_type': 'comprehensive' if args.use_comprehensive_options else 'technical',
            'tuning_date': pd.Timestamp.now()
        }

        joblib.dump(model_data, args.model_out)
        print(f"\nBest tuned model saved to {args.model_out}")

    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")
        raise

if __name__ == "__main__":
    main() 