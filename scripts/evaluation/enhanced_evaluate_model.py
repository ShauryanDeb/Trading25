import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import os

# Add the script's directory to the Python path to allow importing enhanced_features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

# Try to import xgboost for type checking
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def evaluate_model(model_path: str, data_path: str, options_path: str, model_type: str):
    """
    Loads a trained model and evaluates its performance on a given dataset.

    Args:
        model_path (str): Path to the saved model file.
        data_path (str): Path to the historical price data CSV.
        options_path (str): Path to the comprehensive options data CSV.
        model_type (str): The type of model to evaluate ('technical' or 'comprehensive').
    """
    print(f"Loading data and engineering features for '{model_type}' model...")
    
    # Use the same feature engineering pipeline as in training
    X, y, feature_names = load_and_engineer_features(
        data_path,
        options_path if model_type == 'comprehensive' else None,
        target_period=1,
        use_comprehensive_options= (model_type == 'comprehensive')
    )
    
    if X.empty:
        print("Error: No data available for evaluation after feature engineering.")
        print("This can happen if the date ranges of the input files do not overlap.")
        return

    print("Loading model...")
    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            model = model_data['model']
            print("Model extracted from dictionary.")
        else:
            model = model_data
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return

    print("Making predictions...")
    # Use .values for XGBoost models
    if XGBOOST_AVAILABLE and isinstance(model, xgb.XGBClassifier):
        y_pred = model.predict(X.values)
    else:
        y_pred = model.predict(X)
    
    # Ensure y and y_pred have the same indices for comparison
    common_indices = y.index.intersection(X.index)
    y_true = y.loc[common_indices]
    
    # Align predictions with true labels
    y_pred_series = pd.Series(y_pred, index=X.index)
    y_pred_aligned = y_pred_series.loc[common_indices]

    if len(y_true) == 0:
        print("Error: No overlapping data between features and target for evaluation.")
        return

    print("\n--- Model Evaluation ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Features: {model_type.capitalize()}")
    print("-" * 25)

    # --- Metrics ---
    accuracy = accuracy_score(y_true, y_pred_aligned)
    print(f"Accuracy: {accuracy:.2%}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred_aligned))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_aligned))
    print("------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate an enhanced trading model.")
    parser.add_argument("model_path", help="Path to the saved .pkl model file.")
    parser.add_argument("data_path", help="Path to the historical price data (e.g., apple.csv).")
    parser.add_argument("--options_path", help="Path to the comprehensive options data CSV (required for comprehensive model).", default=None)
    parser.add_argument("--model_type", type=str, choices=['technical', 'comprehensive'], required=True,
                        help="Type of model being evaluated: 'technical' or 'comprehensive'.")
    
    args = parser.parse_args()

    if args.model_type == 'comprehensive' and not args.options_path:
        parser.error("--options_path is required when --model_type is 'comprehensive'")

    evaluate_model(args.model_path, args.data_path, args.options_path, args.model_type)

if __name__ == "__main__":
    main() 