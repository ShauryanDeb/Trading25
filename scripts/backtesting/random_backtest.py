import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from contextlib import redirect_stdout

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

def random_backtest(price_file, options_file=None, use_comprehensive=False, model_file='quick_fixed_model.pkl'):
    output_lines = []
    def log(msg):
        print(msg)
        output_lines.append(str(msg))

    log("="*50)
    log("RANDOMIZED BACKTEST (SHUFFLED DATA)")
    log("="*50)
    
    # Load model
    model_data = joblib.load(model_file)
    model = model_data['model']
    selected_features = model_data['features']
    
    # Load and prepare data
    log("Loading and engineering features...")
    X, y, feature_names = load_and_engineer_features(
        price_path=price_file,
        options_path=options_file,
        use_comprehensive_options=use_comprehensive
    )
    
    # Only use the selected features
    X = X[selected_features]
    
    # Shuffle the data
    log("Shuffling data...")
    X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
    
    # Split into train/test (80/20)
    split_idx = int(len(X_shuffled) * 0.8)
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    log(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model from scratch on shuffled data
    log("Training model on shuffled data...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    overfitting = train_accuracy - test_accuracy
    
    log(f"\nModel Performance (Shuffled):")
    log(f"  Train Accuracy: {train_accuracy:.3f}")
    log(f"  Test Accuracy: {test_accuracy:.3f}")
    log(f"  Overfitting: {overfitting:.3f}")
    log(f"\nClassification Report (Test):\n{classification_report(y_test, y_test_pred)}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        log(f"\nTop Feature Importances:")
        log(importance_df)
    
    log("\nIf test accuracy is much higher here than in walk-forward, overfitting is likely.")
    log("If test accuracy is similar, the model is robust.")
    
    # Write results to file
    with open('random_backtest_results.txt', 'w') as f:
        for line in output_lines:
            f.write(str(line) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Randomized backtest to check for overfitting.")
    parser.add_argument("price_file", help="Path to the historical price data (e.g., apple_recent2y.csv).")
    parser.add_argument("--options", help="Path to the comprehensive options data CSV.")
    parser.add_argument("--use-comprehensive", action='store_true',
                        help="Signal that the model uses comprehensive options features.")
    parser.add_argument("--model-file", default="quick_fixed_model.pkl",
                        help="Path to the model file.")
    args = parser.parse_args()
    random_backtest(
        price_file=args.price_file,
        options_file=args.options,
        use_comprehensive=args.use_comprehensive,
        model_file=args.model_file
    )

if __name__ == "__main__":
    main() 