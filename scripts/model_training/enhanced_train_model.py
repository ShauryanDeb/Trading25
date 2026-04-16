import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Import our enhanced features module
from enhanced_features import load_and_engineer_features

# Add XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost is not installed. XGBoost model type will not be available.")


def train_model(X: pd.DataFrame, y: pd.Series, features: list,
                model_type: str = "random_forest", **kwargs) -> object:
    """Train the model with specified parameters."""
    if len(X) < 10:
        raise ValueError(f"Insufficient data: only {len(X)} samples available")

    print(f"Training with {len(X)} samples and {len(features)} features")
    print(f"Feature distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    # Train model
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            random_state=42
        )
        model.fit(X_train, y_train)
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Please install it to use XGBoost.")
        model = xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )
        # XGBoost requires numpy arrays
        model.fit(X_train.values, y_train.values)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Evaluate
    if model_type == "xgboost":
        y_pred = model.predict(X_test.values)
    else:
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    else:
        feature_importance = pd.DataFrame()
        print("\nFeature importance not available for this model type")

    return model, {
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }


def main():
    parser = argparse.ArgumentParser(description="Enhanced model training with options support")
    parser.add_argument("price_file", help="Price data CSV file")
    parser.add_argument("--options", help="Options data CSV file")
    parser.add_argument("--model-out", default="enhanced_model.pkl", help="Output model file")
    parser.add_argument("--use-comprehensive-options", action='store_true',
                        help="Use all available comprehensive options features.")
    parser.add_argument("--model-type", default="random_forest", choices=["random_forest", "xgboost"], help="Model type to train (random_forest or xgboost)")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators for RF/XGBoost")
    parser.add_argument("--max-depth", type=int, help="Max depth for RF/XGBoost")

    args = parser.parse_args()

    try:
        # Prepare data using the centralized function
        X, y, feature_names = load_and_engineer_features(
            price_path=args.price_file,
            options_path=args.options,
            use_comprehensive_options=args.use_comprehensive_options
        )

        # Train model
        model, results = train_model(
            X, y, feature_names,
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )

        # Save model and metadata
        model_data = {
            'model': model,
            'features': feature_names,
            'feature_type': 'comprehensive' if args.use_comprehensive_options else 'basic',
            'results': results,
            'training_date': pd.Timestamp.now()
        }

        joblib.dump(model_data, args.model_out)
        print(f"\nModel saved to {args.model_out}")
        print(f"Model includes {len(feature_names)} features")

    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main() 