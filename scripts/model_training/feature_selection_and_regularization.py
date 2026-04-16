import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

class FeatureSelector:
    """
    Feature selection and regularization system to prevent overfitting.
    """
    
    def __init__(self):
        self.selected_features = []
        self.feature_importance = {}
        self.correlation_matrix = None
        
    def analyze_feature_importance(self, X, y, feature_names):
        """Analyze feature importance using Random Forest."""
        print("Analyzing feature importance...")
        
        # Train a Random Forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 20 Most Important Features:")
        print(importance_df.head(20))
        
        print(f"\nBottom 20 Least Important Features:")
        print(importance_df.tail(20))
        
        self.feature_importance = dict(zip(feature_names, rf.feature_importances_))
        
        return importance_df
    
    def analyze_correlations(self, X, feature_names):
        """Analyze feature correlations to identify redundant features."""
        print("\nAnalyzing feature correlations...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > 0.95
        high_corr_features = []
        for column in upper_tri.columns:
            high_corr = upper_tri[column][upper_tri[column] > 0.95]
            if len(high_corr) > 0:
                for feature in high_corr.index:
                    high_corr_features.append((column, feature, high_corr[feature]))
        
        print(f"\nHighly Correlated Feature Pairs (correlation > 0.95):")
        for feat1, feat2, corr in high_corr_features[:10]:  # Show first 10
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")
        
        self.correlation_matrix = corr_matrix
        return high_corr_features
    
    def select_features_univariate(self, X, y, feature_names, k=30):
        """Select top k features using univariate feature selection."""
        print(f"\nPerforming univariate feature selection (top {k} features)...")
        
        # Use f_classif for feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        print(f"Selected {len(selected_features)} features using univariate selection")
        print(f"Selected features: {selected_features[:10]}...")
        
        return selected_features, X_selected
    
    def select_features_recursive(self, X, y, feature_names, n_features=30):
        """Select features using recursive feature elimination."""
        print(f"\nPerforming recursive feature elimination (top {n_features} features)...")
        
        # Use Random Forest as estimator for RFE
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selector.support_[i]]
        
        print(f"Selected {len(selected_features)} features using recursive elimination")
        print(f"Selected features: {selected_features[:10]}...")
        
        return selected_features, X_selected
    
    def select_features_importance(self, X, y, feature_names, threshold=0.01):
        """Select features based on importance threshold."""
        print(f"\nSelecting features based on importance threshold ({threshold})...")
        
        # Train a Random Forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        # Select features above threshold
        selector = SelectFromModel(rf, threshold=threshold)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selector.get_support()[i]]
        
        print(f"Selected {len(selected_features)} features using importance threshold")
        print(f"Selected features: {selected_features[:10]}...")
        
        return selected_features, X_selected

class RegularizedModel:
    """
    Regularized model training to prevent overfitting.
    """
    
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        
    def train_regularized_model(self, X, y, feature_names, model_type='regularized_rf'):
        """Train a regularized model with cross-validation."""
        print(f"\nTraining {model_type} with cross-validation...")
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        if model_type == 'regularized_rf':
            # Regularized Random Forest with constraints
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,  # Reduced depth to prevent overfitting
                min_samples_split=10,  # Increased to reduce overfitting
                min_samples_leaf=5,  # Increased to reduce overfitting
                max_features='sqrt',  # Limit features per split
                random_state=42,
                bootstrap=True
            )
        elif model_type == 'conservative_rf':
            # More conservative Random Forest
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='log2',
                random_state=42,
                bootstrap=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model on full dataset
        model.fit(X, y)
        
        # Store results
        self.models[model_type] = model
        self.cv_scores[model_type] = cv_scores
        
        return model, cv_scores
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, feature_names):
        """Evaluate model performance."""
        # Train accuracy
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Test accuracy
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Calculate overfitting
        overfitting = train_accuracy - test_accuracy
        
        print(f"\nModel Evaluation:")
        print(f"  Train Accuracy: {train_accuracy:.3f}")
        print(f"  Test Accuracy: {test_accuracy:.3f}")
        print(f"  Overfitting (Train - Test): {overfitting:.3f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Feature Importance:")
            print(importance_df.head(10))
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting': overfitting,
            'feature_importance': importance_df if hasattr(model, 'feature_importances_') else None
        }

def main():
    parser = argparse.ArgumentParser(description="Feature selection and regularization to prevent overfitting.")
    parser.add_argument("price_file", help="Path to the historical price data (e.g., apple.csv).")
    parser.add_argument("--options", help="Path to the comprehensive options data CSV.")
    parser.add_argument("--use-comprehensive", action='store_true',
                        help="Signal that the model uses comprehensive options features.")
    parser.add_argument("--n-features", type=int, default=30,
                        help="Number of features to select (default: 30).")
    parser.add_argument("--importance-threshold", type=float, default=0.01,
                        help="Importance threshold for feature selection (default: 0.01).")
    parser.add_argument("--output", default="regularized_model.pkl",
                        help="Output model file.")
    
    args = parser.parse_args()
    
    print("="*60)
    print("FEATURE SELECTION AND REGULARIZATION")
    print("="*60)
    
    # Load and prepare data
    print("Loading and preparing data...")
    X, y, feature_names = load_and_engineer_features(
        price_path=args.price_file,
        options_path=args.options,
        use_comprehensive_options=args.use_comprehensive
    )
    
    print(f"Original data shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Initialize feature selector
    selector = FeatureSelector()
    
    # Analyze feature importance
    importance_df = selector.analyze_feature_importance(X, y, feature_names)
    
    # Analyze correlations
    high_corr_features = selector.analyze_correlations(X, feature_names)
    
    # Perform feature selection using multiple methods
    print("\n" + "="*60)
    print("FEATURE SELECTION RESULTS")
    print("="*60)
    
    # Method 1: Univariate selection
    univariate_features, X_univariate = selector.select_features_univariate(
        X, y, feature_names, k=args.n_features
    )
    
    # Method 2: Recursive feature elimination
    rfe_features, X_rfe = selector.select_features_recursive(
        X, y, feature_names, n_features=args.n_features
    )
    
    # Method 3: Importance-based selection
    importance_features, X_importance = selector.select_features_importance(
        X, y, feature_names, threshold=args.importance_threshold
    )
    
    # Compare methods
    print(f"\nFeature Selection Comparison:")
    print(f"  Univariate: {len(univariate_features)} features")
    print(f"  RFE: {len(rfe_features)} features")
    print(f"  Importance: {len(importance_features)} features")
    
    # Choose the method with the most features (most conservative)
    if len(importance_features) >= len(rfe_features) and len(importance_features) >= len(univariate_features):
        selected_features = importance_features
        X_selected = X_importance
        method = "importance"
    elif len(rfe_features) >= len(univariate_features):
        selected_features = rfe_features
        X_selected = X_rfe
        method = "rfe"
    else:
        selected_features = univariate_features
        X_selected = X_univariate
        method = "univariate"
    
    print(f"\nSelected method: {method} with {len(selected_features)} features")
    
    # Split data for evaluation (use recent data for testing)
    split_idx = int(len(X_selected) * 0.8)
    X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nData split: Train {len(X_train)}, Test {len(X_test)}")
    
    # Initialize regularized model trainer
    trainer = RegularizedModel()
    
    # Train regularized model
    print("\n" + "="*60)
    print("REGULARIZED MODEL TRAINING")
    print("="*60)
    
    model, cv_scores = trainer.train_regularized_model(
        X_train, y_train, selected_features, model_type='regularized_rf'
    )
    
    # Evaluate model
    results = trainer.evaluate_model(
        model, X_train, y_train, X_test, y_test, selected_features
    )
    
    # Save the regularized model
    model_data = {
        'model': model,
        'features': selected_features,
        'feature_selection_method': method,
        'cv_scores': cv_scores,
        'evaluation_results': results,
        'original_features': feature_names,
        'creation_date': datetime.now()
    }
    
    joblib.dump(model_data, args.output)
    print(f"\nRegularized model saved to: {args.output}")
    
    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original features: {len(feature_names)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Feature reduction: {((len(feature_names) - len(selected_features)) / len(feature_names) * 100):.1f}%")
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print(f"Overfitting: {results['overfitting']:.3f}")
    
    if results['overfitting'] < 0.1:
        print("✅ Overfitting successfully reduced!")
    else:
        print("⚠️  Overfitting still present - consider more aggressive regularization")

if __name__ == "__main__":
    main() 