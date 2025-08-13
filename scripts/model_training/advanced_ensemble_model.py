import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

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

class AdvancedEnsembleModel:
    """
    Advanced ensemble model with feature selection, hyperparameter tuning,
    risk management, and multi-stock dependency handling.
    """
    
    def __init__(self, 
                 n_features_select=30,
                 ensemble_size=3,
                 risk_threshold=0.6,
                 max_position_size=0.1,
                 stop_loss_pct=0.05):
        
        self.n_features_select = n_features_select
        self.ensemble_size = ensemble_size
        self.risk_threshold = risk_threshold
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        
        self.models = {}
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.selected_features = []
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def select_features(self, X, y, method='mutual_info'):
        """Select the most important features to reduce overfitting."""
        print(f"Selecting top {self.n_features_select} features using {method}...")
        
        if method == 'mutual_info':
            # Use mutual information for feature selection
            selector = SelectKBest(score_func=f_classif, k=self.n_features_select)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            self.selected_features = X.columns[selected_indices].tolist()
            
        elif method == 'rfe':
            # Use Recursive Feature Elimination
            base_model = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator=base_model, n_features_to_select=self.n_features_select)
            X_selected = selector.fit_transform(X, y)
            self.selected_features = X.columns[selector.support_].tolist()
            
        elif method == 'importance':
            # Use model-based feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            self.selected_features = importance_df.head(self.n_features_select)['feature'].tolist()
            X_selected = X[self.selected_features]
        
        print(f"Selected features: {len(self.selected_features)}")
        print(f"Top 10 features: {self.selected_features[:10]}")
        
        return X_selected
    
    def create_ensemble_models(self):
        """Create ensemble of different model types."""
        models = {}
        
        # 1. Random Forest
        models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        # 2. Gradient Boosting
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # 3. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        
        return models
    
    def tune_hyperparameters(self, X, y, model_name, model):
        """Tune hyperparameters for each model."""
        print(f"Tuning hyperparameters for {model_name}...")
        
        # Define parameter grids for different models
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [8, 10, 12],
                'min_samples_split': [5, 10, 15]
            }
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8]
            }
        elif model_name == 'xgboost':
            param_grid = {
                'n_estimators': [150, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        else:
            return model
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv, scoring='accuracy',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_ensemble(self, X, y):
        """Train the ensemble of models."""
        print("Training ensemble models...")
        
        # Feature selection
        X_selected = self.select_features(X, y, method='importance')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Create and tune models
        base_models = self.create_ensemble_models()
        
        for name, model in base_models.items():
            print(f"\nTraining {name}...")
            
            # Tune hyperparameters
            tuned_model = self.tune_hyperparameters(X_scaled, y, name, model)
            
            # Train final model
            tuned_model.fit(X_scaled, y)
            
            # Store model and get feature importance
            self.models[name] = tuned_model
            
            if hasattr(tuned_model, 'feature_importances_'):
                # Ensure we only use the selected features for importance
                importance_values = tuned_model.feature_importances_
                if len(importance_values) == len(self.selected_features):
                    self.feature_importance[name] = pd.DataFrame({
                        'feature': self.selected_features,
                        'importance': importance_values
                    }).sort_values('importance', ascending=False)
                else:
                    print(f"Warning: Feature importance length mismatch for {name}")
        
        print(f"Trained {len(self.models)} models in ensemble")
    
    def predict_ensemble(self, X, return_proba=True):
        """Make ensemble predictions with risk management."""
        # Select features and scale
        X_selected = X[self.selected_features]
        X_scaled = self.scaler.transform(X_selected)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            if return_proba and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                probabilities[name] = proba[:, 1]  # Probability of positive class
                predictions[name] = (proba[:, 1] > 0.5).astype(int)
            else:
                pred = model.predict(X_scaled)
                predictions[name] = pred
                probabilities[name] = pred.astype(float)
        
        # Ensemble prediction (weighted average)
        ensemble_proba = np.mean(list(probabilities.values()), axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        return ensemble_pred, ensemble_proba, predictions, probabilities
    
    def calculate_position_size(self, probability, current_price, portfolio_value):
        """Calculate position size based on risk management rules."""
        if probability < self.risk_threshold:
            return 0  # No position if confidence is too low
        
        # Base position size based on probability
        base_size = (probability - self.risk_threshold) / (1 - self.risk_threshold)
        
        # Apply maximum position size constraint
        position_size = min(base_size, self.max_position_size)
        
        # Calculate number of shares
        shares = int((portfolio_value * position_size) / current_price)
        
        return shares
    
    def apply_stop_loss(self, entry_price, current_price, position_type='long'):
        """Apply stop-loss logic."""
        if position_type == 'long':
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price
        
        return loss_pct > self.stop_loss_pct
    
    def handle_multi_stock_dependencies(self, stock_data_dict):
        """
        Handle dependencies between multiple stocks.
        This could include sector correlations, market beta, etc.
        """
        # Calculate market-wide indicators
        market_features = {}
        
        for symbol, data in stock_data_dict.items():
            # Add market beta, sector correlation, etc.
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                
                # Market beta (simplified - using average of all stocks as market)
                all_returns = []
                for other_symbol, other_data in stock_data_dict.items():
                    if other_symbol != symbol and 'Close' in other_data.columns:
                        other_returns = other_data['Close'].pct_change().dropna()
                        all_returns.append(other_returns)
                
                if all_returns:
                    market_returns = pd.concat(all_returns, axis=1).mean(axis=1)
                    # Calculate beta (simplified)
                    covariance = returns.cov(market_returns)
                    market_variance = market_returns.var()
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                    
                    market_features[f'{symbol}_beta'] = beta
                    market_features[f'{symbol}_market_correlation'] = returns.corr(market_returns)
        
        return market_features
    
    def evaluate_model(self, X, y):
        """Evaluate the ensemble model."""
        print("\nEvaluating ensemble model...")
        
        # Make predictions
        y_pred, y_proba, individual_preds, individual_probas = self.predict_ensemble(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Individual model performance
        print("\nIndividual Model Performance:")
        for name, preds in individual_preds.items():
            model_acc = accuracy_score(y, preds)
            print(f"{name}: {model_acc:.4f}")
        
        # Feature importance summary
        print("\nTop 10 Features (Average Importance):")
        all_importance = pd.DataFrame()
        for name, importance_df in self.feature_importance.items():
            all_importance[name] = importance_df.set_index('feature')['importance']
        
        avg_importance = all_importance.mean(axis=1).sort_values(ascending=False)
        print(avg_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_proba,
            'individual_predictions': individual_preds,
            'feature_importance': self.feature_importance
        }

def main():
    parser = argparse.ArgumentParser(description="Advanced ensemble model with feature selection and risk management")
    parser.add_argument("price_file", help="Price data CSV file")
    parser.add_argument("--options", help="Options data CSV file")
    parser.add_argument("--model-out", default="advanced_ensemble_model.pkl", help="Output model file")
    parser.add_argument("--use-comprehensive-options", action='store_true',
                        help="Use comprehensive options features")
    parser.add_argument("--n-features", type=int, default=30, help="Number of features to select")
    parser.add_argument("--risk-threshold", type=float, default=0.6, help="Risk threshold for position sizing")
    parser.add_argument("--max-position-size", type=float, default=0.1, help="Maximum position size as fraction of portfolio")
    
    args = parser.parse_args()
    
    try:
        print("Loading and preparing data...")
        
        # Load and engineer features
        X, y, feature_names = load_and_engineer_features(
            price_path=args.price_file,
            options_path=args.options,
            use_comprehensive_options=args.use_comprehensive_options
        )
        
        print(f"Original features: {len(feature_names)}")
        print(f"Data shape: {X.shape}")
        
        # Create and train ensemble model
        ensemble = AdvancedEnsembleModel(
            n_features_select=args.n_features,
            risk_threshold=args.risk_threshold,
            max_position_size=args.max_position_size
        )
        
        # Train ensemble
        ensemble.train_ensemble(X, y)
        
        # Evaluate model
        results = ensemble.evaluate_model(X, y)
        
        # Save model
        model_data = {
            'ensemble': ensemble,
            'feature_names': feature_names,
            'selected_features': ensemble.selected_features,
            'results': results,
            'training_date': datetime.now(),
            'model_config': {
                'n_features_select': args.n_features,
                'risk_threshold': args.risk_threshold,
                'max_position_size': args.max_position_size,
                'use_comprehensive_options': args.use_comprehensive_options
            }
        }
        
        joblib.dump(model_data, args.model_out)
        print(f"\nAdvanced ensemble model saved to: {args.model_out}")
        print(f"Model includes {len(ensemble.selected_features)} selected features")
        print(f"Ensemble accuracy: {results['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 