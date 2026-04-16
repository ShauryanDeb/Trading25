import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# For XGBoost and LightGBM (if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

class EnsembleModel:
    """Advanced ensemble model combining multiple algorithms."""
    
    def __init__(self, 
                 models_config: dict = None,
                 voting_method: str = 'soft',
                 use_meta_learner: bool = True):
        
        self.models_config = models_config or self._get_default_config()
        self.voting_method = voting_method
        self.use_meta_learner = use_meta_learner
        self.models = {}
        self.meta_learner = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def _get_default_config(self) -> dict:
        """Get default model configurations."""
        
        config = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'class_weight': 'balanced'
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': 1.0,
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'max_iter': 1000
                }
            },
            'svm': {
                'model': SVC,
                'params': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'probability': True,
                    'random_state': 42
                }
            },
            'neural_network': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.001,
                    'random_state': 42,
                    'max_iter': 500
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            config['xgboost'] = {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'eval_metric': 'logloss'
                }
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            config['lightgbm'] = {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbose': -1
                }
            }
        
        return config
    
    def create_models(self, X: pd.DataFrame, y: pd.Series):
        """Create and initialize all models."""
        
        print("Creating ensemble models...")
        
        for name, config in self.models_config.items():
            try:
                model_class = config['model']
                params = config['params']
                
                print(f"Initializing {name}...")
                model = model_class(**params)
                self.models[name] = model
                
            except Exception as e:
                print(f"Error initializing {name}: {e}")
                continue
        
        # Create meta-learner if requested
        if self.use_meta_learner:
            self.meta_learner = LogisticRegression(
                C=1.0, 
                random_state=42,
                max_iter=1000
            )
        
        print(f"Created {len(self.models)} base models")
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5):
        """Train all models with cross-validation."""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Train each model
        cv_scores = {}
        feature_importance = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)
            
            cv_scores[name] = np.mean(scores)
            print(f"{name} CV accuracy: {cv_scores[name]:.4f}")
            
            # Get feature importance if available
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_importance[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    feature_importance[name] = np.abs(model.coef_[0])
            except:
                pass
        
        # Train final models on full dataset
        print("Training final models on full dataset...")
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
        # Train meta-learner if using
        if self.use_meta_learner:
            print("Training meta-learner...")
            meta_features = self._get_meta_features(X_scaled)
            self.meta_learner.fit(meta_features, y)
        
        self.cv_scores = cv_scores
        self.feature_importance = feature_importance
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from base model predictions."""
        
        meta_features = []
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    meta_features.append(proba[:, 1])  # Probability of positive class
                else:
                    pred = model.predict(X)
                    meta_features.append(pred)
            except:
                # Fallback to predictions
                pred = model.predict(X)
                meta_features.append(pred)
        
        return np.column_stack(meta_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_meta_learner:
            # Use meta-learner
            meta_features = self._get_meta_features(X_scaled)
            predictions = self.meta_learner.predict(meta_features)
        else:
            # Use voting
            predictions = self._voting_predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_meta_learner:
            meta_features = self._get_meta_features(X_scaled)
            probabilities = self.meta_learner.predict_proba(meta_features)
        else:
            probabilities = self._voting_predict_proba(X_scaled)
        
        return probabilities
    
    def _voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Voting-based prediction."""
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            
            # Weight by CV score
            weight = self.cv_scores.get(name, 0.5)
            weights.append(weight)
        
        # Weighted voting
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # For binary classification
        weighted_sum = np.sum(predictions * weights[:, np.newaxis], axis=0)
        threshold = np.sum(weights) / 2
        
        return (weighted_sum > threshold).astype(int)
    
    def _voting_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Voting-based probability prediction."""
        
        probabilities = []
        weights = []
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)
                probabilities.append(proba[:, 1])  # Positive class probability
                weight = self.cv_scores.get(name, 0.5)
                weights.append(weight)
            except:
                # Fallback to predictions
                pred = model.predict(X)
                probabilities.append(pred)
                weight = self.cv_scores.get(name, 0.5)
                weights.append(weight)
        
        # Weighted average
        probabilities = np.array(probabilities)
        weights = np.array(weights)
        
        weighted_proba = np.sum(probabilities * weights[:, np.newaxis], axis=0) / np.sum(weights)
        
        return np.column_stack([1 - weighted_proba, weighted_proba])
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from all models."""
        
        importance_df = pd.DataFrame()
        
        for name, importance in self.feature_importance.items():
            if importance is not None:
                importance_df[name] = importance
        
        # Calculate ensemble importance
        if not importance_df.empty:
            importance_df['ensemble'] = importance_df.mean(axis=1)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the ensemble model."""
        
        model_data = {
            'models': self.models,
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance,
            'models_config': self.models_config,
            'voting_method': self.voting_method,
            'use_meta_learner': self.use_meta_learner
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EnsembleModel':
        """Load a saved ensemble model."""
        
        model_data = joblib.load(filepath)
        
        ensemble = cls(
            models_config=model_data['models_config'],
            voting_method=model_data['voting_method'],
            use_meta_learner=model_data['use_meta_learner']
        )
        
        ensemble.models = model_data['models']
        ensemble.meta_learner = model_data['meta_learner']
        ensemble.scaler = model_data['scaler']
        ensemble.cv_scores = model_data['cv_scores']
        ensemble.feature_importance = model_data['feature_importance']
        
        return ensemble

def create_specialized_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Create specialized models for different market conditions."""
    
    models = {}
    
    # Market regime detection
    returns = X['Close'].pct_change() if 'Close' in X.columns else pd.Series(0, index=X.index)
    volatility = returns.rolling(20).std()
    
    # High volatility model
    high_vol_mask = volatility > volatility.quantile(0.7)
    if high_vol_mask.sum() > 100:
        X_high_vol = X[high_vol_mask]
        y_high_vol = y[high_vol_mask]
        
        high_vol_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        high_vol_model.fit(X_high_vol, y_high_vol)
        models['high_volatility'] = high_vol_model
    
    # Low volatility model
    low_vol_mask = volatility < volatility.quantile(0.3)
    if low_vol_mask.sum() > 100:
        X_low_vol = X[low_vol_mask]
        y_low_vol = y[low_vol_mask]
        
        low_vol_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        low_vol_model.fit(X_low_vol, y_low_vol)
        models['low_volatility'] = low_vol_model
    
    # Trend-following model
    trend = (X['Close'] - X['Close'].rolling(20).mean()) / X['Close'].rolling(20).std() if 'Close' in X.columns else pd.Series(0, index=X.index)
    uptrend_mask = trend > 0.5
    downtrend_mask = trend < -0.5
    
    if uptrend_mask.sum() > 100:
        X_uptrend = X[uptrend_mask]
        y_uptrend = y[uptrend_mask]
        
        uptrend_model = xgb.XGBClassifier(n_estimators=100, random_state=42) if XGBOOST_AVAILABLE else RandomForestClassifier(n_estimators=100, random_state=42)
        uptrend_model.fit(X_uptrend, y_uptrend)
        models['uptrend'] = uptrend_model
    
    if downtrend_mask.sum() > 100:
        X_downtrend = X[downtrend_mask]
        y_downtrend = y[downtrend_mask]
        
        downtrend_model = xgb.XGBClassifier(n_estimators=100, random_state=42) if XGBOOST_AVAILABLE else RandomForestClassifier(n_estimators=100, random_state=42)
        downtrend_model.fit(X_downtrend, y_downtrend)
        models['downtrend'] = downtrend_model
    
    return models

def main():
    """Example usage of ensemble learning."""
    
    print("Ensemble Learning System")
    print("This module provides:")
    print("- Multiple model types (RF, XGBoost, LightGBM, SVM, Neural Network)")
    print("- Meta-learning for model combination")
    print("- Cross-validation for model selection")
    print("- Feature importance analysis")
    print("- Specialized models for different market conditions")
    
    print("\nAvailable models:")
    print("- Random Forest")
    print("- Gradient Boosting")
    print("- Logistic Regression")
    print("- Support Vector Machine")
    print("- Neural Network")
    
    if XGBOOST_AVAILABLE:
        print("- XGBoost")
    
    if LIGHTGBM_AVAILABLE:
        print("- LightGBM")

if __name__ == "__main__":
    main() 