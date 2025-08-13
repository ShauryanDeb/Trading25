import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import warnings
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EfficientHyperparameterTuning:
    """
    Efficient hyperparameter tuning focusing on best-performing models.
    
    Features:
    - Focused on SVM and LightGBM (best performers)
    - Smaller parameter spaces for faster search
    - Time series cross-validation
    - Risk-adjusted performance metrics
    """
    
    def __init__(self,
                 feature_data: pd.DataFrame,
                 target_column: str = 'target',
                 cv_splits: int = 3,  # Reduced for speed
                 random_state: int = 42):
        
        self.feature_data = feature_data.copy()
        self.target_column = target_column
        self.cv_splits = cv_splits
        self.random_state = random_state
        
        # Results storage
        self.best_models = {}
        self.tuning_results = {}
        
        # Set random seeds
        np.random.seed(random_state)
        
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for tuning."""
        df = self.feature_data.copy()
        
        # Remove target-related columns
        target_cols = ['target', 'target_multi', 'target_regression', 'future_returns']
        feature_cols = [col for col in df.columns if col not in target_cols and col != 'Date']
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        missing_counts = df[feature_cols].isnull().sum() / len(df)
        valid_features = missing_counts[missing_counts < missing_threshold].index.tolist()
        
        # Remove constant columns
        constant_features = []
        for col in valid_features:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        valid_features = [col for col in valid_features if col not in constant_features]
        
        # Prepare X and y
        X = df[valid_features].fillna(0)
        y = df[self.target_column]
        
        # Remove rows with NaN targets
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X.values, y.values, valid_features
    
    def _create_time_series_cv(self) -> TimeSeriesSplit:
        """Create time series cross-validation splitter."""
        # Use a fixed integer test_size for compatibility
        return TimeSeriesSplit(n_splits=self.cv_splits, test_size=500)
    
    def _custom_scorer(self, estimator, X, y):
        """Custom scorer that considers both accuracy and AUC."""
        try:
            y_pred_proba = estimator.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Basic metrics
            accuracy = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)
            
            # Combined score
            combined_score = (accuracy + auc) / 2
            
            return combined_score
        except:
            return 0.0
    
    def tune_svm_focused(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune SVM with focused parameter space."""
        logger.info("Tuning SVM hyperparameters (focused search)...")
        
        # Focused parameter grid based on previous results
        param_grid = {
            'C': [0.5, 1.0, 2.0, 5.0],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly'],
            'class_weight': ['balanced', None]
        }
        
        # Create SVM model
        svm = SVC(probability=True, random_state=self.random_state)
        
        # Create time series CV
        tscv = self._create_time_series_cv()
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            cv=tscv,
            scoring=self._custom_scorer,
            n_jobs=-1,
            verbose=0
        )
        
        # Fit
        grid_search.fit(X, y)
        
        # Store results
        self.best_models['svm'] = grid_search.best_estimator_
        self.tuning_results['svm'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"SVM best parameters: {grid_search.best_params_}")
        logger.info(f"SVM best score: {grid_search.best_score_:.4f}")
        
        return self.tuning_results['svm']
    
    def tune_lightgbm_focused(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune LightGBM with focused parameter space."""
        logger.info("Tuning LightGBM hyperparameters (focused search)...")
        
        # Focused parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Create LightGBM model
        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            verbose=-1,
            objective='binary'
        )
        
        # Create time series CV
        tscv = self._create_time_series_cv()
        
        # Randomized search with fewer iterations
        random_search = RandomizedSearchCV(
            estimator=lgb_model,
            param_distributions=param_grid,
            n_iter=20,  # Reduced for speed
            cv=tscv,
            scoring=self._custom_scorer,
            n_jobs=-1,
            verbose=0,
            random_state=self.random_state
        )
        
        # Fit
        random_search.fit(X, y)
        
        # Store results
        self.best_models['lightgbm'] = random_search.best_estimator_
        self.tuning_results['lightgbm'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
        
        logger.info(f"LightGBM best parameters: {random_search.best_params_}")
        logger.info(f"LightGBM best score: {random_search.best_score_:.4f}")
        
        return self.tuning_results['lightgbm']
    
    def tune_xgboost_focused(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune XGBoost with focused parameter space."""
        logger.info("Tuning XGBoost hyperparameters (focused search)...")
        
        # Focused parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Create XGBoost model
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        # Create time series CV
        tscv = self._create_time_series_cv()
        
        # Randomized search with fewer iterations
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,  # Reduced for speed
            cv=tscv,
            scoring=self._custom_scorer,
            n_jobs=-1,
            verbose=0,
            random_state=self.random_state
        )
        
        # Fit
        random_search.fit(X, y)
        
        # Store results
        self.best_models['xgboost'] = random_search.best_estimator_
        self.tuning_results['xgboost'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
        
        logger.info(f"XGBoost best parameters: {random_search.best_params_}")
        logger.info(f"XGBoost best score: {random_search.best_score_:.4f}")
        
        return self.tuning_results['xgboost']
    
    def optimize_ensemble_weights_simple(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Optimize ensemble weights using simple grid search."""
        logger.info("Optimizing ensemble weights...")
        
        # Split data for weight optimization
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train all models on training set
        models = {}
        for name, model in self.best_models.items():
            models[name] = model.fit(X_train, y_train)
        
        # Get predictions on validation set
        predictions = {}
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(X_val)[:, 1]
            else:
                predictions[name] = model.predict(X_val)
        
        # Simple weight optimization
        best_score = 0
        best_weights = {}
        
        # Try simple weight combinations
        weight_combinations = [
            {'svm': 0.4, 'lightgbm': 0.3, 'xgboost': 0.3},
            {'svm': 0.5, 'lightgbm': 0.3, 'xgboost': 0.2},
            {'svm': 0.3, 'lightgbm': 0.4, 'xgboost': 0.3},
            {'svm': 0.3, 'lightgbm': 0.3, 'xgboost': 0.4},
            {'svm': 0.6, 'lightgbm': 0.2, 'xgboost': 0.2},
            {'svm': 0.2, 'lightgbm': 0.5, 'xgboost': 0.3},
            {'svm': 0.2, 'lightgbm': 0.3, 'xgboost': 0.5},
        ]
        
        for weights in weight_combinations:
            # Create ensemble prediction
            ensemble_pred = (weights['svm'] * predictions['svm'] + 
                           weights['lightgbm'] * predictions['lightgbm'] + 
                           weights['xgboost'] * predictions['xgboost'])
            
            # Calculate score
            binary_pred = (ensemble_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_val, binary_pred)
            auc = roc_auc_score(y_val, ensemble_pred)
            score = (accuracy + auc) / 2
            
            if score > best_score:
                best_score = score
                best_weights = weights
        
        logger.info(f"Best ensemble weights: {best_weights}")
        logger.info(f"Best ensemble score: {best_score:.4f}")
        
        return best_weights
    
    def run_efficient_tuning(self) -> Dict[str, Any]:
        """Run efficient hyperparameter tuning for best models."""
        logger.info("Starting efficient hyperparameter tuning...")
        
        # Prepare data
        X, y, feature_names = self._prepare_data()
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Tune best models
        self.tune_svm_focused(X_scaled, y)
        self.tune_lightgbm_focused(X_scaled, y)
        self.tune_xgboost_focused(X_scaled, y)
        
        # Optimize ensemble weights
        best_weights = self.optimize_ensemble_weights_simple(X_scaled, y)
        
        # Store scaler and feature names
        self.scaler = scaler
        self.feature_names = feature_names
        
        # Compile results
        results = {
            'tuning_results': self.tuning_results,
            'best_models': self.best_models,
            'best_weights': best_weights,
            'scaler': scaler,
            'feature_names': feature_names
        }
        
        logger.info("Efficient tuning completed!")
        return results
    
    def save_tuned_models(self, filepath: str) -> None:
        """Save tuned models and results."""
        model_data = {
            'best_models': self.best_models,
            'tuning_results': self.tuning_results,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Tuned models saved to {filepath}")

def main():
    """Example usage of efficient hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Efficient hyperparameter tuning for financial prediction")
    parser.add_argument("feature_file", help="Feature data CSV file")
    parser.add_argument("--output", help="Output tuned models file")
    parser.add_argument("--report", help="Output tuning report")
    
    args = parser.parse_args()
    
    try:
        # Load feature data
        feature_data = pd.read_csv(args.feature_file, index_col=0, parse_dates=True)
        
        # Create tuner
        tuner = EfficientHyperparameterTuning(feature_data=feature_data)
        
        # Run efficient tuning
        results = tuner.run_efficient_tuning()
        
        # Save tuned models
        if args.output:
            tuner.save_tuned_models(args.output)
        
        # Save tuning report
        if args.report:
            # Create comprehensive report
            report_data = []
            for model_name, tuning_result in results['tuning_results'].items():
                report_data.append({
                    'model': model_name,
                    'best_score': tuning_result['best_score'],
                    'best_params': str(tuning_result['best_params'])
                })
            
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(args.report, index=False)
        
        # Print summary
        print("\nEfficient Hyperparameter Tuning Results:")
        print("=" * 50)
        
        print("\nBest Model Scores:")
        for model_name, tuning_result in results['tuning_results'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Best Score: {tuning_result['best_score']:.4f}")
            print(f"  Best Parameters: {tuning_result['best_params']}")
        
        print(f"\nBest Ensemble Weights:")
        for model_name, weight in results['best_weights'].items():
            print(f"  {model_name}: {weight:.3f}")
        
        best_model = max(results['tuning_results'].items(), key=lambda x: x[1]['best_score'])
        print(f"\nBest Individual Model: {best_model[0]} (Score: {best_model[1]['best_score']:.4f})")
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        raise

if __name__ == "__main__":
    main() 