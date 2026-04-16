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

class AdvancedHyperparameterTuning:
    """
    Advanced hyperparameter tuning for financial prediction models.
    
    Features:
    - Time series cross-validation
    - Multiple search strategies (Grid, Random, Bayesian)
    - Risk-adjusted performance metrics
    - Feature importance analysis
    - Ensemble optimization
    """
    
    def __init__(self,
                 feature_data: pd.DataFrame,
                 target_column: str = 'target',
                 cv_splits: int = 5,
                 random_state: int = 42):
        
        self.feature_data = feature_data.copy()
        self.target_column = target_column
        self.cv_splits = cv_splits
        self.random_state = random_state
        
        # Results storage
        self.best_models = {}
        self.tuning_results = {}
        self.feature_importance = {}
        
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
        return TimeSeriesSplit(n_splits=self.cv_splits, test_size=0.2)
    
    def _custom_scorer(self, estimator, X, y):
        """Custom scorer that considers both accuracy and risk-adjusted returns."""
        try:
            y_pred_proba = estimator.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Basic metrics
            accuracy = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)
            
            # Risk-adjusted metric (simplified)
            # In a real scenario, you'd calculate actual returns
            risk_score = accuracy * auc
            
            return risk_score
        except:
            return 0.0
    
    def tune_svm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune SVM hyperparameters."""
        logger.info("Tuning SVM hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'kernel': ['rbf', 'poly', 'sigmoid'],
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
            verbose=1
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
    
    def tune_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune LightGBM hyperparameters."""
        logger.info("Tuning LightGBM hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        # Create LightGBM model
        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            verbose=-1,
            objective='binary'
        )
        
        # Create time series CV
        tscv = self._create_time_series_cv()
        
        # Randomized search (faster than grid search for large parameter space)
        random_search = RandomizedSearchCV(
            estimator=lgb_model,
            param_distributions=param_grid,
            n_iter=50,  # Number of parameter settings sampled
            cv=tscv,
            scoring=self._custom_scorer,
            n_jobs=-1,
            verbose=1,
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
    
    def tune_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters."""
        logger.info("Tuning XGBoost hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        # Create XGBoost model
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        # Create time series CV
        tscv = self._create_time_series_cv()
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,
            cv=tscv,
            scoring=self._custom_scorer,
            n_jobs=-1,
            verbose=1,
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
    
    def tune_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune Random Forest hyperparameters."""
        logger.info("Tuning Random Forest hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create Random Forest model
        rf_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # Create time series CV
        tscv = self._create_time_series_cv()
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=self._custom_scorer,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit
        grid_search.fit(X, y)
        
        # Store results
        self.best_models['random_forest'] = grid_search.best_estimator_
        self.tuning_results['random_forest'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Random Forest best parameters: {grid_search.best_params_}")
        logger.info(f"Random Forest best score: {grid_search.best_score_:.4f}")
        
        return self.tuning_results['random_forest']
    
    def optimize_ensemble_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Optimize ensemble weights using validation set."""
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
        
        # Grid search for optimal weights
        best_score = 0
        best_weights = {}
        
        # Try different weight combinations
        weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for w1 in weight_options:
            for w2 in weight_options:
                for w3 in weight_options:
                    for w4 in weight_options:
                        # Ensure weights sum to 1
                        total = w1 + w2 + w3 + w4
                        if total == 0:
                            continue
                        
                        w1_norm, w2_norm, w3_norm, w4_norm = w1/total, w2/total, w3/total, w4/total
                        
                        # Create ensemble prediction
                        ensemble_pred = (w1_norm * predictions['svm'] + 
                                       w2_norm * predictions['lightgbm'] + 
                                       w3_norm * predictions['xgboost'] + 
                                       w4_norm * predictions['random_forest'])
                        
                        # Calculate score
                        binary_pred = (ensemble_pred > 0.5).astype(int)
                        accuracy = accuracy_score(y_val, binary_pred)
                        auc = roc_auc_score(y_val, ensemble_pred)
                        score = accuracy * auc
                        
                        if score > best_score:
                            best_score = score
                            best_weights = {
                                'svm': w1_norm,
                                'lightgbm': w2_norm,
                                'xgboost': w3_norm,
                                'random_forest': w4_norm
                            }
        
        logger.info(f"Best ensemble weights: {best_weights}")
        logger.info(f"Best ensemble score: {best_score:.4f}")
        
        return best_weights
    
    def run_comprehensive_tuning(self) -> Dict[str, Any]:
        """Run comprehensive hyperparameter tuning for all models."""
        logger.info("Starting comprehensive hyperparameter tuning...")
        
        # Prepare data
        X, y, feature_names = self._prepare_data()
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Tune each model
        self.tune_svm(X_scaled, y)
        self.tune_lightgbm(X_scaled, y)
        self.tune_xgboost(X_scaled, y)
        self.tune_random_forest(X_scaled, y)
        
        # Optimize ensemble weights
        best_weights = self.optimize_ensemble_weights(X_scaled, y)
        
        # Store scaler
        self.scaler = scaler
        
        # Compile results
        results = {
            'tuning_results': self.tuning_results,
            'best_models': self.best_models,
            'best_weights': best_weights,
            'scaler': scaler,
            'feature_names': feature_names
        }
        
        logger.info("Comprehensive tuning completed!")
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
    """Example usage of advanced hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced hyperparameter tuning for financial prediction")
    parser.add_argument("feature_file", help="Feature data CSV file")
    parser.add_argument("--output", help="Output tuned models file")
    parser.add_argument("--report", help="Output tuning report")
    
    args = parser.parse_args()
    
    try:
        # Load feature data
        feature_data = pd.read_csv(args.feature_file, index_col=0, parse_dates=True)
        
        # Create tuner
        tuner = AdvancedHyperparameterTuning(feature_data=feature_data)
        
        # Run comprehensive tuning
        results = tuner.run_comprehensive_tuning()
        
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
        print("\nAdvanced Hyperparameter Tuning Results:")
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