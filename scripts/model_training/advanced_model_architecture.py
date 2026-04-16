import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
import joblib
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedModelArchitecture:
    """
    Advanced model architecture for financial time series prediction.
    
    Features:
    - Ensemble methods with multiple base models
    - Adaptive learning and online updates
    - Feature importance and model interpretability
    - Multi-timeframe prediction capabilities
    - Risk-adjusted performance metrics
    """
    
    def __init__(self,
                 feature_data: pd.DataFrame,
                 target_column: str = 'target',
                 prediction_horizon: int = 1,
                 validation_split: float = 0.2,
                 test_split: float = 0.2,
                 random_state: int = 42):
        
        self.feature_data = feature_data.copy()
        self.target_column = target_column
        self.prediction_horizon = prediction_horizon
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_state = random_state
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Data splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Set random seeds
        np.random.seed(random_state)
        
    def _create_target_variable(self) -> pd.DataFrame:
        """Create target variable for prediction."""
        df = self.feature_data.copy()
        
        # Create future returns
        df['future_returns'] = df['Close'].shift(-self.prediction_horizon) / df['Close'] - 1
        
        # Create binary classification target
        df['target'] = np.where(df['future_returns'] > 0, 1, 0)
        
        # Create multi-class target (strong buy, buy, hold, sell, strong sell)
        returns_quantiles = df['future_returns'].quantile([0.2, 0.4, 0.6, 0.8])
        df['target_multi'] = pd.cut(df['future_returns'], 
                                  bins=[-np.inf, returns_quantiles[0.2], returns_quantiles[0.4], 
                                       returns_quantiles[0.6], returns_quantiles[0.8], np.inf],
                                  labels=[0, 1, 2, 3, 4])
        
        # Create regression target (actual return)
        df['target_regression'] = df['future_returns']
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target', 'target_multi', 'target_regression'])
        
        return df
    
    def _prepare_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for modeling."""
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
        
        logger.info(f"Selected {len(valid_features)} features out of {len(feature_cols)} total features")
        
        return df[valid_features], valid_features
    
    def _split_data(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """Split data into train, validation, and test sets."""
        
        # Calculate split indices
        total_samples = len(df)
        test_size = int(total_samples * self.test_split)
        val_size = int(total_samples * self.validation_split)
        train_size = total_samples - test_size - val_size
        
        # Time series split (no look-ahead bias)
        train_end = train_size
        val_end = train_end + val_size
        
        # Split features
        self.X_train = df[feature_cols].iloc[:train_end]
        self.X_val = df[feature_cols].iloc[train_end:val_end]
        self.X_test = df[feature_cols].iloc[val_end:]
        
        # Split targets
        self.y_train = df[self.target_column].iloc[:train_end]
        self.y_val = df[self.target_column].iloc[train_end:val_end]
        self.y_test = df[self.target_column].iloc[val_end:]
        
        logger.info(f"Data split: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
    
    def train_ensemble_model(self) -> Dict[str, Any]:
        """Train ensemble model with multiple base models."""
        
        logger.info("Starting ensemble model training...")
        
        # Prepare data
        df_with_targets = self._create_target_variable()
        X_processed, feature_cols = self._prepare_features_from_df(df_with_targets)
        self._split_data_from_df(df_with_targets, X_processed, feature_cols)
        
        # Scale features
        scaler = RobustScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_val_scaled = scaler.transform(self.X_val)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        self.scalers['robust'] = scaler
        
        # Train base models
        base_models = {}
        
        # 1. Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(self.X_train_scaled, self.y_train)
        base_models['random_forest'] = rf_model
        
        # 2. XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        xgb_model.fit(self.X_train_scaled, self.y_train,
                     eval_set=[(self.X_val_scaled, self.y_val)],
                     verbose=False)
        base_models['xgboost'] = xgb_model
        
        # 3. LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        lgb_model.fit(self.X_train_scaled, self.y_train,
                     eval_set=[(self.X_val_scaled, self.y_val)],
                     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        base_models['lightgbm'] = lgb_model
        
        # 4. Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        gb_model.fit(self.X_train_scaled, self.y_train)
        base_models['gradient_boosting'] = gb_model
        
        # 5. Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.random_state
        )
        lr_model.fit(self.X_train_scaled, self.y_train)
        base_models['logistic_regression'] = lr_model
        
        # 6. Support Vector Machine
        logger.info("Training Support Vector Machine...")
        svm_model = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=self.random_state
        )
        svm_model.fit(self.X_train_scaled, self.y_train)
        base_models['svm'] = svm_model
        
        # Store base models
        self.models.update(base_models)
        
        # Create ensemble predictions
        ensemble_results = self._create_ensemble_predictions()
        
        # Calculate performance metrics
        self._calculate_performance_metrics(ensemble_results)
        
        logger.info("Ensemble model training completed!")
        
        return {
            'models': self.models,
            'scalers': self.scalers,
            'performance': self.performance_metrics,
            'feature_importance': self.feature_importance
        }
    
    def _prepare_features_from_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for modeling from a given DataFrame."""
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
        logger.info(f"Selected {len(valid_features)} features out of {len(feature_cols)} total features")
        return df[valid_features], valid_features

    def _split_data_from_df(self, df: pd.DataFrame, X: pd.DataFrame, feature_cols: List[str]) -> None:
        """Split data into train, validation, and test sets from a given DataFrame."""
        total_samples = len(df)
        test_size = int(total_samples * self.test_split)
        val_size = int(total_samples * self.validation_split)
        train_size = total_samples - test_size - val_size
        train_end = train_size
        val_end = train_end + val_size
        self.X_train = X.iloc[:train_end]
        self.X_val = X.iloc[train_end:val_end]
        self.X_test = X.iloc[val_end:]
        self.y_train = df[self.target_column].iloc[:train_end]
        self.y_val = df[self.target_column].iloc[train_end:val_end]
        self.y_test = df[self.target_column].iloc[val_end:]
        logger.info(f"Data split: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
    
    def _create_ensemble_predictions(self) -> Dict[str, np.ndarray]:
        """Create ensemble predictions from all models."""
        
        ensemble_results = {}
        
        # Get predictions from base models
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                predictions = model.predict(self.X_test_scaled)
            ensemble_results[name] = predictions
        
        # Create weighted ensemble
        weights = {
            'random_forest': 0.2,
            'xgboost': 0.25,
            'lightgbm': 0.25,
            'gradient_boosting': 0.1,
            'logistic_regression': 0.1,
            'svm': 0.1
        }
        
        ensemble_pred = np.zeros(len(self.y_test))
        for name, pred in ensemble_results.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        ensemble_results['ensemble'] = ensemble_pred
        
        return ensemble_results
    
    def _calculate_performance_metrics(self, ensemble_results: Dict[str, np.ndarray]) -> None:
        """Calculate comprehensive performance metrics."""
        
        self.performance_metrics = {}
        
        for name, predictions in ensemble_results.items():
            # Convert probabilities to binary predictions
            binary_pred = (predictions > 0.5).astype(int)
            
            # Basic metrics
            accuracy = accuracy_score(self.y_test, binary_pred)
            auc = roc_auc_score(self.y_test, predictions)
            
            # Classification report
            report = classification_report(self.y_test, binary_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, binary_pred)
            
            # Risk-adjusted metrics
            returns = self._calculate_strategy_returns(predictions)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            self.performance_metrics[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'confusion_matrix': cm,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': (1 + returns).prod() - 1
            }
    
    def _calculate_strategy_returns(self, predictions: np.ndarray) -> pd.Series:
        """Calculate strategy returns based on predictions."""
        
        # Get actual returns for test period
        test_returns = self.feature_data['returns'].iloc[-len(predictions):]
        
        # Create strategy: long when prediction > 0.5, short when < 0.5
        strategy_returns = np.where(predictions > 0.5, test_returns, -test_returns)
        
        return pd.Series(strategy_returns, index=test_returns.index)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance from all models."""
        
        self.feature_importance = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_df = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[name] = importance_df
            
            elif hasattr(model, 'coef_'):
                # Linear models
                importance_df = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': np.abs(model.coef_[0])
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[name] = importance_df
        
        return self.feature_importance
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'performance_metrics': self.performance_metrics,
            'feature_columns': self.X_train.columns.tolist(),
            'target_column': self.target_column,
            'prediction_horizon': self.prediction_horizon
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.performance_metrics = model_data['performance_metrics']
        
        logger.info(f"Model loaded from {filepath}")

def main():
    """Example usage of advanced model architecture."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced model architecture for financial prediction")
    parser.add_argument("feature_file", help="Feature data CSV file")
    parser.add_argument("--target", default="target", help="Target column name")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--output", help="Output model file")
    parser.add_argument("--report", help="Output performance report")
    
    args = parser.parse_args()
    
    try:
        # Load feature data
        feature_data = pd.read_csv(args.feature_file, index_col=0, parse_dates=True)
        
        # Create model architecture
        model_arch = AdvancedModelArchitecture(
            feature_data=feature_data,
            target_column=args.target,
            prediction_horizon=args.horizon
        )
        
        # Train ensemble model
        results = model_arch.train_ensemble_model()
        
        # Get feature importance
        feature_importance = model_arch.get_feature_importance()
        
        # Save model
        if args.output:
            model_arch.save_model(args.output)
        
        # Save performance report
        if args.report:
            performance_df = pd.DataFrame(results['performance']).T
            performance_df.to_csv(args.report)
        
        # Print summary
        print("\nAdvanced Model Architecture Results:")
        print("=" * 50)
        
        print("\nModel Performance Summary:")
        for model_name, metrics in results['performance'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
            print(f"  Total Return: {metrics['total_return']:.4f}")
        
        print(f"\nBest Model by Sharpe Ratio:")
        best_model = max(results['performance'].items(), key=lambda x: x[1]['sharpe_ratio'])
        print(f"  {best_model[0]}: {best_model[1]['sharpe_ratio']:.4f}")
        
        print(f"\nEnsemble Model Performance:")
        ensemble_metrics = results['performance']['ensemble']
        print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
        print(f"  AUC: {ensemble_metrics['auc']:.4f}")
        print(f"  Sharpe Ratio: {ensemble_metrics['sharpe_ratio']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in advanced model architecture: {e}")
        raise

if __name__ == "__main__":
    main() 