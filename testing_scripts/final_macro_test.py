import os
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
import glob

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. Using alternative models.")

# Try to import Optuna for hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available. Using default hyperparameters.")

warnings.filterwarnings('ignore')

def load_macro_data():
    """Load macro features data"""
    print("Loading macro features...")
    macro_df = pd.read_csv('macro_features_full.csv')
    macro_df['DATE'] = pd.to_datetime(macro_df['DATE'])
    macro_df.set_index('DATE', inplace=True)
    
    # Forward fill to handle missing dates
    macro_df = macro_df.ffill().bfill()
    
    print(f"Macro data shape: {macro_df.shape}")
    print(f"Macro date range: {macro_df.index.min()} to {macro_df.index.max()}")
    print(f"Macro columns: {list(macro_df.columns)}")
    
    return macro_df

def load_stock_data():
    """Load a few stock datasets for testing"""
    print("\nLoading stock data...")
    
    # Find stock files
    stock_files = glob.glob('*_data_full.csv')
    stock_files = [f for f in stock_files if not 'macro' in f and not 'clean' in f]
    
    print(f"Found {len(stock_files)} stock files")
    
    # Use all available stock files for comprehensive testing
    test_files = stock_files
    print(f"Testing with {len(test_files)} files.")
    
    all_stocks = []
    
    for file in test_files:
        try:
            symbol = file.replace('_data_full.csv', '').upper()
            print(f"Loading {symbol}...")
            
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            
            # Remove timezone info if present
            if pd.api.types.is_datetime64tz_dtype(df.index):
                df.index = df.index.tz_localize(None)
            
            df['Symbol'] = symbol
            all_stocks.append(df)
            
            print(f"  {symbol} shape: {df.shape}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    if not all_stocks:
        print("No stock data loaded!")
        return None
    
    # Combine all stocks
    combined = pd.concat(all_stocks, ignore_index=False)
    combined = combined.sort_index()
    
    print(f"\nCombined stock data shape: {combined.shape}")
    print(f"Combined date range: {combined.index.min()} to {combined.index.max()}")
    
    return combined

def create_features_with_macro(stock_df, macro_df):
    """Create features combining stock and macro data"""
    print("\nCreating features with macro data...")
    
    # Convert stock index to timezone-naive datetime and normalize to date only
    stock_df_clean = stock_df.copy()
    # Robustly remove timezone info if present
    stock_df_clean.index = pd.Index([
        d.replace(tzinfo=None) if hasattr(d, 'tzinfo') and d.tzinfo is not None else d
        for d in stock_df_clean.index
    ])
    stock_df_clean.index = pd.to_datetime(stock_df_clean.index).normalize()
    
    # Ensure macro index is also timezone-naive and normalized
    macro_df_clean = macro_df.copy()
    macro_df_clean.index = pd.to_datetime(macro_df_clean.index).normalize()
    
    print(f"Stock index type: {type(stock_df_clean.index)}")
    print(f"Macro index type: {type(macro_df_clean.index)}")
    print(f"Stock index sample: {stock_df_clean.index[:3]}")
    print(f"Macro index sample: {macro_df_clean.index[:3]}")
    
    # Use merge_asof to forward-fill macro data for each stock date
    stock_reset = stock_df_clean.reset_index()
    stock_reset = stock_reset.rename(columns={stock_reset.columns[0]: 'Date'})
    macro_reset = macro_df_clean.reset_index()
    macro_reset = macro_reset.rename(columns={macro_reset.columns[0]: 'DATE'})
    merged = pd.merge_asof(
        stock_reset.sort_values('Date'),
        macro_reset.sort_values('DATE'),
        left_on='Date',
        right_on='DATE',
        direction='backward'
    )
    # Set the date back as index
    merged.set_index('Date', inplace=True)
    merged.drop('DATE', axis=1, inplace=True)
    
    print(f"Merged data shape: {merged.shape}")
    
    # Check macro data availability
    macro_cols = macro_df.columns.tolist()
    available_macro = [col for col in macro_cols if col in merged.columns]
    print(f"Available macro features: {available_macro}")
    
    # Check macro data availability after merge
    macro_data_check = merged[available_macro].notna().any(axis=1)
    print(f"Rows with macro data after merge: {macro_data_check.sum()}")
    
    if macro_data_check.sum() > 0:
        print(f"Sample macro values: {merged[available_macro].iloc[0].to_dict()}")
    else:
        print("No macro data available after merge!")
        print(f"Sample merged data: {merged[available_macro].head()}")
        return None, None
    
    # Basic technical features
    merged['Returns'] = merged['Close'].pct_change()
    merged['Log_Returns'] = np.log(merged['Close'] / merged['Close'].shift(1))
    merged['SMA_20'] = merged['Close'].rolling(window=20).mean()
    merged['SMA_50'] = merged['Close'].rolling(window=50).mean()
    merged['RSI'] = calculate_rsi(merged['Close'], 14)
    merged['MACD'] = calculate_macd(merged['Close'])
    merged['BB_Upper'], merged['BB_Lower'] = calculate_bollinger_bands(merged['Close'])
    
    # Volume features
    merged['Volume_SMA'] = merged['Volume'].rolling(window=20).mean()
    merged['Volume_Ratio'] = merged['Volume'] / merged['Volume_SMA']
    
    # Volatility features
    merged['Volatility'] = merged['Returns'].rolling(window=20).std()
    
    # Create target (next day return > 0)
    merged['Target'] = (merged['Returns'].shift(-1) > 0).astype(int)
    
    # Select feature columns
    feature_cols = [
        'Returns', 'Log_Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 
        'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Volatility'
    ] + available_macro
    
    # Remove rows with NaN values, but only require that at least some macro features are present
    # Check which rows have at least one macro feature available
    macro_features_present = merged[available_macro].notna().any(axis=1)
    technical_features_present = merged[['Returns', 'Log_Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 
                                       'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Volatility']].notna().all(axis=1)
    
    # Keep rows where technical features are complete AND at least one macro feature is present
    valid_rows = technical_features_present & macro_features_present
    
    df_clean = merged[feature_cols + ['Target', 'Symbol']].loc[valid_rows]
    
    print(f"Final dataset shape: {df_clean.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Macro features: {len(available_macro)}")
    print(f"Rows with at least one macro feature: {macro_features_present.sum()}")
    print(f"Rows with complete technical features: {technical_features_present.sum()}")
    print(f"Final valid rows: {valid_rows.sum()}")
    
    return df_clean, feature_cols

def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower

def calculate_risk_metrics(y_true, y_pred, returns):
    """Calculate risk-adjusted metrics for trading performance."""
    # Calculate daily returns for correct predictions
    correct_predictions = (y_true == y_pred)
    strategy_returns = returns * correct_predictions
    
    # Basic metrics
    win_rate = np.mean(correct_predictions)
    total_return = np.sum(strategy_returns)
    
    # Risk metrics
    if len(strategy_returns) > 1:
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Volatility
        volatility = np.std(strategy_returns) * np.sqrt(252)
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    else:
        sharpe_ratio = max_drawdown = volatility = calmar_ratio = 0
    
    return {
        'win_rate': win_rate,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'calmar_ratio': calmar_ratio
    }

def optimize_hyperparameters(X, y, n_trials=50):
    """Optimize XGBoost hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        print("Optuna or XGBoost not available. Using default hyperparameters.")
        return None
    
    print(f"\nOptimizing hyperparameters with {n_trials} trials...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'tree_method': 'hist',
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return cv_scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best CV Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")
    
    return study.best_params

def create_ensemble_model(best_params=None):
    """Create an ensemble of multiple models."""
    models = []
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        if best_params:
            xgb_model = xgb.XGBClassifier(**best_params)
        else:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                tree_method='hist', use_label_encoder=False,
                eval_metric='logloss', verbosity=0
            )
        models.append(('xgb', xgb_model))
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    models.append(('rf', rf_model))
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    models.append(('gb', gb_model))
    
    ensemble = VotingClassifier(estimators=models, voting='soft')
    return ensemble

def analyze_sector_performance(df_features, top_features):
    """Analyze performance across different sectors."""
    print("\n=== Sector Performance Analysis ===")
    
    # Define sector mappings (simplified)
    sector_mapping = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'META': 'Technology',
        'AMZN': 'Consumer', 'TSLA': 'Consumer', 'NFLX': 'Consumer', 'SBUX': 'Consumer',
        'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
        'HD': 'Industrial', 'CAT': 'Industrial', 'BA': 'Industrial', 'MMM': 'Industrial'
    }
    
    sector_results = {}
    
    for symbol in df_features['Symbol'].unique():
        if symbol in sector_mapping:
            sector = sector_mapping[symbol]
            sector_data = df_features[df_features['Symbol'] == symbol]
            
            if len(sector_data) > 100:  # Minimum data requirement
                X_sector = sector_data[top_features]
                y_sector = sector_data['Target']
                returns_sector = sector_data['Returns']
                
                # Train model for this sector
                if XGBOOST_AVAILABLE:
                    model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Simple train/test split for sector analysis
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sector, y_sector, test_size=0.2, random_state=42, stratify=y_sector
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_returns = returns_sector.iloc[X_test.index]
                
                accuracy = accuracy_score(y_test, y_pred)
                risk_metrics = calculate_risk_metrics(y_test, y_pred, test_returns)
                
                if sector not in sector_results:
                    sector_results[sector] = []
                
                sector_results[sector].append({
                    'symbol': symbol,
                    'accuracy': accuracy,
                    'sharpe_ratio': risk_metrics['sharpe_ratio'],
                    'win_rate': risk_metrics['win_rate']
                })
    
    # Print sector summary
    for sector, results in sector_results.items():
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            avg_win_rate = np.mean([r['win_rate'] for r in results])
            
            print(f"{sector}: {len(results)} stocks, "
                  f"Avg Accuracy: {avg_accuracy:.3f}, "
                  f"Avg Sharpe: {avg_sharpe:.3f}, "
                  f"Avg Win Rate: {avg_win_rate:.3f}")

def enhanced_walk_forward_backtest(X, y, dates, returns, model_type='ensemble', train_years=2, test_months=3):
    """Enhanced walk-forward backtest with risk metrics and ensemble models."""
    print(f"\nStarting enhanced walk-forward backtest: {train_years} years train, {test_months} months test...")
    
    # Optimize hyperparameters if using XGBoost
    best_params = None
    if model_type == 'xgb' and OPTUNA_AVAILABLE and XGBOOST_AVAILABLE:
        best_params = optimize_hyperparameters(X, y, n_trials=20)  # Reduced trials for speed
    
    results = []
    risk_metrics_list = []
    
    unique_dates = pd.Series(dates).sort_values().unique()
    start = 0
    n = len(unique_dates)
    fold = 1
    
    while True:
        # Find train/test split indices
        train_end = start
        train_start_date = unique_dates[start]
        train_end_date = train_start_date + pd.DateOffset(years=train_years)
        while train_end < n and unique_dates[train_end] < train_end_date:
            train_end += 1
        test_start = train_end
        test_end = test_start
        test_end_date = unique_dates[test_start] + pd.DateOffset(months=test_months) if test_start < n else None
        while test_end < n and (test_end_date is not None and unique_dates[test_end] < test_end_date):
            test_end += 1
        if test_start >= n or test_end - test_start < 10:
            break
        
        # Get indices for train/test
        train_mask = (dates >= unique_dates[start]) & (dates < unique_dates[train_end])
        test_mask = (dates >= unique_dates[test_start]) & (dates < unique_dates[test_end])
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        test_returns = returns.loc[test_mask]
        
        # Train model
        if model_type == 'ensemble':
            model = create_ensemble_model(best_params)
        elif model_type == 'xgb' and XGBOOST_AVAILABLE:
            if best_params:
                model = xgb.XGBClassifier(**best_params)
            else:
                model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                                        tree_method='hist', use_label_encoder=False,
                                        eval_metric='logloss', verbosity=0)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        risk_metrics = calculate_risk_metrics(y_test, y_pred, test_returns)
        
        print(f"Fold {fold}: {X_train.shape[0]} train, {X_test.shape[0]} test, "
              f"accuracy={acc:.4f}, sharpe={risk_metrics['sharpe_ratio']:.3f}, "
              f"win_rate={risk_metrics['win_rate']:.3f}")
        
        results.append(acc)
        risk_metrics_list.append(risk_metrics)
        start = test_start
        fold += 1
    
    # Aggregate results
    avg_accuracy = np.mean(results)
    avg_sharpe = np.mean([m['sharpe_ratio'] for m in risk_metrics_list])
    avg_win_rate = np.mean([m['win_rate'] for m in risk_metrics_list])
    avg_max_drawdown = np.mean([m['max_drawdown'] for m in risk_metrics_list])
    
    print(f"\nWalk-forward Results:")
    print(f"Mean Accuracy: {avg_accuracy:.4f} (+/- {np.std(results):.4f})")
    print(f"Mean Sharpe Ratio: {avg_sharpe:.3f}")
    print(f"Mean Win Rate: {avg_win_rate:.3f}")
    print(f"Mean Max Drawdown: {avg_max_drawdown:.3f}")
    
    return results, risk_metrics_list

def train_and_evaluate_model(X, y):
    """Train and evaluate the model, using XGBoost if available, else RandomForest."""
    print("\nTraining and evaluating model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use XGBoost if available
    if XGBOOST_AVAILABLE:
        print("Using XGBoost (XGBClassifier)")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=1
        )
    else:
        print("Using RandomForestClassifier (XGBoost not available)")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    if XGBOOST_AVAILABLE:
        importances = model.feature_importances_
        features = X.columns
    else:
        importances = model.feature_importances_
        features = X.columns
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, accuracy, cv_scores.mean()

def select_top_features(X, y, top_n=15):
    """Select top N features using XGBoost or RandomForest importances."""
    print(f"\nSelecting top {top_n} features using model importances...")
    if XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(top_n))
    top_features = feature_importance['feature'].iloc[:top_n].tolist()
    return top_features

def walk_forward_backtest(X, y, dates, model_type='xgb', train_years=2, test_months=3):
    """Walk-forward backtest: train on N years, test on next M months, rolling forward."""
    print(f"\nStarting walk-forward backtest: {train_years} years train, {test_months} months test...")
    results = []
    unique_dates = pd.Series(dates).sort_values().unique()
    start = 0
    n = len(unique_dates)
    fold = 1
    while True:
        # Find train/test split indices
        train_end = start
        train_start_date = unique_dates[start]
        train_end_date = train_start_date + pd.DateOffset(years=train_years)
        while train_end < n and unique_dates[train_end] < train_end_date:
            train_end += 1
        test_start = train_end
        test_end = test_start
        test_end_date = unique_dates[test_start] + pd.DateOffset(months=test_months) if test_start < n else None
        while test_end < n and (test_end_date is not None and unique_dates[test_end] < test_end_date):
            test_end += 1
        if test_start >= n or test_end - test_start < 10:
            break
        # Get indices for train/test
        train_mask = (dates >= unique_dates[start]) & (dates < unique_dates[train_end])
        test_mask = (dates >= unique_dates[test_start]) & (dates < unique_dates[test_end])
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        if XGBOOST_AVAILABLE and model_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method='hist',
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Fold {fold}: {X_train.shape[0]} train, {X_test.shape[0]} test, accuracy={acc:.4f}")
        results.append(acc)
        start = test_start
        fold += 1
    print(f"\nWalk-forward mean accuracy: {np.mean(results):.4f} (+/- {np.std(results):.4f})")
    return results

def main():
    print("=== Enhanced Comprehensive Macro Testing ===")
    
    # Load macro data
    macro_df = load_macro_data()
    
    # Load stock data
    stock_df = load_stock_data()
    
    if stock_df is None:
        print("Failed to load stock data. Exiting.")
        return
    
    # Create features with macro data
    df_features, feature_cols = create_features_with_macro(stock_df, macro_df)
    
    if df_features is None or len(df_features) == 0:
        print("No valid samples after feature creation. Exiting.")
        return
    
    # Prepare X and y
    X = df_features[feature_cols]
    y = df_features['Target']
    dates = df_features.index
    returns = df_features['Returns']
    
    print(f"\nFinal dataset:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Feature selection
    top_features = select_top_features(X, y, top_n=15)
    X_top = X[top_features]
    
    # Enhanced walk-forward backtest with ensemble
    enhanced_walk_forward_backtest(X_top, y, dates, returns, model_type='ensemble', train_years=2, test_months=3)
    
    # Sector analysis
    analyze_sector_performance(df_features, top_features)
    
    # Train and evaluate on all data for reference
    model, accuracy, cv_accuracy = train_and_evaluate_model(X_top, y)
    
    # Save the trained model
    try:
        joblib.dump(model, 'final_macro_test_model.pkl')
        print(f"Model saved as 'final_macro_test_model.pkl'")
    except Exception as e:
        print(f"Warning: Could not save model: {e}")
    
    print(f"\n=== Enhanced Results Summary ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Cross-validation Accuracy: {cv_accuracy:.4f}")
    print(f"Number of samples: {len(X_top)}")
    print(f"Number of features: {len(top_features)}")
    print(f"Macro features used: {[f for f in top_features if f in macro_df.columns]}")

if __name__ == "__main__":
    main() 