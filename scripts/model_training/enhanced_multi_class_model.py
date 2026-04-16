import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced features module
from enhanced_features import create_feature_matrix, add_option_features

def create_multi_class_target(df: pd.DataFrame, method: str = "three_class") -> pd.Series:
    """Create multi-class target variable for more nuanced predictions."""
    if method == "three_class":
        # Three classes: 0=sell, 1=hold, 2=buy
        price_change = df['Close'].shift(-1) / df['Close'] - 1
        
        # Define thresholds based on volatility
        volatility = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
        dynamic_threshold = volatility * 0.5  # Adaptive threshold
        
        # Create labels
        df['Target'] = pd.cut(price_change, 
                             bins=[-np.inf, -dynamic_threshold, dynamic_threshold, np.inf], 
                             labels=[0, 1, 2]).astype(int)
        
    elif method == "five_class":
        # Five classes: strong sell, sell, hold, buy, strong buy
        price_change = df['Close'].shift(-1) / df['Close'] - 1
        volatility = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
        threshold = volatility * 0.5
        
        df['Target'] = pd.cut(price_change, 
                             bins=[-np.inf, -threshold*2, -threshold, threshold, threshold*2, np.inf], 
                             labels=[0, 1, 2, 3, 4]).astype(int)
    
    elif method == "momentum_based":
        # Based on momentum and trend
        short_ma = df['Close'].rolling(5).mean()
        long_ma = df['Close'].rolling(20).mean()
        momentum = (short_ma - long_ma) / long_ma
        
        # Combine price change and momentum
        price_change = df['Close'].shift(-1) / df['Close'] - 1
        combined_signal = price_change + momentum * 0.5
        
        df['Target'] = pd.cut(combined_signal, 
                             bins=[-np.inf, -0.02, 0.02, np.inf], 
                             labels=[0, 1, 2]).astype(int)
    
    return df['Target']

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced features that improve prediction accuracy."""
    df = df.copy()
    
    # Price-based features
    df['Price_Change_1d'] = df['Close'].pct_change(1)
    df['Price_Change_5d'] = df['Close'].pct_change(5)
    df['Price_Change_20d'] = df['Close'].pct_change(20)
    
    # Volatility features
    df['Volatility_5d'] = df['Close'].rolling(5).std() / df['Close'].rolling(5).mean()
    df['Volatility_20d'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
    df['Volatility_Ratio'] = df['Volatility_5d'] / df['Volatility_20d']
    
    # Trend features
    df['Trend_Strength'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    df['Trend_Direction'] = np.where(df['Close'] > df['Close'].rolling(20).mean(), 1, -1)
    
    # Momentum features
    df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
    df['Momentum_Acceleration'] = df['Momentum_5d'] - df['Momentum_20d']
    
    # Support/Resistance features
    df['Support_Level'] = df['Low'].rolling(20).min()
    df['Resistance_Level'] = df['High'].rolling(20).max()
    df['Price_vs_Support'] = (df['Close'] - df['Support_Level']) / df['Close']
    df['Price_vs_Resistance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
    
    # Volume-based features
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    df['Volume_Price_Trend'] = df['Volume'] * df['Price_Change_1d']
    
    # Gap analysis
    df['Gap_Up'] = np.where(df['Open'] > df['Close'].shift(1), 1, 0)
    df['Gap_Down'] = np.where(df['Open'] < df['Close'].shift(1), 1, 0)
    df['Gap_Size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Market regime features
    df['Bull_Market'] = np.where(df['Close'] > df['Close'].rolling(50).mean(), 1, 0)
    df['Bear_Market'] = np.where(df['Close'] < df['Close'].rolling(50).mean(), 1, 0)
    df['Sideways_Market'] = np.where((df['Close'] > df['Close'].rolling(50).mean() * 0.95) & 
                                    (df['Close'] < df['Close'].rolling(50).mean() * 1.05), 1, 0)
    
    # Advanced technical indicators
    # Williams %R
    df['Williams_R'] = ((df['High'].rolling(14).max() - df['Close']) / 
                        (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * -100
    
    # Money Flow Index
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + positive_flow / negative_flow))
    
    # Parabolic SAR (simplified)
    df['PSAR'] = df['Close'].rolling(5).min()  # Simplified version
    
    # Ichimoku Cloud components
    df['Tenkan_sen'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun_sen'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['Ichimoku_Signal'] = np.where(df['Tenkan_sen'] > df['Kijun_sen'], 1, -1)
    
    return df

def create_ensemble_model():
    """Create an ensemble of multiple models for better accuracy."""
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
    }
    return models

def train_ensemble_model(df: pd.DataFrame, features: list, target: pd.Series):
    """Train ensemble model with cross-validation."""
    # Prepare data
    valid_data = df[features + ['Target']].dropna()
    
    if len(valid_data) < 200:
        raise ValueError(f"Insufficient data: only {len(valid_data)} samples available")
    
    X = valid_data[features]
    y = valid_data['Target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train ensemble models
    models = create_ensemble_model()
    trained_models = {}
    cv_scores = {}
    
    for name, model in models.items():
        scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            scores.append(score)
        
        cv_scores[name] = np.mean(scores)
        trained_models[name] = model
    
    # Train final models on full dataset
    for name, model in trained_models.items():
        model.fit(X_scaled, y)
    
    return trained_models, scaler, cv_scores

def main():
    parser = argparse.ArgumentParser(description="Enhanced multi-class ensemble model")
    parser.add_argument("price_file", help="Price data CSV file")
    parser.add_argument("--options", help="Options data CSV file")
    parser.add_argument("--model-out", default="enhanced_ensemble_model.pkl", help="Output model file")
    parser.add_argument("--target-method", choices=["three_class", "five_class", "momentum_based"], 
                       default="three_class", help="Target variable creation method")
    
    args = parser.parse_args()
    
    try:
        print("Loading and preparing data...")
        df = pd.read_csv(args.price_file, index_col=0, parse_dates=True)
        
        # Add options data if available
        if args.options:
            print("Adding options data...")
            opt_df = pd.read_csv(args.options, index_col=0, parse_dates=True)
            df = add_option_features(df, opt_df, "auto")
        
        # Create feature matrix
        include_options = args.options is not None
        df, features = create_feature_matrix(df, include_options=include_options)
        
        # Add advanced features
        print("Adding advanced features...")
        df = add_advanced_features(df)
        
        # Update features list
        new_features = [col for col in df.columns if col not in ['Target'] and col not in features]
        all_features = features + new_features
        
        # Create multi-class target
        print(f"Creating {args.target_method} target variable...")
        target = create_multi_class_target(df, args.target_method)
        
        print(f"Training ensemble model with {len(all_features)} features...")
        models, scaler, cv_scores = train_ensemble_model(df, all_features, target)
        
        # Save model and metadata
        model_data = {
            'models': models,
            'scaler': scaler,
            'features': all_features,
            'target_method': args.target_method,
            'cv_scores': cv_scores,
            'training_date': pd.Timestamp.now()
        }
        
        joblib.dump(model_data, args.model_out)
        
        print(f"\n=== Enhanced Ensemble Model Results ===")
        print(f"Model saved to: {args.model_out}")
        print(f"Features: {len(all_features)}")
        print(f"Target method: {args.target_method}")
        print(f"Cross-validation scores: {cv_scores}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 