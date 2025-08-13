import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def calculate_market_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced market microstructure features for improved prediction accuracy.
    
    These features capture:
    - Order flow dynamics
    - Liquidity measures
    - Market efficiency indicators
    - Price impact measures
    - Volatility clustering
    """
    
    df = df.copy()
    
    # === PRICE-BASED MICROSTRUCTURE FEATURES ===
    
    # Realized volatility (high-frequency)
    df['Realized_Vol_1d'] = df['Close'].pct_change().rolling(1).std()
    df['Realized_Vol_5d'] = df['Close'].pct_change().rolling(5).std()
    df['Realized_Vol_20d'] = df['Close'].pct_change().rolling(20).std()
    
    # Volatility clustering
    df['Vol_Clustering'] = df['Realized_Vol_1d'].rolling(5).autocorr()
    
    # Price efficiency (how quickly prices reflect information)
    df['Price_Efficiency'] = 1 - abs(df['Close'].pct_change().rolling(20).autocorr())
    
    # Price momentum at different frequencies
    df['Momentum_1d'] = df['Close'] / df['Close'].shift(1) - 1
    df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Momentum acceleration
    df['Momentum_Accel'] = df['Momentum_5d'] - df['Momentum_20d']
    
    # === VOLUME-BASED MICROSTRUCTURE FEATURES ===
    
    # Volume-weighted average price (VWAP)
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # Volume profile
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Volume-price trend (VPT)
    df['VPT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * df['Volume']
    df['VPT_Cumulative'] = df['VPT'].cumsum()
    
    # Money flow index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + positive_flow / negative_flow))
    
    # === LIQUIDITY MEASURES ===
    
    # Amihud illiquidity measure
    df['Amihud_Illiquidity'] = abs(df['Close'].pct_change()) / df['Volume']
    df['Amihud_MA'] = df['Amihud_Illiquidity'].rolling(20).mean()
    
    # Roll's implicit spread estimator
    df['Roll_Spread'] = 2 * np.sqrt(-df['Close'].pct_change().rolling(5).autocorr()) * df['Close']
    
    # Kyle's lambda (price impact)
    df['Kyle_Lambda'] = abs(df['Close'].pct_change()) / df['Volume'].rolling(5).mean()
    
    # === ORDER FLOW FEATURES ===
    
    # Buy/sell pressure indicators
    df['Buy_Pressure'] = np.where(df['Close'] > df['Open'], df['Volume'], 0)
    df['Sell_Pressure'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)
    df['Pressure_Ratio'] = df['Buy_Pressure'].rolling(5).sum() / (df['Sell_Pressure'].rolling(5).sum() + 1e-8)
    
    # Gap analysis
    df['Gap_Size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['Gap_Fill'] = np.where(
        (df['Gap_Size'] > 0) & (df['Low'] <= df['Close'].shift(1)), 1,
        np.where((df['Gap_Size'] < 0) & (df['High'] >= df['Close'].shift(1)), 1, 0)
    )
    
    # === MARKET REGIME FEATURES ===
    
    # Trend strength
    df['Trend_Strength'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
    
    # Market regime classification
    df['Market_Regime'] = pd.cut(df['Trend_Strength'], 
                                bins=[-np.inf, -1, 1, np.inf], 
                                labels=['bear', 'sideways', 'bull'])
    
    # Regime persistence
    df['Regime_Persistence'] = (df['Market_Regime'] == df['Market_Regime'].shift(1)).rolling(10).sum()
    
    # === ADVANCED TECHNICAL INDICATORS ===
    
    # Williams %R
    df['Williams_R'] = ((df['High'].rolling(14).max() - df['Close']) / 
                        (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * -100
    
    # Commodity Channel Index (CCI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Parabolic SAR (simplified)
    df['PSAR'] = df['Close'].rolling(5).min()
    
    # Ichimoku Cloud components
    df['Tenkan_sen'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun_sen'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    df['Senkou_Span_B'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    
    # === CROSS-SECTIONAL FEATURES ===
    
    # Price relative to moving averages
    df['Price_vs_MA5'] = df['Close'] / df['Close'].rolling(5).mean() - 1
    df['Price_vs_MA20'] = df['Close'] / df['Close'].rolling(20).mean() - 1
    df['Price_vs_MA50'] = df['Close'] / df['Close'].rolling(50).mean() - 1
    
    # Moving average crossovers
    df['MA5_vs_MA20'] = df['Close'].rolling(5).mean() / df['Close'].rolling(20).mean() - 1
    df['MA20_vs_MA50'] = df['Close'].rolling(20).mean() / df['Close'].rolling(50).mean() - 1
    
    # === VOLATILITY FEATURES ===
    
    # Parkinson volatility (uses high-low range)
    df['Parkinson_Vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        ((np.log(df['High'] / df['Low']) ** 2).rolling(20).mean())
    )
    
    # Garman-Klass volatility
    df['Garman_Klass_Vol'] = np.sqrt(
        (0.5 * (np.log(df['High'] / df['Low']) ** 2) - 
         (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)).rolling(20).mean()
    )
    
    # === MOMENTUM AND MEAN REVERSION ===
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic oscillator
    df['Stoch_K'] = ((df['Close'] - df['Low'].rolling(14).min()) / 
                     (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # === SUPPORT AND RESISTANCE ===
    
    # Pivot points
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low']
    df['S1'] = 2 * df['Pivot'] - df['High']
    
    # Support and resistance levels
    df['Support_Level'] = df['Low'].rolling(20).min()
    df['Resistance_Level'] = df['High'].rolling(20).max()
    df['Price_vs_Support'] = (df['Close'] - df['Support_Level']) / df['Close']
    df['Price_vs_Resistance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
    
    # === FEATURE INTERACTIONS ===
    
    # Volume-volatility interaction
    df['Vol_Vol_Interaction'] = df['Volume_Ratio'] * df['Realized_Vol_5d']
    
    # Price-momentum interaction
    df['Price_Momentum_Interaction'] = df['Price_vs_MA20'] * df['Momentum_5d']
    
    # Liquidity-trend interaction
    df['Liquidity_Trend_Interaction'] = df['Amihud_MA'] * df['Trend_Strength']
    
    return df

def calculate_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market regime-specific features."""
    
    df = df.copy()
    
    # Market regime detection
    returns = df['Close'].pct_change()
    
    # Volatility regime
    vol_ma = returns.rolling(20).std()
    vol_regime = pd.cut(vol_ma, 
                       bins=[0, vol_ma.quantile(0.33), vol_ma.quantile(0.67), np.inf],
                       labels=['low_vol', 'medium_vol', 'high_vol'])
    df['Volatility_Regime'] = vol_regime
    
    # Trend regime
    trend_strength = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).std()
    trend_regime = pd.cut(trend_strength,
                         bins=[-np.inf, -1, 1, np.inf],
                         labels=['downtrend', 'sideways', 'uptrend'])
    df['Trend_Regime'] = trend_regime
    
    # Regime-specific features
    for regime in ['low_vol', 'medium_vol', 'high_vol']:
        mask = df['Volatility_Regime'] == regime
        df[f'Vol_Regime_{regime}'] = mask.astype(int)
    
    for regime in ['downtrend', 'sideways', 'uptrend']:
        mask = df['Trend_Regime'] == regime
        df[f'Trend_Regime_{regime}'] = mask.astype(int)
    
    return df

def calculate_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-based features (day of week, month, etc.)."""
    
    df = df.copy()
    
    # Day of week effects
    df['Day_of_Week'] = df.index.dayofweek
    for day in range(5):
        df[f'Day_{day}'] = (df['Day_of_Week'] == day).astype(int)
    
    # Month effects
    df['Month'] = df.index.month
    for month in range(1, 13):
        df[f'Month_{month}'] = (df['Month'] == month).astype(int)
    
    # Quarter effects
    df['Quarter'] = df.index.quarter
    for quarter in range(1, 5):
        df[f'Quarter_{quarter}'] = (df['Quarter'] == quarter).astype(int)
    
    # End of month/quarter effects
    df['End_of_Month'] = (df.index.day >= 25).astype(int)
    df['End_of_Quarter'] = ((df.index.month.isin([3, 6, 9, 12])) & (df.index.day >= 25)).astype(int)
    
    return df

def create_feature_interactions(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Create interaction features between selected pairs."""
    
    df = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df[interaction_name] = df[feat1] * df[feat2]
    
    return df

def select_optimal_features(df: pd.DataFrame, target: pd.Series, max_features: int = 50) -> List[str]:
    """
    Select optimal features using correlation and mutual information.
    
    Args:
        df: Feature DataFrame
        target: Target variable
        max_features: Maximum number of features to select
    
    Returns:
        List of selected feature names
    """
    
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['Target', 'Date']]
    X = df[feature_cols].fillna(0)
    y = target.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate feature importance using multiple methods
    # F-statistic
    f_scores = f_regression(X_scaled, y)[0]
    f_importance = pd.Series(f_scores, index=feature_cols)
    
    # Mutual information
    mi_scores = mutual_info_regression(X_scaled, y)
    mi_importance = pd.Series(mi_scores, index=feature_cols)
    
    # Combine scores
    combined_importance = (f_importance.rank() + mi_importance.rank()) / 2
    
    # Select top features
    selected_features = combined_importance.nlargest(max_features).index.tolist()
    
    return selected_features

def main():
    """Example usage of market microstructure features."""
    
    print("Market Microstructure Features Module")
    print("This module provides:")
    print("- Advanced price and volume analysis")
    print("- Liquidity measures (Amihud, Roll, Kyle)")
    print("- Order flow indicators")
    print("- Market regime detection")
    print("- Time-based features")
    print("- Feature selection and interactions")
    
    print("\nKey features include:")
    print("- Realized volatility measures")
    print("- Volume-weighted indicators")
    print("- Market efficiency metrics")
    print("- Support/resistance levels")
    print("- Advanced technical indicators")
    print("- Cross-sectional comparisons")

if __name__ == "__main__":
    main() 