import pandas as pd
import numpy as np
import ta  # For some advanced indicators
from sklearn.preprocessing import StandardScaler

# Basic option features (original)
BASIC_OPTION_FEATURES = [
    "CallVolume", "PutVolume", "CallOI", "PutOI", "CallIV", "PutIV"
]

# Comprehensive option features (new)
COMPREHENSIVE_OPTION_FEATURES = [
    # Volume and Open Interest
    "CallVolume", "PutVolume", "CallOI", "PutOI",
    "TotalVolume", "TotalOI", "PCR_Volume", "PCR_OI",
    
    # Implied Volatility
    "CallIV", "PutIV", "AvgIV", "IV_Skew",
    
    # Greeks (if available)
    "CallDelta", "PutDelta", "CallGamma", "PutGamma",
    "CallTheta", "PutTheta", "CallVega", "PutVega",
    
    # Strike Analysis
    "ATM_CallIV", "ATM_PutIV", "ITM_CallIV", "OTM_PutIV",
    "Strike_Range", "Strike_Count",
    
    # Market Sentiment
    "CallPut_Ratio", "IV_Premium", "Skew_Index",
    
    # Expiration Analysis
    "DaysToExpiry", "Spot_Price", "ATM_Strike"
]

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add several common technical indicators to the dataframe."""
    df = df.copy()

    # Simple moving averages
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    # Exponential moving averages
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Bollinger Bands
    mid = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = mid + 2 * std
    df["BB_Lower"] = mid - 2 * std

    # MACD and signal line
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["RSI_14"] = compute_rsi(df["Close"], window=14)

    # Stochastic Oscillator
    stoch_k, stoch_d = compute_stochastic(df, window=14)
    df["Stoch_%K"] = stoch_k
    df["Stoch_%D"] = stoch_d

    # Average True Range
    df["ATR_14"] = compute_atr(df, window=14)

    # Commodity Channel Index
    df["CCI_20"] = compute_cci(df, window=20)

    # On-Balance Volume
    df["OBV"] = compute_obv(df)

    return df

def add_option_features(df: pd.DataFrame, opt_df: pd.DataFrame, 
                       feature_type: str = "auto") -> pd.DataFrame:
    """Merge option-based features into the price dataframe with smart fallback."""
    opt_df = opt_df.copy()
    opt_df.index = pd.to_datetime(opt_df.index)
    
    # Determine which features to use
    if feature_type == "basic":
        features_to_use = BASIC_OPTION_FEATURES
    elif feature_type == "comprehensive":
        features_to_use = COMPREHENSIVE_OPTION_FEATURES
    else:  # auto - detect based on available columns
        available_features = [col for col in COMPREHENSIVE_OPTION_FEATURES if col in opt_df.columns]
        if len(available_features) > len(BASIC_OPTION_FEATURES):
            features_to_use = available_features
        else:
            features_to_use = [col for col in BASIC_OPTION_FEATURES if col in opt_df.columns]
    
    # Filter to only available features
    available_features = [col for col in features_to_use if col in opt_df.columns]
    
    if not available_features:
        print("Warning: No option features available, proceeding without options data")
        return df
    
    print(f"Using {len(available_features)} option features: {available_features[:5]}...")
    
    # If options data has only one row, expand it to match price data timeframe
    if len(opt_df) == 1:
        print("Single row options data detected - expanding to match price data timeframe")
        # Create a new dataframe with the same index as price data
        expanded_opt_df = pd.DataFrame(index=df.index)
        for col in available_features:
            expanded_opt_df[col] = opt_df[col].iloc[0]
    else:
        expanded_opt_df = opt_df[available_features]
    
    # Join the dataframes
    df = df.join(expanded_opt_df, how="left")
    
    # Fill missing values with forward fill, then backward fill
    df[available_features] = df[available_features].ffill().bfill()
    
    # For any remaining NaN values, fill with 0
    df[available_features] = df[available_features].fillna(0)
    
    return df

def add_derived_options_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features from existing options data."""
    df = df.copy()
    
    # PCR (Put-Call Ratio) derived features
    if 'CallVolume' in df.columns and 'PutVolume' in df.columns:
        df['PCR_Volume_MA5'] = (df['PutVolume'] / df['CallVolume']).rolling(5).mean()
        df['PCR_Volume_MA20'] = (df['PutVolume'] / df['CallVolume']).rolling(20).mean()
    
    if 'CallOI' in df.columns and 'PutOI' in df.columns:
        df['PCR_OI_MA5'] = (df['PutOI'] / df['CallOI']).rolling(5).mean()
        df['PCR_OI_MA20'] = (df['PutOI'] / df['CallOI']).rolling(20).mean()
    
    # IV derived features
    if 'CallIV' in df.columns and 'PutIV' in df.columns:
        df['IV_Spread'] = df['PutIV'] - df['CallIV']
        df['IV_Spread_MA5'] = df['IV_Spread'].rolling(5).mean()
        df['IV_Spread_MA20'] = df['IV_Spread'].rolling(20).mean()
    
    if 'AvgIV' in df.columns:
        df['IV_MA5'] = df['AvgIV'].rolling(5).mean()
        df['IV_MA20'] = df['AvgIV'].rolling(20).mean()
        df['IV_Volatility'] = df['AvgIV'].rolling(20).std()
    
    # Volume derived features
    if 'TotalVolume' in df.columns:
        df['Options_Volume_MA5'] = df['TotalVolume'].rolling(5).mean()
        df['Options_Volume_MA20'] = df['TotalVolume'].rolling(20).mean()
        df['Options_Volume_Ratio'] = df['TotalVolume'] / df['Volume']
    
    # Greeks derived features
    if 'CallDelta' in df.columns and 'PutDelta' in df.columns:
        df['Net_Delta'] = df['CallDelta'] + df['PutDelta']
        df['Delta_Exposure'] = abs(df['CallDelta']) + abs(df['PutDelta'])
    
    if 'CallGamma' in df.columns and 'PutGamma' in df.columns:
        df['Total_Gamma'] = df['CallGamma'] + df['PutGamma']
    
    # Market sentiment indicators
    if 'Skew_Index' in df.columns:
        df['Skew_MA5'] = df['Skew_Index'].rolling(5).mean()
        df['Skew_MA20'] = df['Skew_Index'].rolling(20).mean()
    
    # Expiration effects
    if 'DaysToExpiry' in df.columns:
        df['Expiry_Effect'] = 1 / (df['DaysToExpiry'] + 1)  # Inverse relationship
        df['Near_Expiry'] = (df['DaysToExpiry'] <= 7).astype(int)
        df['Monthly_Expiry'] = (df['DaysToExpiry'] <= 30).astype(int)
    
    return df

def get_available_option_features(df: pd.DataFrame) -> list:
    """Get list of available option features in the dataframe."""
    all_option_features = BASIC_OPTION_FEATURES + COMPREHENSIVE_OPTION_FEATURES
    return [col for col in all_option_features if col in df.columns]

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_stochastic(df: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series]:
    low_min = df["Low"].rolling(window=window).min()
    high_max = df["High"].rolling(window=window).max()
    k_percent = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=3).mean()
    return k_percent, d_percent

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()
    return atr

def compute_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

def compute_obv(df: pd.DataFrame) -> pd.Series:
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def add_advanced_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ADX
    try:
        df['ADX_14'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14, fillna=True)
    except Exception:
        df['ADX_14'] = np.nan
    # Rate of Change
    df['ROC_10'] = df['Close'].pct_change(periods=10)
    # Williams %R
    df['Williams_%R_14'] = (df['High'].rolling(14).max() - df['Close']) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) * -100
    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    # Money Flow Index
    try:
        df['MFI_14'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14, fillna=True)
    except Exception:
        df['MFI_14'] = np.nan
    # True Strength Index
    try:
        df['TSI_25_13'] = ta.momentum.tsi(df['Close'], window_slow=25, window_fast=13, fillna=True)
    except Exception:
        df['TSI_25_13'] = np.nan
    return df

def add_lagged_features(df: pd.DataFrame, columns: list, lags: list = [1,2,3]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, columns: list, windows: list = [5, 10, 20]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for win in windows:
            df[f'{col}_roll{win}_mean'] = df[col].rolling(win).mean()
            df[f'{col}_roll{win}_std'] = df[col].rolling(win).std()
            df[f'{col}_roll{win}_min'] = df[col].rolling(win).min()
            df[f'{col}_roll{win}_max'] = df[col].rolling(win).max()
            df[f'{col}_roll{win}_median'] = df[col].rolling(win).median()
    return df

def add_rolling_correlation(df: pd.DataFrame, col1: str, col2: str, windows: list = [5, 10, 20]) -> pd.DataFrame:
    df = df.copy()
    for win in windows:
        df[f'{col1}_{col2}_roll{win}_corr'] = df[col1].rolling(win).corr(df[col2])
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example crosses and ratios
    if 'RSI_14' in df.columns and 'Volume' in df.columns:
        df['RSI_14_x_Volume'] = df['RSI_14'] * df['Volume']
    if 'MACD' in df.columns and 'ATR_14' in df.columns:
        df['MACD_x_ATR_14'] = df['MACD'] * df['ATR_14']
    if 'Close' in df.columns and 'Open' in df.columns:
        df['Close_Open_ratio'] = df['Close'] / df['Open']
    if 'High' in df.columns and 'Low' in df.columns:
        df['High_Low_ratio'] = df['High'] / df['Low']
    return df

def add_log_zscore_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if (df[col] > 0).all():
            df[f'{col}_log'] = np.log(df[col])
        # Z-score
        scaler = StandardScaler()
        df[f'{col}_zscore'] = scaler.fit_transform(df[[col]].fillna(0))
    return df

def create_feature_matrix(df: pd.DataFrame, include_options: bool = True) -> tuple[pd.DataFrame, list]:
    """Create feature matrix with all available features."""
    # Technical indicators
    df = add_technical_indicators(df)
    
    # Option features (if available and requested)
    if include_options:
        option_features = get_available_option_features(df)
        if option_features:
            df = add_derived_options_features(df)
            # Get updated list after adding derived features
            option_features = get_available_option_features(df)
    
    # Define feature columns
    technical_features = [
        'MA_20', 'MA_50', 'EMA_20', 'EMA_50',
        'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
        'RSI_14', 'Stoch_%K', 'Stoch_%D', 'ATR_14',
        'CCI_20', 'OBV'
    ]
    
    option_features = get_available_option_features(df) if include_options else []
    
    # Filter to only available features
    available_technical = [col for col in technical_features if col in df.columns]
    available_options = [col for col in option_features if col in df.columns]
    
    all_features = available_technical + available_options
    
    print(f"Available technical features: {len(available_technical)}")
    print(f"Available option features: {len(available_options)}")
    print(f"Total features: {len(all_features)}")
    
    # Clean up dataframe
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df, all_features

def create_target(df: pd.DataFrame, target_period: int = 1) -> pd.Series:
    """Creates the binary classification target."""
    df['Target'] = (df['Close'].shift(-target_period) > df['Close']).astype(int)
    return df['Target']

def load_and_engineer_features(
    price_path: str, 
    options_path: str = None, 
    target_period: int = 1,
    use_comprehensive_options: bool = False
) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Main function to load data, engineer all features, and create the target variable.
    """
    # Load price data
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)

    # Load options data if provided
    opt_df = None
    if options_path:
        opt_df = pd.read_csv(options_path, index_col=0, parse_dates=True)

    # Add technical indicators
    featured_df = add_technical_indicators(price_df)
    featured_df = add_advanced_technical_indicators(featured_df)

    # Add options features if applicable
    if opt_df is not None:
        feature_type = "comprehensive" if use_comprehensive_options else "basic"
        featured_df = add_option_features(featured_df, opt_df, feature_type=feature_type)
        featured_df = add_derived_options_features(featured_df)

    # Add lagged features
    lag_cols = ['Close', 'Volume', 'RSI_14', 'MACD', 'ATR_14']
    featured_df = add_lagged_features(featured_df, [col for col in lag_cols if col in featured_df.columns])

    # Add rolling features
    roll_cols = ['Close', 'Volume', 'RSI_14', 'MACD', 'ATR_14']
    featured_df = add_rolling_features(featured_df, [col for col in roll_cols if col in featured_df.columns])

    # Add rolling correlation
    if 'Close' in featured_df.columns and 'Volume' in featured_df.columns:
        featured_df = add_rolling_correlation(featured_df, 'Close', 'Volume')

    # Add interaction features
    featured_df = add_interaction_features(featured_df)

    # Add log/z-score features
    log_z_cols = ['Close', 'Volume', 'RSI_14', 'MACD', 'ATR_14']
    featured_df = add_log_zscore_features(featured_df, [col for col in log_z_cols if col in featured_df.columns])

    # Define all feature columns
    technical_features = [
        'MA_20', 'MA_50', 'EMA_20', 'EMA_50', 'BB_Upper', 'BB_Lower',
        'MACD', 'MACD_Signal', 'RSI_14', 'Stoch_%K', 'Stoch_%D',
        'ATR_14', 'CCI_20', 'OBV', 'ADX_14', 'ROC_10', 'Williams_%R_14',
        'Momentum_10', 'MFI_14', 'TSI_25_13'
    ]
    
    # Add new features
    lagged = [col for col in featured_df.columns if 'lag' in col]
    rolling = [col for col in featured_df.columns if 'roll' in col]
    corr = [col for col in featured_df.columns if 'corr' in col]
    interaction = [col for col in featured_df.columns if '_x_' in col or '_ratio' in col]
    logz = [col for col in featured_df.columns if col.endswith('_log') or col.endswith('_zscore')]
    option_cols = get_available_option_features(featured_df)
    
    # Add macro features if available
    macro_features = [
        'FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI',
        'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 
        'IndustrialProduction', 'RetailSales', 'ConsumerConfidence', 'ISM_Manufacturing',
        'VIX', 'SP500'
    ]
    available_macro = [col for col in macro_features if col in featured_df.columns]
    
    feature_names = technical_features + lagged + rolling + corr + interaction + logz + option_cols + available_macro

    # Create target variable
    target = create_target(featured_df, target_period)
    
    # Align data by dropping NaNs from features and target
    combined = featured_df[feature_names].join(target).dropna()
    
    X = combined[feature_names]
    y = combined['Target']
    
    # Ensure feature names in X match the list
    final_feature_names = X.columns.tolist()

    return X, y, final_feature_names 