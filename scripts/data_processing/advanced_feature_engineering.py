import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(d_period).mean()
    return k_percent, d_percent

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma_tp) / (0.015 * mad)
    return cci

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = pd.Series(0, index=typical_price.index)
    negative_flow = pd.Series(0, index=typical_price.index)
    
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = money_flow.iloc[i]
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow.iloc[i] = money_flow.iloc[i]
    
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for financial time series data.
    
    Features:
    - Comprehensive technical indicators
    - Market microstructure features
    - Volatility and risk measures
    - Machine learning-based features
    - Multi-timeframe analysis
    - Options-based features (when available)
    """
    
    def __init__(self,
                 price_data: pd.DataFrame,
                 options_data: Optional[pd.DataFrame] = None,
                 lookback_periods: List[int] = None,
                 volatility_windows: List[int] = None):
        
        self.price_data = price_data.copy()
        self.options_data = options_data
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100]
        self.volatility_windows = volatility_windows or [5, 10, 20, 50]
        self.features = {}
        self.feature_importance = {}
        
    def engineer_features(self) -> pd.DataFrame:
        """Engineer all features."""
        logger.info("Starting feature engineering...")
        
        # Start with a copy of the original data
        df = self.price_data.copy()
        
        # Add all feature categories in the correct order
        self._add_basic_technical_indicators_to_df(df)
        logger.info("Added basic technical indicators")
        
        self._add_advanced_technical_indicators_to_df(df)
        logger.info("Added advanced technical indicators")
        
        self._add_volatility_features_to_df(df)
        logger.info("Added volatility features")
        
        # Now that returns are calculated, add microstructure features
        self._add_market_microstructure_features_to_df(df)
        logger.info("Added market microstructure features")
        
        self._add_machine_learning_features_to_df(df)
        logger.info("Added machine learning features")
        
        self._add_options_features_to_df(df)
        logger.info("Added options features")
        
        self._add_multi_timeframe_features_to_df(df)
        logger.info("Added multi-timeframe features")
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Store feature information
        self.features = {
            'total_features': len(df.columns),
            'price_features': len([col for col in df.columns if col in ['Open', 'High', 'Low', 'Close', 'Volume']]),
            'technical_features': len([col for col in df.columns if any(x in col for x in ['sma_', 'ema_', 'rsi_', 'macd_', 'bb_', 'stoch_'])]),
            'volatility_features': len([col for col in df.columns if 'volatility' in col or 'var_' in col or 'drawdown' in col]),
            'microstructure_features': len([col for col in df.columns if any(x in col for x in ['spread_', 'vwap_', 'volume_', 'amihud_', 'efficiency_'])]),
            'ml_features': len([col for col in df.columns if any(x in col for x in ['pca_', 'momentum_', 'trend_', 'regime_'])]),
            'options_features': len([col for col in df.columns if any(x in col for x in ['iv_', 'options_', 'put_call_'])]),
            'multi_timeframe_features': len([col for col in df.columns if any(x in col for x in ['_1w', '_1m', '_3m'])])
        }
        
        logger.info(f"Feature engineering completed. Total features: {self.features['total_features']}")
        return df
    
    def _add_basic_technical_indicators_to_df(self, df: pd.DataFrame):
        """Add comprehensive technical indicators to existing DataFrame."""
        
        # Moving averages
        for period in self.lookback_periods:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_sma_ratio_{period}'] = df['Close'] / df[f'sma_{period}']
            df[f'price_ema_ratio_{period}'] = df['Close'] / df[f'ema_{period}']
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'], period)
            df[f'bb_upper_{period}'] = bb_upper
            df[f'bb_middle_{period}'] = bb_middle
            df[f'bb_lower_{period}'] = bb_lower
            df[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
            df[f'bb_position_{period}'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # RSI
        for period in [14, 21]:
            df[f'rsi_{period}'] = calculate_rsi(df['Close'], period)
        
        # MACD
        macd, macd_signal, macd_hist = calculate_macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        df['macd_cross'] = np.where(macd > macd_signal, 1, -1)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        df['stoch_cross'] = np.where(stoch_k > stoch_d, 1, -1)
        
        # Williams %R
        df['williams_r'] = calculate_williams_r(df['High'], df['Low'], df['Close'])
        
        # Commodity Channel Index
        df['cci'] = calculate_cci(df['High'], df['Low'], df['Close'])
        
        # Average True Range
        df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'])
        df['atr_ratio'] = df['atr'] / df['Close']
        
        # On-Balance Volume
        df['obv'] = calculate_obv(df['Close'], df['Volume'])
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_ma']
        
        # Money Flow Index
        df['mfi'] = calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])
    
    def _add_advanced_technical_indicators_to_df(self, df: pd.DataFrame):
        """Add advanced technical indicators to existing DataFrame."""
        
        # Ichimoku Cloud
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        high_52 = df['High'].rolling(52).max()
        low_52 = df['Low'].rolling(52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Fibonacci Retracements
        for period in [20, 50]:
            high = df['High'].rolling(period).max()
            low = df['Low'].rolling(period).min()
            diff = high - low
            
            df[f'fib_236_{period}'] = high - 0.236 * diff
            df[f'fib_382_{period}'] = high - 0.382 * diff
            df[f'fib_500_{period}'] = high - 0.500 * diff
            df[f'fib_618_{period}'] = high - 0.618 * diff
            df[f'fib_786_{period}'] = high - 0.786 * diff
        
        # Pivot Points
        df['pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['r1'] = 2 * df['pivot'] - df['Low']
        df['s1'] = 2 * df['pivot'] - df['High']
        df['r2'] = df['pivot'] + (df['High'] - df['Low'])
        df['s2'] = df['pivot'] - (df['High'] - df['Low'])
        
        # Volume-based indicators
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['volume_price_trend'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
        df['volume_price_trend_ma'] = df['volume_price_trend'].rolling(20).mean()
        
        # Price action patterns
        df['doji'] = np.abs(df['Open'] - df['Close']) <= (df['High'] - df['Low']) * 0.1
        df['hammer'] = (df['Low'] < df['Open']) & (df['Low'] < df['Close']) & \
                      ((df['High'] - df['Low']) > 3 * (df['Open'] - df['Low']))
        df['shooting_star'] = (df['High'] > df['Open']) & (df['High'] > df['Close']) & \
                             ((df['High'] - df['Low']) > 3 * (df['High'] - df['Close']))
    
    def _add_volatility_features_to_df(self, df: pd.DataFrame):
        """Add volatility and risk measures to existing DataFrame."""
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility measures
        for window in self.volatility_windows:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'log_volatility_{window}'] = df['log_returns'].rolling(window).std()
            df[f'realized_volatility_{window}'] = np.sqrt((df['returns']**2).rolling(window).sum())
        
        # GARCH-like volatility (simplified)
        for window in [20, 50]:
            returns_squared = df['returns']**2
            df[f'garch_vol_{window}'] = np.sqrt(returns_squared.ewm(alpha=0.06).mean())
        
        # Risk measures
        for window in [20, 50]:
            returns_window = df['returns'].rolling(window)
            df[f'var_95_{window}'] = returns_window.quantile(0.05)
            df[f'var_99_{window}'] = returns_window.quantile(0.01)
            df[f'cvar_95_{window}'] = returns_window.apply(lambda x: x[x <= x.quantile(0.05)].mean())
            df[f'cvar_99_{window}'] = returns_window.apply(lambda x: x[x <= x.quantile(0.01)].mean())
        
        # Maximum drawdown
        for window in [20, 50, 100]:
            rolling_max = df['Close'].rolling(window).max()
            df[f'drawdown_{window}'] = (df['Close'] - rolling_max) / rolling_max
            df[f'max_drawdown_{window}'] = df[f'drawdown_{window}'].rolling(window).min()
        
        # Volatility clustering
        for window in [5, 10, 20]:
            vol = df['returns'].rolling(window).std()
            df[f'vol_clustering_{window}'] = vol.rolling(window).std() / vol.rolling(window).mean()
    
    def _add_market_microstructure_features_to_df(self, df: pd.DataFrame):
        """Add market microstructure features to existing DataFrame."""
        
        # Bid-ask spread proxy
        df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
        df['spread_proxy_ma'] = df['spread_proxy'].rolling(20).mean()
        df['spread_proxy_std'] = df['spread_proxy'].rolling(20).std()
        
        # Price impact
        df['price_impact'] = df['returns'].abs() / df['Volume']
        df['price_impact_ma'] = df['price_impact'].rolling(20).mean()
        
        # Volume-weighted average price (VWAP)
        df['vwap'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        df['price_vwap_ratio'] = df['Close'] / df['vwap']
        
        # Volume profile
        df['volume_profile'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_std'] = df['Volume'].rolling(20).std()
        df['volume_skew'] = df['Volume'].rolling(20).skew()
        
        # Order flow imbalance (simplified)
        df['flow_imbalance'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
        
        # Market efficiency ratio
        for window in [10, 20, 50]:
            price_range = (df['High'].rolling(window).max() - df['Low'].rolling(window).min())
            path_length = df['returns'].abs().rolling(window).sum()
            df[f'efficiency_ratio_{window}'] = price_range / (df['Close'] * path_length)
        
        # Liquidity measures
        df['amihud_illiquidity'] = df['returns'].abs() / df['Volume']
        df['amihud_ma'] = df['amihud_illiquidity'].rolling(20).mean()
    
    def _add_machine_learning_features_to_df(self, df: pd.DataFrame):
        """Add machine learning-based features to existing DataFrame."""
        
        # Principal Component Analysis on technical indicators
        tech_features = ['rsi_14', 'macd', 'stoch_k', 'williams_r', 'cci', 'mfi']
        available_features = [f for f in tech_features if f in df.columns]
        
        if len(available_features) >= 3:
            # Fill missing values for PCA
            tech_data = df[available_features].fillna(method='ffill').fillna(0)
            
            # Standardize
            scaler = StandardScaler()
            tech_scaled = scaler.fit_transform(tech_data)
            
            # Apply PCA
            pca = PCA(n_components=min(3, len(available_features)))
            pca_features = pca.fit_transform(tech_scaled)
            
            for i in range(pca_features.shape[1]):
                df[f'pca_tech_{i+1}'] = pca_features[:, i]
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'momentum_ma_{period}'] = df[f'momentum_{period}'].rolling(period).mean()
        
        # Mean reversion features
        for period in [20, 50]:
            df[f'mean_reversion_{period}'] = (df['Close'] - df['Close'].rolling(period).mean()) / df['Close'].rolling(period).std()
        
        # Trend strength
        for period in [20, 50]:
            # Linear regression slope
            x = np.arange(period)
            slopes = []
            for i in range(period, len(df)):
                y = df['Close'].iloc[i-period:i].values
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
            
            df[f'trend_slope_{period}'] = [np.nan] * period + slopes
        
        # Volatility regime features
        for window in [20, 50]:
            vol = df['returns'].rolling(window).std()
            vol_ma = vol.rolling(window).mean()
            vol_std = vol.rolling(window).std()
            df[f'vol_regime_{window}'] = (vol - vol_ma) / vol_std
    
    def _add_options_features_to_df(self, df: pd.DataFrame):
        """Add options-based features to existing DataFrame when available."""
        if self.options_data is None:
            return
        
        options_df = self.options_data.copy()
        
        # Aggregate options data by date
        if 'implied_volatility' in options_df.columns:
            # IV features
            iv_mean = options_df.groupby(options_df.index)['implied_volatility'].mean()
            iv_std = options_df.groupby(options_df.index)['implied_volatility'].std()
            iv_skew = options_df.groupby(options_df.index)['implied_volatility'].skew()
            
            df['iv_mean'] = iv_mean
            df['iv_std'] = iv_std
            df['iv_skew'] = iv_skew
            
            # IV term structure (if multiple expirations available)
            if 'expiration' in options_df.columns:
                # Calculate days to expiration
                options_df['dte'] = (pd.to_datetime(options_df['expiration']) - options_df.index).dt.days
                
                # Short-term vs long-term IV
                short_term = options_df[options_df['dte'] <= 30]
                long_term = options_df[options_df['dte'] > 30]
                
                if not short_term.empty and not long_term.empty:
                    short_iv = short_term.groupby(short_term.index)['implied_volatility'].mean()
                    long_iv = long_term.groupby(long_term.index)['implied_volatility'].mean()
                    
                    df['iv_short_term'] = short_iv
                    df['iv_long_term'] = long_iv
                    df['iv_term_structure'] = short_iv - long_iv
        
        # Volume and open interest features
        if 'volume' in options_df.columns:
            total_volume = options_df.groupby(options_df.index)['volume'].sum()
            df['options_volume'] = total_volume
            df['options_volume_ma'] = total_volume.rolling(20).mean()
            df['options_volume_ratio'] = total_volume / total_volume.rolling(20).mean()
        
        if 'open_interest' in options_df.columns:
            total_oi = options_df.groupby(options_df.index)['open_interest'].sum()
            df['options_oi'] = total_oi
            df['options_oi_ma'] = total_oi.rolling(20).mean()
            df['options_oi_ratio'] = total_oi / total_oi.rolling(20).mean()
        
        # Put-call ratio
        if 'type' in options_df.columns:
            calls = options_df[options_df['type'] == 'call']
            puts = options_df[options_df['type'] == 'put']
            
            if not calls.empty and not puts.empty:
                call_volume = calls.groupby(calls.index)['volume'].sum()
                put_volume = puts.groupby(puts.index)['volume'].sum()
                
                df['put_call_ratio'] = put_volume / call_volume
                df['put_call_ratio_ma'] = df['put_call_ratio'].rolling(20).mean()
    
    def _add_multi_timeframe_features_to_df(self, df: pd.DataFrame):
        """Add multi-timeframe analysis features to existing DataFrame."""
        
        # Resample to different timeframes
        timeframes = {
            '1w': 'W',
            '1m': 'M',
            '3m': 'Q'
        }
        
        for tf_name, tf_code in timeframes.items():
            # Resample OHLCV
            resampled = df.resample(tf_code).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            # Calculate features on resampled data
            resampled[f'returns_{tf_name}'] = resampled['Close'].pct_change()
            resampled[f'volatility_{tf_name}'] = resampled[f'returns_{tf_name}'].rolling(5).std()
            resampled[f'sma_20_{tf_name}'] = resampled['Close'].rolling(20).mean()
            
            # Reindex to original frequency and forward fill
            for col in [f'returns_{tf_name}', f'volatility_{tf_name}', f'sma_20_{tf_name}']:
                df[col] = resampled[col].reindex(df.index, method='ffill')
    
    def get_feature_summary(self) -> Dict:
        """Get summary of engineered features."""
        return self.features

def main():
    """Example usage of advanced feature engineering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced feature engineering for financial data")
    parser.add_argument("price_file", help="Price data CSV file")
    parser.add_argument("--options", help="Options data CSV file")
    parser.add_argument("--output", help="Output file for engineered features")
    parser.add_argument("--summary", help="Output file for feature summary")
    
    args = parser.parse_args()
    
    try:
        # Load data
        price_data = pd.read_csv(args.price_file, index_col=0, parse_dates=True)
        options_data = pd.read_csv(args.options, index_col=0, parse_dates=True) if args.options else None
        
        # Create feature engineer
        engineer = AdvancedFeatureEngineer(price_data, options_data)
        
        # Engineer features
        df_with_features = engineer.engineer_features()
        
        # Get feature summary
        summary = engineer.get_feature_summary()
        
        # Save results
        if args.output:
            df_with_features.to_csv(args.output)
            logger.info(f"Features saved to {args.output}")
        
        if args.summary:
            pd.DataFrame([summary]).to_csv(args.summary)
            logger.info(f"Feature summary saved to {args.summary}")
        
        # Print summary
        print("\nFeature Engineering Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

if __name__ == "__main__":
    main() 