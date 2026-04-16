import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging
from datetime import datetime, timedelta
import warnings
import pytz
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedDataValidator:
    """
    Enhanced data validation and cleaning for financial time series data.
    
    Advanced Features:
    - Multi-frequency data support (intraday, daily, weekly, monthly)
    - Advanced outlier detection using multiple methods
    - Sophisticated data alignment and synchronization
    - Market microstructure validation
    - Comprehensive data quality scoring
    - Visualization and reporting capabilities
    """
    
    def __init__(self,
                 price_data: Optional[pd.DataFrame] = None,
                 options_data: Optional[pd.DataFrame] = None,
                 frequency: str = '1d',
                 outlier_methods: List[str] = None,
                 min_price: float = 0.01,
                 max_price_change: float = 0.3,
                 volume_outlier_threshold: float = 5.0,
                 market_hours: Tuple[str, str] = ('09:30', '16:00'),
                 timezone: str = 'US/Eastern'):
        
        self.price_data = price_data
        self.options_data = options_data
        self.frequency = frequency
        self.outlier_methods = outlier_methods or ['zscore', 'iqr', 'isolation_forest']
        self.min_price = min_price
        self.max_price_change = max_price_change
        self.volume_outlier_threshold = volume_outlier_threshold
        self.market_hours = market_hours
        self.timezone = pytz.timezone(timezone)
        self.validation_report = {}
        self.quality_metrics = {}
        
    def _ensure_timezone_aware(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has timezone-aware index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(self.timezone)
        return df
    
    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detect the frequency of the time series data."""
        if len(df) < 2:
            return 'unknown'
        
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff <= pd.Timedelta(minutes=1):
            return '1m'
        elif median_diff <= pd.Timedelta(minutes=5):
            return '5m'
        elif median_diff <= pd.Timedelta(minutes=15):
            return '15m'
        elif median_diff <= pd.Timedelta(hours=1):
            return '1h'
        elif median_diff <= pd.Timedelta(days=1):
            return '1d'
        elif median_diff <= pd.Timedelta(weeks=1):
            return '1wk'
        elif median_diff <= pd.Timedelta(days=30):
            return '1mo'
        else:
            return 'unknown'
    
    def _validate_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and filter data to market hours for intraday data."""
        if self.frequency in ['1m', '5m', '15m', '1h']:
            # Convert to market timezone
            df_market = df.copy()
            df_market.index = df_market.index.tz_convert(self.timezone)
            
            # Filter to market hours
            market_start = pd.to_datetime(self.market_hours[0]).time()
            market_end = pd.to_datetime(self.market_hours[1]).time()
            
            mask = (df_market.index.time >= market_start) & (df_market.index.time <= market_end)
            df_filtered = df_market[mask]
            
            logger.info(f"Filtered {len(df) - len(df_filtered)} non-market hours records")
            return df_filtered
        
        return df
    
    def _advanced_outlier_detection(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Advanced outlier detection using multiple methods."""
        outliers = pd.Series(False, index=df.index)
        
        for method in self.outlier_methods:
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outliers |= (z_scores > 3)
                
            elif method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers |= ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
                
            elif method == 'isolation_forest':
                if len(df) > 10:  # Need sufficient data
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers_iso = iso_forest.fit_predict(df[[column]].dropna()) == -1
                    valid_indices = df[column].dropna().index
                    outliers.loc[valid_indices] |= outliers_iso
                    
            elif method == 'rolling_stats':
                window = min(20, len(df) // 4)
                if window > 5:
                    rolling_mean = df[column].rolling(window).mean()
                    rolling_std = df[column].rolling(window).std()
                    outliers |= ((df[column] - rolling_mean).abs() > (3 * rolling_std))
        
        return outliers
    
    def _validate_market_microstructure(self, df: pd.DataFrame) -> Dict:
        """Validate market microstructure features."""
        microstructure_report = {}
        
        # Check for zero-volume days (suspicious for liquid stocks)
        zero_volume_days = (df['Volume'] == 0).sum()
        microstructure_report['zero_volume_days'] = zero_volume_days
        
        # Check for price gaps
        price_gaps = abs(df['Close'].pct_change()) > 0.05  # 5% gaps
        microstructure_report['large_price_gaps'] = price_gaps.sum()
        
        # Check for bid-ask spread proxies (High-Low relative to price)
        spread_proxy = (df['High'] - df['Low']) / df['Close']
        large_spreads = spread_proxy > 0.1  # 10% spread
        microstructure_report['large_spreads'] = large_spreads.sum()
        
        # Check for volume-price relationship
        volume_price_corr = df['Volume'].corr(df['Close'])
        microstructure_report['volume_price_correlation'] = volume_price_corr
        
        # Check for autocorrelation in returns
        returns = df['Close'].pct_change().dropna()
        if len(returns) > 10:
            autocorr = returns.autocorr()
            microstructure_report['returns_autocorrelation'] = autocorr
        
        return microstructure_report
    
    def validate_price_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Enhanced price data validation and cleaning."""
        
        if self.price_data is None:
            raise ValueError("Price data not provided")
        
        df = self.price_data.copy()
        report = {}
        
        # Ensure timezone-aware index
        df = self._ensure_timezone_aware(df)
        
        # Detect actual frequency
        detected_freq = self._detect_frequency(df)
        report['detected_frequency'] = detected_freq
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 1. Basic data quality checks
        initial_rows = len(df)
        report['initial_rows'] = initial_rows
        
        # Check for duplicate timestamps
        duplicates = df.index.duplicated()
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate timestamps")
            df = df[~duplicates]
        report['duplicate_timestamps'] = duplicates.sum()
        
        # 2. Market hours validation for intraday data
        df = self._validate_market_hours(df)
        report['rows_after_market_hours_filter'] = len(df)
        
        # 3. Missing data detection and handling
        missing_data = df[required_columns].isnull().sum()
        report['missing_data'] = missing_data.to_dict()
        
        # Advanced missing data handling based on frequency
        if self.frequency in ['1m', '5m', '15m']:
            # For intraday data, forward fill small gaps
            df = df.fillna(method='ffill', limit=3)
        else:
            # For daily data, forward fill larger gaps
            df = df.fillna(method='ffill', limit=5)
        
        # 4. Price validation
        invalid_prices = (df[['Open', 'High', 'Low', 'Close']] <= self.min_price).any(axis=1)
        report['invalid_prices'] = invalid_prices.sum()
        
        # Enhanced OHLC relationship validation
        invalid_ohlc = (
            (df['High'] < df['Low']) |
            (df['Open'] > df['High']) |
            (df['Open'] < df['Low']) |
            (df['Close'] > df['High']) |
            (df['Close'] < df['Low'])
        )
        report['invalid_ohlc'] = invalid_ohlc.sum()
        
        # 5. Advanced outlier detection
        outlier_report = {}
        for col in ['Open', 'High', 'Low', 'Close']:
            outliers = self._advanced_outlier_detection(df, col)
            outlier_report[f'{col}_outliers'] = outliers.sum()
            
            if outliers.any():
                logger.warning(f"Found {outliers.sum()} outliers in {col}")
                # Replace outliers with rolling median
                df.loc[outliers, col] = df[col].rolling(5, center=True).median()
        
        report['outlier_detection'] = outlier_report
        
        # 6. Market microstructure validation
        microstructure_report = self._validate_market_microstructure(df)
        report['market_microstructure'] = microstructure_report
        
        # 7. Fix OHLC relationships
        df.loc[invalid_ohlc, 'High'] = df.loc[invalid_ohlc, ['Open', 'High', 'Close']].max(axis=1)
        df.loc[invalid_ohlc, 'Low'] = df.loc[invalid_ohlc, ['Open', 'Low', 'Close']].min(axis=1)
        
        # 8. Calculate comprehensive quality metrics
        report['rows_after_cleaning'] = len(df)
        report['data_completeness'] = 1 - (df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns)))
        report['price_consistency'] = 1 - (invalid_ohlc.sum() / len(df))
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(report)
        report['overall_quality_score'] = quality_score
        
        self.validation_report['price_data'] = report
        return df, report
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculate comprehensive quality score."""
        weights = {
            'data_completeness': 0.3,
            'price_consistency': 0.25,
            'outlier_cleanliness': 0.2,
            'microstructure_quality': 0.15,
            'frequency_consistency': 0.1
        }
        
        scores = {}
        
        # Data completeness
        scores['data_completeness'] = report.get('data_completeness', 0)
        
        # Price consistency
        scores['price_consistency'] = report.get('price_consistency', 0)
        
        # Outlier cleanliness (inverse of outlier ratio)
        total_outliers = sum(report.get('outlier_detection', {}).values())
        total_observations = report.get('rows_after_cleaning', 1)
        scores['outlier_cleanliness'] = max(0, 1 - (total_outliers / (total_observations * 4)))  # 4 OHLC columns
        
        # Microstructure quality
        microstructure = report.get('market_microstructure', {})
        zero_volume_penalty = min(1, microstructure.get('zero_volume_days', 0) / 10)
        large_gaps_penalty = min(1, microstructure.get('large_price_gaps', 0) / 50)
        scores['microstructure_quality'] = 1 - (zero_volume_penalty + large_gaps_penalty) / 2
        
        # Frequency consistency
        detected_freq = report.get('detected_frequency', 'unknown')
        expected_freq = self.frequency
        scores['frequency_consistency'] = 1.0 if detected_freq == expected_freq else 0.5
        
        # Calculate weighted average
        overall_score = sum(scores[key] * weights[key] for key in weights)
        return overall_score
    
    def validate_options_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Validate and clean options data."""
        
        if self.options_data is None:
            return None, {}
        
        df = self.options_data.copy()
        report = {}
        
        # Ensure timezone-aware index
        df = self._ensure_timezone_aware(df)
        
        # 1. Basic validation
        initial_rows = len(df)
        report['initial_rows'] = initial_rows
        
        # Check required columns
        required_columns = ['strike', 'expiration', 'type', 'volume', 'open_interest', 'implied_volatility']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing options columns: {missing_columns}")
        
        # 2. Data cleaning
        # Remove duplicates
        duplicates = df.index.duplicated()
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate option entries")
            df = df[~duplicates]
        report['duplicate_entries'] = duplicates.sum()
        
        # 3. Options-specific validation
        if 'type' in df.columns:
            invalid_types = ~df['type'].isin(['call', 'put'])
            report['invalid_option_types'] = invalid_types.sum()
            df = df[~invalid_types]
        
        if 'implied_volatility' in df.columns:
            # Remove extreme IV values
            extreme_iv = (df['implied_volatility'] <= 0) | (df['implied_volatility'] > 5)  # > 500% IV
            report['extreme_iv'] = extreme_iv.sum()
            df.loc[extreme_iv, 'implied_volatility'] = np.nan
        
        if 'volume' in df.columns and 'open_interest' in df.columns:
            # Flag suspicious volume/OI relationships
            suspicious_vol = df['volume'] > df['open_interest'] * 2
            report['suspicious_volume'] = suspicious_vol.sum()
        
        # 4. Calculate quality metrics
        report['rows_after_cleaning'] = len(df)
        if set(required_columns).issubset(df.columns):
            report['data_completeness'] = 1 - (df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns)))
        
        self.validation_report['options_data'] = report
        return df, report
    
    def align_data(self, price_data: pd.DataFrame, options_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Enhanced data alignment with frequency conversion support."""
        
        # Ensure both datasets have timezone-aware indices
        price_data = self._ensure_timezone_aware(price_data)
        if options_data is not None:
            options_data = self._ensure_timezone_aware(options_data)
            
            # Find common date range
            start_date = max(price_data.index.min(), options_data.index.min())
            end_date = min(price_data.index.max(), options_data.index.max())
            
            # Trim both datasets to common range
            price_data = price_data[start_date:end_date]
            options_data = options_data[start_date:end_date]
            
            # Frequency alignment
            if self.frequency != '1d':
                # Resample options data to match price frequency
                options_data = options_data.resample(self.frequency).ffill()
            
            # Ensure all price dates have options data
            missing_options_dates = price_data.index.difference(options_data.index)
            if len(missing_options_dates) > 0:
                logger.warning(f"Missing options data for {len(missing_options_dates)} dates")
        
        return price_data, options_data
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report with visualizations."""
        
        if not self.validation_report:
            logger.warning("No validation has been performed yet")
            return {}
        
        report = {
            'timestamp': datetime.now(tz=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'),
            'price_data': self.validation_report.get('price_data', {}),
            'options_data': self.validation_report.get('options_data', {}),
        }
        
        # Add overall quality score
        if 'price_data' in self.validation_report:
            report['overall_quality_score'] = self.validation_report['price_data'].get('overall_quality_score', 0)
        
        return report
    
    def create_validation_plots(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create validation plots for data quality assessment."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Validation Plots', fontsize=16)
        
        # Price time series
        axes[0, 0].plot(df.index, df['Close'])
        axes[0, 0].set_title('Price Time Series')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Volume time series
        axes[0, 1].plot(df.index, df['Volume'])
        axes[0, 1].set_title('Volume Time Series')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Returns distribution
        returns = df['Close'].pct_change().dropna()
        axes[1, 0].hist(returns, bins=50, alpha=0.7)
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Frequency')
        
        # OHLC relationship check
        ohlc_errors = (
            (df['High'] < df['Low']) |
            (df['Open'] > df['High']) |
            (df['Open'] < df['Low']) |
            (df['Close'] > df['High']) |
            (df['Close'] < df['Low'])
        )
        axes[1, 1].plot(df.index, ohlc_errors.astype(int))
        axes[1, 1].set_title('OHLC Relationship Errors')
        axes[1, 1].set_ylabel('Error (1=Error, 0=OK)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation plots saved to {save_path}")
        
        plt.show()

def main():
    """Example usage of enhanced data validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced validation and cleaning of financial data")
    parser.add_argument("price_file", help="Price data CSV file")
    parser.add_argument("--options", help="Options data CSV file")
    parser.add_argument("--frequency", default="1d", help="Expected data frequency")
    parser.add_argument("--output", help="Output file for cleaned data")
    parser.add_argument("--report", help="Output file for validation report")
    parser.add_argument("--plots", help="Output file for validation plots")
    
    args = parser.parse_args()
    
    try:
        # Load data
        price_data = pd.read_csv(args.price_file, index_col=0, parse_dates=True)
        options_data = pd.read_csv(args.options, index_col=0, parse_dates=True) if args.options else None
        
        # Create enhanced validator
        validator = EnhancedDataValidator(
            price_data=price_data,
            options_data=options_data,
            frequency=args.frequency
        )
        
        # Validate data
        cleaned_price, price_report = validator.validate_price_data()
        cleaned_options, options_report = validator.validate_options_data() if options_data is not None else (None, {})
        
        # Align data
        aligned_price, aligned_options = validator.align_data(cleaned_price, cleaned_options)
        
        # Generate report
        report = validator.generate_validation_report()
        
        # Create plots
        if args.plots:
            validator.create_validation_plots(aligned_price, args.plots)
        
        # Save results
        if args.output:
            aligned_price.index = aligned_price.index.tz_localize(None)
            aligned_price.to_csv(f"enhanced_cleaned_{args.price_file}")
            if aligned_options is not None:
                aligned_options.index = aligned_options.index.tz_localize(None)
                aligned_options.to_csv(f"enhanced_cleaned_{args.options}")
        
        if args.report:
            pd.DataFrame([report]).to_csv(args.report)
        
        # Print summary
        print("\nEnhanced Data Validation Summary:")
        print(f"Overall Quality Score: {report.get('overall_quality_score', 'N/A'):.2%}")
        print(f"Detected Frequency: {price_report.get('detected_frequency', 'N/A')}")
        print(f"Data Completeness: {price_report.get('data_completeness', 'N/A'):.2%}")
        print(f"Price Consistency: {price_report.get('price_consistency', 'N/A'):.2%}")
        
    except Exception as e:
        logger.error(f"Error during enhanced data validation: {e}")
        raise

if __name__ == "__main__":
    main() 