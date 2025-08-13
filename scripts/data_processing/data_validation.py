import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from datetime import datetime, timedelta
import warnings
import pytz
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation and cleaning for financial time series data.
    
    Features:
    - Missing data detection and handling
    - Outlier detection and treatment
    - Data alignment between price and options
    - Timestamp validation and standardization
    - Data quality metrics and reporting
    """
    
    def __init__(self,
                 price_data: Optional[pd.DataFrame] = None,
                 options_data: Optional[pd.DataFrame] = None,
                 outlier_std_threshold: float = 3.0,
                 min_price: float = 0.01,
                 max_price_change: float = 0.3,  # 30% max price change
                 volume_outlier_threshold: float = 5.0):
        
        self.price_data = price_data
        self.options_data = options_data
        self.outlier_std_threshold = outlier_std_threshold
        self.min_price = min_price
        self.max_price_change = max_price_change
        self.volume_outlier_threshold = volume_outlier_threshold
        self.validation_report = {}
        
    def _ensure_timezone_aware(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has timezone-aware index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.UTC)
        return df
    
    def validate_price_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and clean price data.
        
        Returns:
            Tuple of (cleaned_data, validation_report)
        """
        
        if self.price_data is None:
            raise ValueError("Price data not provided")
        
        df = self.price_data.copy()
        report = {}
        
        # Ensure timezone-aware index
        df = self._ensure_timezone_aware(df)
        
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
        
        # 2. Missing data detection
        missing_data = df[required_columns].isnull().sum()
        report['missing_data'] = missing_data.to_dict()
        
        # Forward fill small gaps (up to 5 minutes for intraday, 1 day for daily)
        df = df.fillna(method='ffill', limit=5)
        
        # 3. Price validation
        # Check for negative prices
        invalid_prices = (df[['Open', 'High', 'Low', 'Close']] <= self.min_price).any(axis=1)
        report['invalid_prices'] = invalid_prices.sum()
        
        # Check OHLC relationship
        invalid_ohlc = (
            (df['High'] < df['Low']) |
            (df['Open'] > df['High']) |
            (df['Open'] < df['Low']) |
            (df['Close'] > df['High']) |
            (df['Close'] < df['Low'])
        )
        report['invalid_ohlc'] = invalid_ohlc.sum()
        
        # 4. Outlier detection
        # Price change outliers
        returns = df['Close'].pct_change()
        price_outliers = abs(returns) > self.max_price_change
        report['price_outliers'] = price_outliers.sum()
        
        # Volume outliers
        volume_ma = df['Volume'].rolling(20).mean()
        volume_std = df['Volume'].rolling(20).std()
        volume_outliers = (df['Volume'] - volume_ma).abs() > (self.volume_outlier_threshold * volume_std)
        report['volume_outliers'] = volume_outliers.sum()
        
        # 5. Clean data
        # Remove or fix price outliers
        for col in ['Open', 'High', 'Low', 'Close']:
            # Detect outliers using rolling statistics
            rolling_mean = df[col].rolling(20).mean()
            rolling_std = df[col].rolling(20).std()
            outliers = (df[col] - rolling_mean).abs() > (self.outlier_std_threshold * rolling_std)
            
            if outliers.any():
                logger.warning(f"Found {outliers.sum()} outliers in {col}")
                # Replace outliers with rolling median
                df.loc[outliers, col] = df[col].rolling(5, center=True).median()
        
        # Fix OHLC relationships
        df.loc[invalid_ohlc, 'High'] = df.loc[invalid_ohlc, ['Open', 'High', 'Close']].max(axis=1)
        df.loc[invalid_ohlc, 'Low'] = df.loc[invalid_ohlc, ['Open', 'Low', 'Close']].min(axis=1)
        
        # 6. Calculate quality metrics
        report['rows_after_cleaning'] = len(df)
        report['data_completeness'] = 1 - (df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns)))
        report['price_consistency'] = 1 - (invalid_ohlc.sum() / len(df))
        
        self.validation_report['price_data'] = report
        return df, report
    
    def validate_options_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and clean options data.
        
        Returns:
            Tuple of (cleaned_data, validation_report)
        """
        
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
        """
        Align price and options data to ensure consistent timestamps.
        
        Returns:
            Tuple of (aligned_price_data, aligned_options_data)
        """
        
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
            
            # Ensure all price dates have options data
            missing_options_dates = price_data.index.difference(options_data.index)
            if len(missing_options_dates) > 0:
                logger.warning(f"Missing options data for {len(missing_options_dates)} dates")
        
        return price_data, options_data
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report."""
        
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
            price_metrics = self.validation_report['price_data']
            price_score = (
                price_metrics.get('data_completeness', 0) * 0.4 +
                price_metrics.get('price_consistency', 0) * 0.6
            )
            report['price_data_quality_score'] = price_score
        
        if 'options_data' in self.validation_report:
            options_metrics = self.validation_report['options_data']
            if 'data_completeness' in options_metrics:
                report['options_data_quality_score'] = options_metrics['data_completeness']
        
        return report

def main():
    """Example usage of data validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and clean financial data")
    parser.add_argument("price_file", help="Price data CSV file")
    parser.add_argument("--options", help="Options data CSV file")
    parser.add_argument("--output", help="Output file for cleaned data")
    parser.add_argument("--report", help="Output file for validation report")
    
    args = parser.parse_args()
    
    try:
        # Load data
        price_data = pd.read_csv(args.price_file, index_col=0, parse_dates=True)
        options_data = pd.read_csv(args.options, index_col=0, parse_dates=True) if args.options else None
        
        # Create validator
        validator = DataValidator(price_data, options_data)
        
        # Validate data
        cleaned_price, price_report = validator.validate_price_data()
        cleaned_options, options_report = validator.validate_options_data() if options_data is not None else (None, {})
        
        # Align data
        aligned_price, aligned_options = validator.align_data(cleaned_price, cleaned_options)
        
        # Generate report
        report = validator.generate_validation_report()
        
        # Save results
        if args.output:
            # Convert to timezone-naive before saving
            aligned_price.index = aligned_price.index.tz_localize(None)
            aligned_price.to_csv(f"cleaned_{args.price_file}")
            if aligned_options is not None:
                aligned_options.index = aligned_options.index.tz_localize(None)
                aligned_options.to_csv(f"cleaned_{args.options}")
        
        if args.report:
            pd.DataFrame([report]).to_csv(args.report)
        
        # Print summary
        print("\nData Validation Summary:")
        print(f"Price data quality score: {report.get('price_data_quality_score', 'N/A'):.2%}")
        if options_data is not None:
            print(f"Options data quality score: {report.get('options_data_quality_score', 'N/A'):.2%}")
        
        print("\nCleaning Summary:")
        print(f"Original price rows: {report['price_data']['initial_rows']}")
        print(f"Cleaned price rows: {report['price_data']['rows_after_cleaning']}")
        if options_data is not None:
            print(f"Original options rows: {report['options_data']['initial_rows']}")
            print(f"Cleaned options rows: {report['options_data']['rows_after_cleaning']}")
        
    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        raise

if __name__ == "__main__":
    main() 