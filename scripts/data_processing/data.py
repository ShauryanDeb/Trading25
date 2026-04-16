import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging
from typing import Optional, Tuple
from data_validation import DataValidator
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_stock_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> Tuple[pd.DataFrame, dict]:
    """
    Download and validate stock data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD) or 'none' for max history
        end_date: End date (YYYY-MM-DD) or None for today
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        Tuple of (validated_data, validation_report)
    """
    
    try:
        # Initialize ticker
        ticker = yf.Ticker(symbol)
        
        # Handle date ranges
        if start_date and start_date.lower() != 'none':
            start = pd.to_datetime(start_date)
        else:
            start = None
        
        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = pd.Timestamp.now(tz=pytz.UTC)
        
        # Download data with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True  # Adjust for splits and dividends
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                continue
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Convert index to timezone-aware datetime if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.UTC)
        
        # Ensure column names are standardized
        df.columns = [col.title() for col in df.columns]
        
        # Create validator
        validator = DataValidator(price_data=df)
        
        # Validate and clean data
        cleaned_data, validation_report = validator.validate_price_data()
        
        # Additional checks specific to downloaded data
        validation_report['download_info'] = {
            'symbol': symbol,
            'start_date': str(cleaned_data.index.min()),
            'end_date': str(cleaned_data.index.max()),
            'trading_days': len(cleaned_data),
            'interval': interval
        }
        
        # Check for sufficient data
        min_required_days = 252  # One trading year
        if len(cleaned_data) < min_required_days:
            logger.warning(f"Less than {min_required_days} days of data available")
            validation_report['download_info']['warning'] = f"Limited data: only {len(cleaned_data)} days available"
        
        # Check for recent data
        last_date = cleaned_data.index.max()
        current_time = pd.Timestamp.now(tz=pytz.UTC)
        if (current_time - last_date).days > 5:
            logger.warning(f"Most recent data is {(current_time - last_date).days} days old")
            validation_report['download_info']['warning'] = "Data may not be current"
        
        return cleaned_data, validation_report
    
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        raise

def save_data(
    df: pd.DataFrame,
    validation_report: dict,
    output_file: str,
    report_file: Optional[str] = None
):
    """Save validated data and report."""
    
    try:
        # Convert timezone-aware dates to naive before saving
        df.index = df.index.tz_localize(None)
        
        # Save data
        df.to_csv(output_file)
        logger.info(f"Saved validated data to {output_file}")
        
        # Save report if requested
        if report_file:
            pd.DataFrame([validation_report]).to_csv(report_file)
            logger.info(f"Saved validation report to {report_file}")
        
        # Print summary
        print("\nData Download and Validation Summary:")
        print(f"Symbol: {validation_report['download_info']['symbol']}")
        print(f"Date Range: {validation_report['download_info']['start_date']} to {validation_report['download_info']['end_date']}")
        print(f"Trading Days: {validation_report['download_info']['trading_days']}")
        
        quality_score = validation_report.get('price_data_quality_score', 'N/A')
        if isinstance(quality_score, (int, float)):
            print(f"Data Quality Score: {quality_score:.2%}")
        else:
            print(f"Data Quality Score: {quality_score}")
        
        if 'warning' in validation_report['download_info']:
            print(f"\nWarning: {validation_report['download_info']['warning']}")
            
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download and validate stock data")
    parser.add_argument("symbol", help="Stock symbol")
    parser.add_argument("--start", default="none", help="Start date (YYYY-MM-DD) or 'none' for max history")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", choices=['1d', '1wk', '1mo'], help="Data interval")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--report", help="Output report file")
    
    args = parser.parse_args()
    
    try:
        # Download and validate data
        df, report = download_stock_data(
            args.symbol,
            args.start,
            args.end,
            args.interval
        )
        
        # Determine output file if not provided
        output_file = args.output or f"{args.symbol.lower()}.csv"
        report_file = args.report or f"{args.symbol.lower()}_validation.csv"
        
        # Save results
        save_data(df, report, output_file, report_file)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
