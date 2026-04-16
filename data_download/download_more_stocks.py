import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Additional stocks to download
ADDITIONAL_STOCKS = [
    # Technology
    'NVDA', 'META', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
    # Financial
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
    # Consumer
    'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'TGT',
    # Industrial
    'BA', 'CAT', 'MMM', 'GE', 'HON', 'UPS', 'FDX', 'RTX', 'LMT', 'NOC',
    # Energy
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'PSX', 'VLO', 'MPC',
    # Materials
    'LIN', 'APD', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'AA', 'BLL', 'ALB',
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'PSA', 'SPG', 'O', 'WELL', 'VICI',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'WEC', 'DTE', 'ED',
    # Communication
    'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'DISH', 'PARA', 'FOX', 'NWSA', 'NWS'
]

def download_stocks(symbols, start_date='2022-01-01', end_date='2024-12-31'):
    """Download stock data for multiple symbols."""
    print(f"Downloading data for {len(symbols)} additional stocks...")
    
    successful_downloads = []
    failed_downloads = []
    
    for i, symbol in enumerate(symbols):
        try:
            print(f"Downloading {symbol} ({i+1}/{len(symbols)})...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if len(data) > 100:  # Ensure sufficient data
                filename = f"{symbol.lower()}_data.csv"
                data.to_csv(filename)
                successful_downloads.append(symbol)
                print(f"  ✅ {symbol}: {len(data)} days of data")
            else:
                print(f"  ❌ {symbol}: Insufficient data ({len(data)} days)")
                failed_downloads.append(symbol)
                
        except Exception as e:
            print(f"  ❌ Error downloading {symbol}: {e}")
            failed_downloads.append(symbol)
    
    print(f"\nDownload Summary:")
    print(f"  ✅ Successful: {len(successful_downloads)} stocks")
    print(f"  ❌ Failed: {len(failed_downloads)} stocks")
    
    if failed_downloads:
        print(f"  Failed symbols: {failed_downloads}")
    
    return successful_downloads

if __name__ == "__main__":
    # Download additional stocks
    successful = download_stocks(ADDITIONAL_STOCKS)
    
    print(f"\n🎉 Successfully downloaded {len(successful)} additional stocks!")
    print(f"📊 Total stocks available: {len(successful) + 5} (including original 5)")
    print(f"🚀 Ready to retrain multi-stock model with expanded dataset!")
    
    # List all available stock files
    stock_files = [f for f in os.listdir('.') if f.endswith('_data.csv')]
    print(f"\n📁 Available stock files ({len(stock_files)}):")
    for file in sorted(stock_files):
        print(f"  - {file}") 