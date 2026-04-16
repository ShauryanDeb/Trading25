import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Even more stocks for maximum diversification
ADDITIONAL_STOCKS_PHASE2 = [
    # More Technology
    'TXN', 'MU', 'AMAT', 'KLAC', 'LRCX', 'ADI', 'MCHP', 'MRVL', 'ON', 'SWKS',
    # More Healthcare
    'GILD', 'CVS', 'CI', 'ANTM', 'HUM', 'CNC', 'WBA', 'ZTS', 'REGN', 'VRTX',
    # More Financial
    'AXP', 'BLK', 'SCHW', 'CME', 'ICE', 'SPGI', 'MCO', 'CB', 'AIG', 'MET',
    # More Consumer
    'COST', 'LOW', 'TJX', 'ROST', 'ULTA', 'YUM', 'CMG', 'DPZ', 'MCD', 'SBUX',
    # More Industrial
    'DE', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'AME', 'FTV', 'XYL', 'DOV',
    # More Energy
    'OXY', 'PXD', 'DVN', 'HES', 'FANG', 'MRO', 'APA', 'NBL', 'COG', 'EQT',
    # More Materials
    'ECL', 'IFF', 'CTVA', 'MOS', 'NTR', 'IP', 'PKG', 'WRK', 'SEE', 'BMS',
    # More Real Estate
    'ARE', 'EQR', 'AVB', 'MAA', 'UDR', 'ESS', 'CPT', 'BXP', 'KIM', 'REG',
    # More Utilities
    'EIX', 'PEG', 'AEE', 'CMS', 'CNP', 'NI', 'LNT', 'ATO', 'BKH', 'IDA',
    # More Communication
    'LUMN', 'CTL', 'PARA', 'FOX', 'NWSA', 'NWS', 'TMUS', 'CMCSA', 'CHTR',
    # Small/Mid Cap Growth
    'SQ', 'ROKU', 'ZM', 'PTON', 'SNAP', 'UBER', 'LYFT', 'DASH', 'ABNB', 'SNOW',
    # Biotech/Pharma
    'BIIB', 'ALXN', 'ILMN', 'EXAS', 'DXCM', 'TWST', 'CRSP', 'EDIT', 'BEAM', 'NTLA',
    # Semiconductor
    'ASML', 'TSM', 'SMCI', 'AMD', 'NVDA', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI',
    # Cloud/SaaS
    'CRM', 'ADBE', 'ORCL', 'WDAY', 'NOW', 'TEAM', 'OKTA', 'ZS', 'CRWD', 'NET',
    # E-commerce/Retail
    'AMZN', 'EBAY', 'ETSY', 'SHOP', 'BABA', 'JD', 'PDD', 'SE', 'MELI', 'BIDU'
]

def download_phase2_stocks(symbols, start_date='2022-01-01', end_date='2024-12-31'):
    """Download data for phase 2 additional stocks."""
    print(f"Downloading Phase 2 data for {len(symbols)} additional stocks...")
    
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
    
    print(f"\nPhase 2 Download Summary:")
    print(f"  ✅ Successful: {len(successful_downloads)} stocks")
    print(f"  ❌ Failed: {len(failed_downloads)} stocks")
    
    if failed_downloads:
        print(f"  Failed symbols: {failed_downloads}")
    
    return successful_downloads

if __name__ == "__main__":
    # Download phase 2 stocks
    successful = download_phase2_stocks(ADDITIONAL_STOCKS_PHASE2)
    
    # Count total available stocks
    stock_files = [f for f in os.listdir('.') if f.endswith('_data.csv')]
    
    print(f"\n🎉 Successfully downloaded {len(successful)} additional stocks!")
    print(f"📊 Total stocks available: {len(stock_files)}")
    print(f"🚀 Ready to train ultra-expanded multi-stock model!")
    
    # List all available stock files
    print(f"\n📁 Available stock files ({len(stock_files)}):")
    for file in sorted(stock_files):
        print(f"  - {file}") 