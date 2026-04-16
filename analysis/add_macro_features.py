import os
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from pandas_datareader import data as pdr

# List of FRED macroeconomic indicators (symbol: FRED code)
FRED_CODES = {
    'FedFundsRate': 'FEDFUNDS',
    '10Y_Treasury': 'DGS10',
    '2Y_Treasury': 'DGS2',
    'CPI': 'CPIAUCSL',
    'CoreCPI': 'CPILFESL',
    'PPI': 'PPIACO',
    'UnemploymentRate': 'UNRATE',
    'NonfarmPayrolls': 'PAYEMS',
    'InitialClaims': 'ICSA',
    'GDP': 'GDP',
    'IndustrialProduction': 'INDPRO',
    'RetailSales': 'RSAFS',
    'ConsumerConfidence': 'UMCSENT',
    'ISM_Manufacturing': 'NAPM',
}

START_DATE = '2010-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

print("Downloading macroeconomic indicators from FRED...")
macro_df = pd.DataFrame()
for name, code in FRED_CODES.items():
    try:
        print(f"  Downloading {name} ({code})...")
        series = pdr.DataReader(code, 'fred', START_DATE, END_DATE)
        macro_df[name] = series[code]
    except Exception as e:
        print(f"  ❌ Failed to download {name}: {e}")

# Download VIX from Yahoo Finance
print("\nDownloading VIX from Yahoo Finance...")
vix = yf.Ticker('^VIX').history(start=START_DATE, end=END_DATE)
if not vix.empty:
    macro_df['VIX'] = vix['Close']
    print("  ✅ VIX downloaded and added.")
else:
    print("  ❌ Failed to download VIX.")

# Download S&P 500 index from Yahoo Finance
print("\nDownloading S&P 500 index (^GSPC) from Yahoo Finance...")
gspc = yf.Ticker('^GSPC').history(start=START_DATE, end=END_DATE)
if not gspc.empty:
    macro_df['SP500'] = gspc['Close']
    print("  ✅ S&P 500 index downloaded and added.")
else:
    print("  ❌ Failed to download S&P 500 index.")

# Forward-fill and back-fill macro data to cover all dates
macro_df = macro_df.ffill().bfill()

# Save macro features for reference
macro_df.to_csv('macro_features_full.csv')
print("\nMacro features saved to macro_features_full.csv")

# Merge macro features into each stock file
stock_files = [f for f in os.listdir('.') if f.endswith('_data_full_clean.csv')]
print(f"\nMerging macro features into {len(stock_files)} stock files...")

for i, file in enumerate(stock_files):
    symbol = file.replace('_data_full_clean.csv', '').upper()
    try:
        df = pd.read_csv(file)
        # Parse date and set as index
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None, ambiguous='NaT', nonexistent='NaT')
        df.set_index('Date', inplace=True)
        # Prepare macro_df index
        macro_df_reset = macro_df.reset_index()
        macro_df_reset['index'] = pd.to_datetime(macro_df_reset['index']).dt.tz_localize(None, ambiguous='NaT', nonexistent='NaT')
        macro_df_reset = macro_df_reset.rename(columns={'index': 'Date'})
        macro_df_reset = macro_df_reset.sort_values('Date')
        df = df.sort_index()
        # Merge using merge_asof
        merged = pd.merge_asof(
            df.reset_index().sort_values('Date'),
            macro_df_reset,
            on='Date',
            direction='backward'
        )
        merged.set_index('Date', inplace=True)
        # Forward-fill any remaining NaN values at the beginning
        macro_cols = [col for col in macro_df.columns if col in merged.columns]
        for col in macro_cols:
            merged[col] = merged[col].ffill().bfill()
        # Save merged file
        merged_file = file.replace('_data_full_clean.csv', '_data_full_macro_fixed.csv')
        merged.to_csv(merged_file)
        macro_data_count = merged[macro_cols].notna().sum().sum()
        total_possible = len(merged) * len(macro_cols)
        success_rate = macro_data_count / total_possible if total_possible > 0 else 0
        print(f"[{i+1}/{len(stock_files)}] {symbol}: Macro features merged ({success_rate:.1%} success rate) -> {merged_file}")
    except Exception as e:
        print(f"[{i+1}/{len(stock_files)}] {symbol}: ❌ Error merging macro features: {e}")
    if (i+1) % 10 == 0:
        print(f"Progress: {i+1}/{len(stock_files)} files processed.")
print("\nAll macro features merged using merge_asof!") 