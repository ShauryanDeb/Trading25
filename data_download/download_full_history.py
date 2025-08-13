import yfinance as yf
import os
from datetime import datetime

# Find all stock symbols from existing *_data.csv files
stock_files = [f for f in os.listdir('.') if f.endswith('_data.csv')]
all_symbols = [f.replace('_data.csv', '').upper() for f in stock_files]

START_DATE = '2010-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

print(f"Found {len(all_symbols)} stock symbols to download.")

success = 0
fail = 0
failed_symbols = []

for i, symbol in enumerate(all_symbols):
    try:
        print(f"[{i+1}/{len(all_symbols)}] Downloading {symbol} from {START_DATE} to {END_DATE}...")
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=START_DATE, end=END_DATE)
        if len(data) > 200:
            filename = f"{symbol.lower()}_data_full.csv"
            data.to_csv(filename)
            print(f"  ✅ {symbol}: {len(data)} days of data saved to {filename}")
            success += 1
        else:
            print(f"  ❌ {symbol}: Insufficient data ({len(data)} days)")
            fail += 1
            failed_symbols.append(symbol)
    except Exception as e:
        print(f"  ❌ Error downloading {symbol}: {e}")
        fail += 1
        failed_symbols.append(symbol)
    if (i+1) % 10 == 0:
        print(f"Progress: {i+1}/{len(all_symbols)} stocks processed.")

print("\nDownload complete.")
print(f"  ✅ Successful: {success}")
print(f"  ❌ Failed: {fail}")
if failed_symbols:
    print(f"  Failed symbols: {failed_symbols}") 