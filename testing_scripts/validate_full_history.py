import os
import pandas as pd
import numpy as np
from datetime import datetime

stock_files = [f for f in os.listdir('.') if f.endswith('_data_full.csv')]

print(f"Validating {len(stock_files)} full-history stock files...")

summary = []

def strip_timezone(idx):
    # Remove timezone info from pandas DatetimeIndex
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_localize(None)
    return idx

for i, file in enumerate(stock_files):
    symbol = file.replace('_data_full.csv', '').upper()
    try:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        orig_len = len(df)
        print(f"[{i+1}/{len(stock_files)}] {symbol}: {orig_len} rows")

        # Remove timezone info from index
        df.index = strip_timezone(df.index)

        # Check for duplicate rows
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            print(f"  ⚠️  {dupes} duplicate rows found. Removing...")
            df = df[~df.index.duplicated(keep='first')]

        # Check for missing dates (weekdays)
        all_days = pd.date_range(df.index.min(), df.index.max(), freq='B')
        all_days = strip_timezone(all_days)
        missing = set(all_days) - set(df.index)
        print(f"  📅 Missing trading days: {len(missing)}")

        # Check for NaNs or infinite values
        nans = df.isna().sum().sum()
        infs = np.isinf(df.values).sum()
        if nans > 0 or infs > 0:
            print(f"  ⚠️  {nans} NaNs, {infs} infinite values found. Filling...")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill')

        # Check for outliers in price/volume
        price_cols = [c for c in ['Open','High','Low','Close'] if c in df.columns]
        vol_col = 'Volume' if 'Volume' in df.columns else None
        outlier_count = 0
        for col in price_cols + ([vol_col] if vol_col else []):
            if col:
                series = df[col]
                z = (series - series.mean()) / series.std()
                outliers = (np.abs(z) > 6).sum()
                if outliers > 0:
                    print(f"  ⚠️  {outliers} outliers detected in {col}. Clipping...")
                    df[col] = series.clip(lower=series.quantile(0.001), upper=series.quantile(0.999))
                outlier_count += outliers

        # Save cleaned file
        cleaned_file = file.replace('_data_full.csv', '_data_full_clean.csv')
        df.to_csv(cleaned_file)
        print(f"  ✅ Cleaned data saved to {cleaned_file}")

        summary.append({
            'Symbol': symbol,
            'OriginalRows': orig_len,
            'Duplicates': dupes,
            'MissingDays': len(missing),
            'NaNs': nans,
            'Infs': infs,
            'Outliers': outlier_count,
            'FinalRows': len(df)
        })
    except Exception as e:
        print(f"  ❌ Error processing {symbol}: {e}")
        summary.append({
            'Symbol': symbol,
            'Error': str(e)
        })
    if (i+1) % 10 == 0:
        print(f"Progress: {i+1}/{len(stock_files)} files processed.")

# Save summary report
summary_df = pd.DataFrame(summary)
summary_df.to_csv('full_history_validation_report.csv', index=False)
print("\nValidation complete. Summary saved to full_history_validation_report.csv") 