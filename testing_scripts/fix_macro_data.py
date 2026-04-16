import os
import pandas as pd
import numpy as np
from datetime import datetime

def fix_macro_data():
    """Fix macro data quality issues."""
    print("Fixing macro data quality...")
    
    # Get macro files
    files = [f for f in os.listdir('.') if f.endswith('_data_full_macro.csv')]
    print(f"Found {len(files)} macro files")
    
    for i, file in enumerate(files):
        try:
            print(f"\nProcessing {i+1}/{len(files)}: {file}")
            
            # Load data
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            original_shape = df.shape
            
            # Forward-fill macro features
            macro_features = ['FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI', 
                             'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 
                             'IndustrialProduction', 'RetailSales', 'ConsumerConfidence', 'ISM_Manufacturing', 'VIX', 'SP500']
            
            for feature in macro_features:
                if feature in df.columns:
                    # Forward-fill and then back-fill
                    df[feature] = df[feature].ffill().bfill()
            
            # Save fixed data
            fixed_file = file.replace('_data_full_macro.csv', '_data_full_macro_fixed.csv')
            df.to_csv(fixed_file)
            
            # Check improvement
            print(f"  Original shape: {original_shape}")
            print(f"  Fixed shape: {df.shape}")
            
            # Check recent data quality
            recent = df.tail(100)
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            valid_recent = recent[price_cols + ['VIX', 'SP500']].dropna()
            print(f"  Recent valid rows: {len(valid_recent)}/100")
            
        except Exception as e:
            print(f"  Error processing {file}: {e}")
    
    print(f"\nFixed {len(files)} macro files")

if __name__ == "__main__":
    fix_macro_data() 