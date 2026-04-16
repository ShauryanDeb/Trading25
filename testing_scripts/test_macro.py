import os
import pandas as pd
import numpy as np
from datetime import datetime

def test_macro_files():
    """Test macro-enriched files."""
    print("Testing macro-enriched files...")
    
    # Get macro files
    files = [f for f in os.listdir('.') if f.endswith('_data_full_macro.csv')]
    print(f"Found {len(files)} macro files")
    
    if len(files) == 0:
        print("No macro files found!")
        return
    
    # Test first file
    test_file = files[0]
    print(f"Testing file: {test_file}")
    
    try:
        df = pd.read_csv(test_file, index_col=0, parse_dates=True)
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Check for macro features
        macro_features = ['FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI', 
                         'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 
                         'IndustrialProduction', 'RetailSales', 'ConsumerConfidence', 'ISM_Manufacturing', 'VIX', 'SP500']
        
        available_macro = [col for col in macro_features if col in df.columns]
        print(f"Available macro features: {len(available_macro)}")
        print(f"Macro features: {available_macro}")
        
        # Show sample data
        print("\nSample macro data:")
        if available_macro:
            sample_data = df[available_macro].head()
            print(sample_data)
        
        return True
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

if __name__ == "__main__":
    test_macro_files() 