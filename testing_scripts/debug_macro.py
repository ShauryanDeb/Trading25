import os
import pandas as pd
import numpy as np
import sys

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.enhanced_features import load_and_engineer_features

def debug_single_file():
    """Debug feature engineering with a single macro file."""
    print("Debugging single macro file...")
    
    # Get first macro file
    files = [f for f in os.listdir('.') if f.endswith('_data_full_macro.csv')]
    if not files:
        print("No macro files found!")
        return
    
    test_file = files[0]
    print(f"Testing file: {test_file}")
    
    try:
        # Load raw data first
        df = pd.read_csv(test_file, index_col=0, parse_dates=True)
        print(f"Raw data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return
        
        # Try feature engineering
        print("\nTrying feature engineering...")
        X, y, feature_names = load_and_engineer_features(
            price_path=test_file,
            options_path=None,
            use_comprehensive_options=False
        )
        
        print(f"Feature engineering successful!")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        # Check macro features
        macro_features = [col for col in feature_names if col in ['FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI', 'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 'IndustrialProduction', 'RetailSales', 'ConsumerConfidence', 'ISM_Manufacturing', 'VIX', 'SP500']]
        print(f"Macro features in final dataset: {len(macro_features)}")
        print(f"Macro features: {macro_features}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_single_file() 