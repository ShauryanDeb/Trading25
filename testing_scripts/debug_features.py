import os
import pandas as pd
import numpy as np
import sys

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.enhanced_features import load_and_engineer_features

def debug_feature_engineering():
    """Debug feature engineering step by step."""
    print("Debugging feature engineering...")
    
    # Test with a single file
    test_file = 'baba_data_full_macro_fixed.csv'
    print(f"Testing file: {test_file}")
    
    # Load raw data
    df = pd.read_csv(test_file, index_col=0, parse_dates=True)
    print(f"Raw data shape: {df.shape}")
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return
    
    # Check for NaN in price data
    price_nan = df[required_cols].isnull().sum()
    print(f"NaN in price data: {price_nan.to_dict()}")
    
    # Remove VIX and SP500 if they're all NaN
    if 'VIX' in df.columns and df['VIX'].isnull().all():
        df = df.drop('VIX', axis=1)
        print("Removed VIX (all NaN)")
    if 'SP500' in df.columns and df['SP500'].isnull().all():
        df = df.drop('SP500', axis=1)
        print("Removed SP500 (all NaN)")
    
    # Save cleaned data
    cleaned_file = 'temp_cleaned.csv'
    df.to_csv(cleaned_file)
    print(f"Saved cleaned data to {cleaned_file}")
    
    try:
        # Try feature engineering
        print("\nTrying feature engineering...")
        X, y, feature_names = load_and_engineer_features(
            price_path=cleaned_file,
            options_path=None,
            use_comprehensive_options=False
        )
        
        print(f"Feature engineering result:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Features: {len(feature_names)}")
        
        if len(X) > 0:
            print("SUCCESS! Feature engineering worked.")
            
            # Check macro features
            macro_features = [col for col in feature_names if col in ['FedFundsRate', '10Y_Treasury', '2Y_Treasury', 'CPI', 'CoreCPI', 'PPI', 'UnemploymentRate', 'NonfarmPayrolls', 'InitialClaims', 'GDP', 'IndustrialProduction', 'RetailSales', 'ConsumerConfidence', 'ISM_Manufacturing']]
            print(f"  Macro features: {len(macro_features)}")
            print(f"  Macro features: {macro_features}")
            
            return True
        else:
            print("FAILED! Feature engineering produced 0 samples.")
            return False
            
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(cleaned_file):
            os.remove(cleaned_file)

if __name__ == "__main__":
    debug_feature_engineering() 