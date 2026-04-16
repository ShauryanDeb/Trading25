import pandas as pd
import numpy as np
import argparse

def add_target_column(input_file, output_file, horizon=1):
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    # Create future returns
    df['future_returns'] = df['Close'].shift(-horizon) / df['Close'] - 1
    # Binary target: 1 if next-day return > 0, else 0
    df['target'] = np.where(df['future_returns'] > 0, 1, 0)
    # Remove rows with NaN target
    df = df.dropna(subset=['target'])
    df.to_csv(output_file)
    print(f"Saved features with target to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add target column to features file")
    parser.add_argument("input_file", help="Input features CSV file")
    parser.add_argument("output_file", help="Output CSV file with target column")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (default: 1)")
    args = parser.parse_args()
    add_target_column(args.input_file, args.output_file, args.horizon) 