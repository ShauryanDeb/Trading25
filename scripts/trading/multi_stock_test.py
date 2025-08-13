import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from enhanced_features import create_feature_matrix, add_option_features
from comprehensive_options import fetch_comprehensive_options_data

# Top stocks by market cap and sector diversity
TOP_STOCKS = {
    # Technology
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc.',
    
    # Financial
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corp.',
    'WFC': 'Wells Fargo & Company',
    
    # Healthcare
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc.',
    'UNH': 'UnitedHealth Group Inc.',
    
    # Consumer
    'KO': 'The Coca-Cola Company',
    'PG': 'Procter & Gamble Co.',
    'WMT': 'Walmart Inc.',
    'HD': 'The Home Depot Inc.',
    
    # Energy
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    
    # Industrial
    'JNJ': 'Johnson & Johnson',
    'BA': 'Boeing Co.',
    
    # Communication
    'NFLX': 'Netflix Inc.',
    'DIS': 'The Walt Disney Company'
}

def download_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download stock data for a given ticker."""
    try:
        print(f"Downloading data for {ticker}...")
        df = yf.download(ticker, period=period, progress=False)
        
        if df.empty:
            print(f"No data found for {ticker}")
            return pd.DataFrame()
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.index.name = 'Date'
        df.reset_index(inplace=True)
        df.set_index('Date', inplace=True)
        
        print(f"Downloaded {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()

def get_options_data(ticker: str) -> pd.DataFrame:
    """Get current options data for a ticker."""
    try:
        print(f"Getting options data for {ticker}...")
        df = fetch_comprehensive_options_data(ticker)
        if not df.empty:
            print(f"Got options data for {ticker}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error getting options data for {ticker}: {e}")
        return pd.DataFrame()

def prepare_stock_data(price_df: pd.DataFrame, options_df: pd.DataFrame = None) -> tuple[pd.DataFrame, list]:
    """Prepare stock data with features."""
    if price_df.empty:
        return pd.DataFrame(), []
    
    # Create feature matrix
    include_options = options_df is not None and not options_df.empty
    
    if include_options:
        # Add options features
        price_df = add_option_features(price_df, options_df, "auto")
    
    # Create feature matrix
    df, features = create_feature_matrix(price_df, include_options=include_options)
    
    return df, features

def create_target_variable(df: pd.DataFrame) -> pd.Series:
    """Create binary target variable."""
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df['Target']

def evaluate_stock_performance(df: pd.DataFrame, features: list, ticker: str) -> dict:
    """Evaluate model performance for a single stock."""
    if df.empty or not features:
        return {'ticker': ticker, 'status': 'failed', 'error': 'No data or features'}
    
    # Create target
    target = create_target_variable(df)
    
    # Prepare data
    valid_data = df[features + ['Target']].dropna()
    
    if len(valid_data) < 100:
        return {'ticker': ticker, 'status': 'failed', 'error': f'Insufficient data: {len(valid_data)} samples'}
    
    X = valid_data[features]
    y = valid_data['Target']
    
    # Split data (80% train, 20% test)
    split_idx = int(len(valid_data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    # Calculate returns
    test_prices = df.iloc[split_idx:]['Close']
    returns = []
    
    for i in range(len(y_pred)):
        if i < len(test_prices) - 1:
            current_price = test_prices.iloc[i]
            next_price = test_prices.iloc[i + 1]
            
            if y_pred[i] == 1:  # Buy signal
                return_val = (next_price - current_price) / current_price
            else:  # Hold
                return_val = 0
            
            returns.append(return_val)
    
    cumulative_return = np.sum(returns) if returns else 0
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        # Use the features that were actually used in training
        used_features = X.columns.tolist()
        feature_importance = pd.DataFrame({
            'feature': used_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = pd.DataFrame()
    
    return {
        'ticker': ticker,
        'status': 'success',
        'samples': len(valid_data),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(features),
        'accuracy': accuracy,
        'cumulative_return': cumulative_return,
        'avg_return': np.mean(returns) if returns else 0,
        'volatility': np.std(returns) if returns else 0,
        'sharpe_ratio': (np.mean(returns) / np.std(returns)) if returns and np.std(returns) > 0 else 0,
        'top_features': feature_importance.head(5)['feature'].tolist(),
        'has_options': len([f for f in features if 'Call' in f or 'Put' in f or 'IV' in f]) > 0
    }

def run_comprehensive_test(stocks: dict, use_options: bool = True, output_file: str = "comprehensive_test_results.csv"):
    """Run comprehensive test on multiple stocks."""
    print(f"Starting comprehensive test on {len(stocks)} stocks...")
    print(f"Options data: {'Enabled' if use_options else 'Disabled'}")
    
    results = []
    
    for ticker, name in stocks.items():
        print(f"\n{'='*50}")
        print(f"Processing {ticker} ({name})")
        print(f"{'='*50}")
        
        # Download price data
        price_df = download_stock_data(ticker)
        if price_df.empty:
            results.append({'ticker': ticker, 'status': 'failed', 'error': 'No price data'})
            continue
        
        # Get options data if requested
        options_df = None
        if use_options:
            options_df = get_options_data(ticker)
        
        # Prepare data
        df, features = prepare_stock_data(price_df, options_df)
        
        # Evaluate performance
        result = evaluate_stock_performance(df, features, ticker)
        results.append(result)
        
        # Print summary
        if result['status'] == 'success':
            print(f"✅ {ticker}: Accuracy={result['accuracy']:.3f}, Return={result['cumulative_return']:.4f}")
        else:
            print(f"❌ {ticker}: {result.get('error', 'Unknown error')}")
        
        # Rate limiting
        time.sleep(1)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    return results_df

def analyze_results(results_df: pd.DataFrame):
    """Analyze and summarize test results."""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Filter successful tests
    successful = results_df[results_df['status'] == 'success']
    failed = results_df[results_df['status'] == 'failed']
    
    print(f"Total stocks tested: {len(results_df)}")
    print(f"Successful tests: {len(successful)}")
    print(f"Failed tests: {len(failed)}")
    
    if len(successful) == 0:
        print("No successful tests to analyze")
        return
    
    # Performance metrics
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"Average Accuracy: {successful['accuracy'].mean():.3f}")
    print(f"Accuracy Std Dev: {successful['accuracy'].std():.3f}")
    print(f"Min Accuracy: {successful['accuracy'].min():.3f}")
    print(f"Max Accuracy: {successful['accuracy'].max():.3f}")
    
    print(f"\n💰 RETURN METRICS:")
    print(f"Average Cumulative Return: {successful['cumulative_return'].mean():.4f}")
    print(f"Average Return per Trade: {successful['avg_return'].mean():.4f}")
    print(f"Average Volatility: {successful['volatility'].mean():.4f}")
    print(f"Average Sharpe Ratio: {successful['sharpe_ratio'].mean():.4f}")
    
    # Feature analysis
    stocks_with_options = successful[successful['has_options'] == True]
    stocks_without_options = successful[successful['has_options'] == False]
    
    print(f"\n🔧 FEATURE ANALYSIS:")
    print(f"Stocks with options data: {len(stocks_with_options)}")
    print(f"Stocks without options data: {len(stocks_without_options)}")
    
    if len(stocks_with_options) > 0:
        print(f"Options stocks avg accuracy: {stocks_with_options['accuracy'].mean():.3f}")
    if len(stocks_without_options) > 0:
        print(f"Non-options stocks avg accuracy: {stocks_without_options['accuracy'].mean():.3f}")
    
    # Top performers
    print(f"\n🏆 TOP PERFORMERS (by accuracy):")
    top_accuracy = successful.nlargest(5, 'accuracy')[['ticker', 'accuracy', 'cumulative_return']]
    print(top_accuracy.to_string(index=False))
    
    print(f"\n💰 TOP PERFORMERS (by returns):")
    top_returns = successful.nlargest(5, 'cumulative_return')[['ticker', 'accuracy', 'cumulative_return']]
    print(top_returns.to_string(index=False))
    
    # Sector analysis (if we have sector info)
    print(f"\n📈 SECTOR ANALYSIS:")
    print("Note: Detailed sector analysis would require additional sector mapping")
    
    return successful

def main():
    parser = argparse.ArgumentParser(description="Comprehensive multi-stock testing")
    parser.add_argument("--stocks", nargs="+", help="Specific stocks to test")
    parser.add_argument("--no-options", action="store_true", help="Disable options data")
    parser.add_argument("--output", default="comprehensive_test_results.csv", help="Output file")
    
    args = parser.parse_args()
    
    # Determine stocks to test
    if args.stocks:
        stocks_to_test = {ticker: TOP_STOCKS.get(ticker, ticker) for ticker in args.stocks}
    else:
        stocks_to_test = TOP_STOCKS
    
    print(f"Testing {len(stocks_to_test)} stocks: {list(stocks_to_test.keys())}")
    
    # Run test
    results_df = run_comprehensive_test(stocks_to_test, not args.no_options, args.output)
    
    # Analyze results
    successful_results = analyze_results(results_df)
    
    # Save detailed analysis
    if successful_results is not None and not successful_results.empty:
        analysis_file = args.output.replace('.csv', '_analysis.csv')
        successful_results.to_csv(analysis_file, index=False)
        print(f"\nDetailed analysis saved to {analysis_file}")

if __name__ == "__main__":
    main() 