import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
import matplotlib.pyplot as plt

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

# === CONFIGURATION ===
CONFIDENCE_THRESHOLD = 0.69  # Only make a trade if confidence >= threshold (or <= 1-threshold)

def download_new_data(symbols, start_date, end_date):
    """Download fresh data for testing."""
    print(f"Downloading new data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    all_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if len(data) > 0:
                data['Symbol'] = symbol
                all_data.append(data)
                print(f"  {symbol}: {len(data)} rows")
        except Exception as e:
            print(f"  Error downloading {symbol}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=False)
        print(f"Total new data: {len(combined)} rows")
        return combined
    else:
        return None

def create_features_new_data(df):
    """Create features for new data (same as training)."""
    print("Creating features for new data...")
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    
    # Bollinger Bands
    df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    
    # Volume ratio
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Create target (1 if next day return > 0, else 0)
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def load_macro_data():
    """Load macro data for the new period."""
    try:
        macro_df = pd.read_csv('macro_features_full.csv', index_col='DATE', parse_dates=True)
        print(f"Loaded macro data: {macro_df.shape}")
        return macro_df
    except:
        print("Warning: Could not load macro data, proceeding without it")
        return pd.DataFrame()

def merge_macro_data(stock_df, macro_df):
    """Merge macro data with stock data."""
    if macro_df.empty:
        return stock_df
    
    # Convert timezone-aware datetime to timezone-naive
    stock_df_clean = stock_df.copy()
    stock_df_clean.index = stock_df_clean.index.tz_localize(None)
    
    # Use merge_asof for proper time alignment
    stock_reset = stock_df_clean.reset_index()
    macro_reset = macro_df.reset_index()
    
    merged = pd.merge_asof(
        stock_reset.sort_values('Date'),
        macro_reset.sort_values('DATE'),
        left_on='Date',
        right_on='DATE',
        direction='backward'
    )
    
    merged.set_index('Date', inplace=True)
    merged.drop('DATE', axis=1, inplace=True)
    
    # Forward fill macro features
    macro_cols = macro_df.columns.tolist()
    for col in macro_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()
    
    return merged

def test_model_on_new_data():
    """Test the trained model on completely new data."""
    print("=== Testing Model on New Data ===")
    
    # Test symbols (different from training set)
    test_symbols = ['TSLA', 'NVDA', 'AMD', 'META', 'NFLX', 'UBER', 'LYFT', 'SNAP']
    
    # Download recent data (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    # Download new data
    new_data = download_new_data(test_symbols, start_date, end_date)
    
    if new_data is None or len(new_data) == 0:
        print("Failed to download new data")
        return
    
    # Load macro data
    macro_df = load_macro_data()
    
    # Process each symbol separately, then concatenate
    processed = []
    for symbol in test_symbols:
        df_symbol = new_data[new_data['Symbol'] == symbol].copy()
        if len(df_symbol) < 60:
            continue
        df_symbol = create_features_new_data(df_symbol)
        if not macro_df.empty and len(df_symbol) > 0:
            df_symbol = merge_macro_data(df_symbol, macro_df)
        if len(df_symbol) > 0:
            processed.append(df_symbol)
    
    if not processed:
        print("No valid new data for testing after feature engineering.")
        return
    
    new_data_all = pd.concat(processed).reset_index(drop=True)
    
    # Select features (same as training)
    technical_cols = [
        'Returns', 'Log_Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 
        'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Volatility'
    ]
    macro_features = ['FedFundsRate', '10Y_Treasury', 'CPI', 'CoreCPI', 'GDP', 'VIX']
    feature_cols = technical_cols.copy()
    for feature in macro_features:
        if feature in new_data_all.columns:
            feature_cols.append(feature)
    
    # Only drop rows missing technical features
    X_new = new_data_all[feature_cols]
    mask = X_new[technical_cols].notna().all(axis=1)
    X_new = X_new[mask]
    y_new = new_data_all.loc[X_new.index, 'Target']
    
    print(f"\nNew data shape: {X_new.shape}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target distribution: {y_new.value_counts().to_dict()}")
    
    # Load the trained model
    model = None
    try:
        model = joblib.load('final_macro_test_model.pkl')
        print("✅ Loaded trained model successfully")
    except:
        print("⚠️  Could not load trained model, using a simple baseline")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Train on a small subset of new data for baseline comparison
        if len(X_new) > 100:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_new, y_new, test_size=0.3, random_state=42, stratify=y_new
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            baseline_acc = accuracy_score(y_test, y_pred)
            print(f"Baseline accuracy on new data: {baseline_acc:.4f}")
    
    # Make predictions on new data
    if len(X_new) > 0:
        y_pred_new = model.predict(X_new)
        y_pred_proba = model.predict_proba(X_new)[:, 1] if hasattr(model, 'predict_proba') else None
        accuracy_new = accuracy_score(y_new, y_pred_new)
        
        print(f"\n=== New Data Test Results ===")
        print(f"Accuracy on new data: {accuracy_new:.4f}")
        print(f"Number of predictions: {len(y_pred_new)}")
        print(f"Prediction distribution: {pd.Series(y_pred_new).value_counts().to_dict()}")
        
        # Detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(y_new, y_pred_new))
        
        # Compare with training performance
        print(f"\n=== Performance Comparison ===")
        print(f"Training CV Accuracy: ~70.5%")
        print(f"New Data Accuracy: {accuracy_new:.4f}")
        
        if accuracy_new < 0.65:
            print("⚠️  WARNING: Significant performance drop suggests overfitting!")
        elif accuracy_new < 0.70:
            print("⚠️  CAUTION: Some performance drop detected")
        else:
            print("✅ Good: Performance on new data is consistent")
        
        # 3. Analyze feature importances and errors
        analyze_model_insights(model, X_new, y_new, y_pred_new, y_pred_proba, feature_cols)
        
        # 2. Test different timeframes
        test_different_timeframes(test_symbols, macro_df, model, feature_cols)
        
        # 4. Confidence threshold analysis
        # Apply confidence threshold for trade signals
        summary_lines = []
        if y_pred_proba is not None:
            mask = (y_pred_proba >= CONFIDENCE_THRESHOLD) | (y_pred_proba <= 1-CONFIDENCE_THRESHOLD)
            trades_taken = mask.sum()
            if trades_taken > 0:
                acc = (y_new.values[mask] == y_pred_new[mask]).mean()
                summary_lines.append(f"Threshold: {CONFIDENCE_THRESHOLD:.2f}")
                summary_lines.append(f"Accuracy: {acc:.4f}")
                summary_lines.append(f"Coverage: {mask.mean():.2%}")
                summary_lines.append(f"Trades: {trades_taken}")
                summary_lines.append(f"Skipped: {len(y_new)-trades_taken}")
                # Profit analysis
                if 'Returns' in new_data_all.columns:
                    trade_returns = new_data_all.loc[X_new.index, 'Returns'][mask] * (2*(y_new.values[mask] == y_pred_new[mask]) - 1)
                    cum_profit = trade_returns.sum()
                    win_rate = (trade_returns > 0).mean()
                    sharpe = trade_returns.mean() / trade_returns.std() * np.sqrt(252) if trade_returns.std() > 0 else 0
                    summary_lines.append(f"Cumulative Profit: {cum_profit:.4f}")
                    summary_lines.append(f"Win Rate: {win_rate:.2%}")
                    summary_lines.append(f"Sharpe Ratio: {sharpe:.2f}")
            else:
                summary_lines.append(f"No trades met the confidence threshold of {CONFIDENCE_THRESHOLD:.2f}.")
        else:
            summary_lines.append("Model does not support probability/confidence output.")
        # Print summary at the end
        print("\n=== CONFIDENCE-THRESHOLD TRADE SUMMARY ===")
        for line in summary_lines:
            print(line)
        
        # 5. Tune confidence threshold for returns
        if y_pred_proba is not None and 'Returns' in new_data_all.columns:
            best_thresh = tune_confidence_threshold_for_returns(
                y_new.values, y_pred_new, y_pred_proba, new_data_all.loc[X_new.index, 'Returns'].values)
            print(f"Suggested optimal confidence threshold: {best_thresh}")
    
    else:
        print("No valid new data for testing")

def analyze_model_insights(model, X_new, y_new, y_pred_new, y_pred_proba, feature_cols):
    """Analyze feature importances and prediction errors."""
    print(f"\n=== Model Insights Analysis ===")
    
    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Feature Importances:")
        print(feature_importance.head(10))
        
        # Analyze macro vs technical features
        macro_features = [f for f in feature_cols if f in ['FedFundsRate', '10Y_Treasury', 'CPI', 'CoreCPI', 'GDP', 'VIX']]
        technical_features = [f for f in feature_cols if f not in macro_features]
        
        macro_importance = feature_importance[feature_importance['feature'].isin(macro_features)]['importance'].sum()
        technical_importance = feature_importance[feature_importance['feature'].isin(technical_features)]['importance'].sum()
        
        print(f"\nFeature Type Analysis:")
        print(f"Technical features importance: {technical_importance:.4f} ({technical_importance/(macro_importance+technical_importance)*100:.1f}%)")
        print(f"Macro features importance: {macro_importance:.4f} ({macro_importance/(macro_importance+technical_importance)*100:.1f}%)")
    
    # Error analysis
    errors = (y_new != y_pred_new)
    error_rate = errors.mean()
    print(f"\nError Analysis:")
    print(f"Overall error rate: {error_rate:.4f}")
    
    # Analyze errors by feature values
    if len(X_new) > 0:
        error_df = X_new[errors].copy()
        correct_df = X_new[~errors].copy()
        
        if len(error_df) > 0 and len(correct_df) > 0:
            print(f"\nFeature Statistics (Errors vs Correct):")
            for feature in feature_cols[:5]:  # Top 5 features
                if feature in error_df.columns and feature in correct_df.columns:
                    error_mean = error_df[feature].mean()
                    correct_mean = correct_df[feature].mean()
                    print(f"{feature}: Errors={error_mean:.4f}, Correct={correct_mean:.4f}")
    
    # Confidence analysis (if probabilities available)
    if y_pred_proba is not None:
        confidence_df = pd.DataFrame({
            'true': y_new,
            'pred': y_pred_new,
            'confidence': np.maximum(y_pred_proba, 1-y_pred_proba)
        })
        
        correct_confidence = confidence_df[confidence_df['true'] == confidence_df['pred']]['confidence'].mean()
        error_confidence = confidence_df[confidence_df['true'] != confidence_df['pred']]['confidence'].mean()
        
        print(f"\nConfidence Analysis:")
        print(f"Correct predictions avg confidence: {correct_confidence:.4f}")
        print(f"Error predictions avg confidence: {error_confidence:.4f}")

def test_different_timeframes(test_symbols, macro_df, model, feature_cols):
    """Test the model on different timeframes."""
    print(f"\n=== Testing Different Timeframes ===")
    
    timeframes = [
        ("Last 6 months", 180),
        ("Last 1 year", 365),
        ("Last 1.5 years", 547),
        ("Last 2 years", 730)
    ]
    
    results = []
    
    for timeframe_name, days in timeframes:
        print(f"\nTesting {timeframe_name}...")
        
        # Download data for this timeframe
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        new_data = download_new_data(test_symbols[:4], start_date, end_date)  # Use fewer symbols for speed
        
        if new_data is None or len(new_data) == 0:
            continue
        
        # Process data
        processed = []
        for symbol in test_symbols[:4]:
            df_symbol = new_data[new_data['Symbol'] == symbol].copy()
            if len(df_symbol) < 60:
                continue
            df_symbol = create_features_new_data(df_symbol)
            if not macro_df.empty and len(df_symbol) > 0:
                df_symbol = merge_macro_data(df_symbol, macro_df)
            if len(df_symbol) > 0:
                processed.append(df_symbol)
        
        if not processed:
            continue
        
        new_data_all = pd.concat(processed).reset_index(drop=True)
        
        # Prepare features
        technical_cols = [
            'Returns', 'Log_Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 
            'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Volatility'
        ]
        
        X_new = new_data_all[feature_cols]
        mask = X_new[technical_cols].notna().all(axis=1)
        X_new = X_new[mask]
        y_new = new_data_all.loc[X_new.index, 'Target']
        
        if len(X_new) > 0:
            y_pred_new = model.predict(X_new)
            accuracy_new = accuracy_score(y_new, y_pred_new)
            
            results.append({
                'timeframe': timeframe_name,
                'samples': len(X_new),
                'accuracy': accuracy_new
            })
            
            print(f"  {timeframe_name}: {len(X_new)} samples, accuracy={accuracy_new:.4f}")
    
    # Summary of timeframe results
    if results:
        print(f"\nTimeframe Performance Summary:")
        for result in results:
            print(f"  {result['timeframe']}: {result['accuracy']:.4f} ({result['samples']} samples)")

def real_time_prediction_performance(model, symbols, macro_df, feature_cols, days_to_evaluate=5, context_days=60):
    """Fetch real-time data, make daily predictions using only past data, and measure performance."""
    print(f"\n=== Real-Time Prediction Performance (last {days_to_evaluate} days) ===")
    from datetime import timedelta
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report

    all_results = []
    for symbol in symbols:
        print(f"\nSymbol: {symbol}")
        # Download enough data for context + evaluation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=context_days + days_to_evaluate + 10)
        df = yf.Ticker(symbol).history(start=start_date, end=end_date)
        if len(df) < context_days + days_to_evaluate:
            print(f"  Not enough data for {symbol}")
            continue
        df['Symbol'] = symbol
        df = create_features_new_data(df)
        if not macro_df.empty and len(df) > 0:
            df = merge_macro_data(df, macro_df)
        if len(df) == 0:
            print(f"  No valid feature rows for {symbol}")
            continue
        # Only drop rows missing technical features
        technical_cols = [
            'Returns', 'Log_Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Volatility'
        ]
        mask = df[technical_cols].notna().all(axis=1)
        df = df[mask]
        if len(df) < days_to_evaluate + 1:
            print(f"  Not enough valid rows for {symbol}")
            continue
        # Evaluate only the last N days
        eval_rows = df.iloc[-days_to_evaluate-1:-1].copy()
        next_day_returns = df['Returns'].iloc[-days_to_evaluate:].values
        true_next = (next_day_returns > 0).astype(int)
        # Predict for each day using only past data
        preds = []
        for i, idx in enumerate(eval_rows.index):
            X_row = eval_rows.loc[idx, feature_cols].values.reshape(1, -1)
            pred = model.predict(X_row)[0]
            preds.append(pred)
        # Metrics
        acc = accuracy_score(true_next, preds)
        print(f"  Real-time accuracy (last {days_to_evaluate} days): {acc:.4f}")
        print(f"  Prediction distribution: {pd.Series(preds).value_counts().to_dict()}")
        print(f"  True distribution: {pd.Series(true_next).value_counts().to_dict()}")
        print(classification_report(true_next, preds))
        all_results.extend(list(zip([symbol]*days_to_evaluate, preds, true_next)))
    # Aggregate
    if all_results:
        df_results = pd.DataFrame(all_results, columns=['Symbol', 'Pred', 'True'])
        overall_acc = accuracy_score(df_results['True'], df_results['Pred'])
        print(f"\n=== Aggregate Real-Time Accuracy (all symbols): {overall_acc:.4f}")
        print(df_results.groupby('Symbol').apply(lambda x: accuracy_score(x['True'], x['Pred'])))
    else:
        print("No real-time results to report.")

def confidence_threshold_analysis(y_true, y_pred, y_proba):
    """Analyze and visualize accuracy vs. confidence threshold tradeoff."""
    print("\n=== Confidence Threshold Analysis ===")
    thresholds = np.arange(0.5, 0.96, 0.05)
    accuracies = []
    coverages = []
    for thresh in thresholds:
        mask = (y_proba >= thresh) | (y_proba <= 1-thresh)
        if mask.sum() == 0:
            accuracies.append(np.nan)
            coverages.append(0)
            continue
        acc = (y_true[mask] == y_pred[mask]).mean()
        accuracies.append(acc)
        coverages.append(mask.mean())
        print(f"Threshold: {thresh:.2f} | Accuracy: {acc:.4f} | Coverage: {mask.mean():.2%} | Trades: {mask.sum()}")
    # Plot
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(thresholds, accuracies, color=color, marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Coverage (Fraction of Trades)', color=color)
    ax2.plot(thresholds, coverages, color=color, marker='x', label='Coverage')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Accuracy and Coverage vs. Confidence Threshold')
    fig.tight_layout()
    plt.show()

def tune_confidence_threshold_for_returns(y_true, y_pred, y_proba, returns):
    """Tune the confidence threshold to maximize cumulative return and Sharpe ratio."""
    print("\n=== Confidence Threshold Return Tuning ===")
    thresholds = np.arange(0.5, 0.96, 0.05)
    results = []
    for thresh in thresholds:
        mask = (y_proba >= thresh) | (y_proba <= 1-thresh)
        if mask.sum() == 0:
            continue
        # Simulate trades: +return if correct, -return if wrong
        trade_returns = returns[mask] * (2*(y_true[mask] == y_pred[mask]) - 1)
        cum_return = trade_returns.sum()
        win_rate = (trade_returns > 0).mean()
        if trade_returns.std() > 0:
            sharpe = trade_returns.mean() / trade_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        results.append({
            'threshold': thresh,
            'accuracy': (y_true[mask] == y_pred[mask]).mean(),
            'coverage': mask.mean(),
            'trades': mask.sum(),
            'cum_return': cum_return,
            'win_rate': win_rate,
            'sharpe': sharpe
        })
        print(f"Threshold: {thresh:.2f} | Accuracy: {results[-1]['accuracy']:.4f} | Coverage: {results[-1]['coverage']:.2%} | Trades: {results[-1]['trades']} | Return: {cum_return:.4f} | Sharpe: {sharpe:.2f} | Win rate: {win_rate:.2%}")
    # Find best threshold by total return
    if results:
        best_return = max(results, key=lambda x: x['cum_return'])
        best_sharpe = max(results, key=lambda x: x['sharpe'])
        print(f"\nBest threshold by Total Return: {best_return['threshold']:.2f} | Return: {best_return['cum_return']:.4f} | Sharpe: {best_return['sharpe']:.2f} | Accuracy: {best_return['accuracy']:.4f} | Coverage: {best_return['coverage']:.2%}")
        print(f"Best threshold by Sharpe ratio: {best_sharpe['threshold']:.2f} | Sharpe: {best_sharpe['sharpe']:.2f} | Return: {best_sharpe['cum_return']:.4f} | Accuracy: {best_sharpe['accuracy']:.4f} | Coverage: {best_sharpe['coverage']:.2%}")
        return best_return['threshold']
    else:
        print("No trades for any threshold.")
        return None

if __name__ == "__main__":
    test_model_on_new_data()
    # Real-time prediction performance (optional, can comment/uncomment)
    try:
        model = joblib.load('final_macro_test_model.pkl')
        macro_df = load_macro_data()
        technical_cols = [
            'Returns', 'Log_Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'Volatility'
        ]
        macro_features = ['FedFundsRate', '10Y_Treasury', 'CPI', 'CoreCPI', 'GDP', 'VIX']
        feature_cols = technical_cols.copy()
        for feature in macro_features:
            if feature in macro_df.columns:
                feature_cols.append(feature)
        real_time_symbols = ['TSLA', 'NVDA', 'AMD', 'META', 'NFLX', 'UBER', 'LYFT', 'SNAP']
        real_time_prediction_performance(model, real_time_symbols, macro_df, feature_cols, days_to_evaluate=5, context_days=60)
    except Exception as e:
        print(f"[Real-time prediction] Could not run: {e}") 