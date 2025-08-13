import argparse
import pandas as pd
import numpy as np
import joblib
import backtrader as bt
import sys
import os

# Add the script's directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_features import load_and_engineer_features

class MLStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('feature_names', None),
        ('threshold', 0.55),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        
        if not self.p.model or not self.p.feature_names:
            raise ValueError("Model and feature names must be provided to the strategy.")
        
        self.model = self.p.model
        self.feature_names = self.p.feature_names
        
        # Store predictions as a list that we can index by bar number
        self.predictions = None
        self.current_bar = 0
        
    def prenext(self):
        # The prenext method is called until all data and indicators are fully initialized.
        # We can use this to make all predictions at once.
        if self.predictions is None:
            # Get the feature data from the dataframe
            feature_data = self.datas[0].df[self.p.feature_names]
            # Make predictions for all bars at once
            self.predictions = self.p.model.predict_proba(feature_data)[:, 1]

    def next(self):
        if self.order:
            return  # An order is pending, do nothing

        # Get the prediction for the current bar
        if self.predictions is not None and self.current_bar < len(self.predictions):
            prob = self.predictions[self.current_bar]
        else:
            prob = 0.5  # Default to 0.5 if not found

        if not self.position:  # Not in the market
            if prob > self.p.threshold:
                self.order = self.buy()
        else:  # In the market
            if prob < (1 - self.p.threshold):
                self.order = self.sell()
        
        self.current_bar += 1
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                pass
            elif order.issell():
                pass
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass

        # Write down: no pending order
        self.order = None

def run_backtest(
    model_path: str,
    price_file: str,
    options_file: str = None,
    use_comprehensive: bool = False,
    threshold: float = 0.55,
    cash: float = 10000.0,
    commission: float = 0.001
):
    print("Loading model and data...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    
    # Use the same feature engineering logic as training
    X, y, feature_names = load_and_engineer_features(
        price_path=price_file,
        options_path=options_file,
        use_comprehensive_options=use_comprehensive
    )
    
    # The backtrader feed needs the original OHLCV data plus our features
    price_df = pd.read_csv(price_file, index_col='Date', parse_dates=True)
    
    # Align the price data with the feature data
    backtest_df = price_df.join(X, how='inner')
    
    if len(backtest_df) == 0:
        print("Error: No data available for backtesting after merging price and features.")
        return

    data_feed = bt.feeds.PandasData(dataname=backtest_df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(
        MLStrategy,
        model=model,
        feature_names=feature_names,
        threshold=threshold
    )

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print("Starting backtest...")
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    print("Backtest finished.")

    # --- Print Analysis ---
    print(f"\n--- Backtest Results ---")
    print(f"Starting Portfolio Value: {start_value:,.2f}")
    print(f"Final Portfolio Value:   {end_value:,.2f}")
    print(f"Total Return:            {(end_value / start_value - 1):.2%}")
    
    analysis = results[0].analyzers
    
    sharpe = analysis.sharpe_ratio.get_analysis()
    returns = analysis.returns.get_analysis()
    drawdown = analysis.drawdown.get_analysis()
    trades = analysis.trades.get_analysis()
    
    print("\nKey Performance Metrics:")
    print(f"  Sharpe Ratio:          {sharpe.get('sharperatio', 'N/A')}")
    print(f"  Annualized Return:     {returns.get('rnorm100', 'N/A'):.2f}%")
    print(f"  Max Drawdown:          {drawdown.max.drawdown:.2f}%")
    
    # Handle trade analysis more safely
    try:
        total_trades = trades.total.total if 'total' in trades else 0
        print(f"  Total Trades:          {total_trades}")
        
        if total_trades > 0:
            winning_trades = trades.won.total if 'won' in trades else 0
            losing_trades = trades.lost.total if 'lost' in trades else 0
            print(f"  Winning Trades:        {winning_trades}")
            print(f"  Losing Trades:         {losing_trades}")
            print(f"  Win Rate:              {(winning_trades / total_trades) * 100:.2f}%")
        else:
            print(f"  Winning Trades:        0")
            print(f"  Losing Trades:         0")
            print(f"  Win Rate:              N/A (no trades)")
    except:
        print(f"  Total Trades:          0")
        print(f"  Winning Trades:        0")
        print(f"  Losing Trades:         0")
        print(f"  Win Rate:              N/A (no trades)")
    print("--------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Enhanced backtesting with backtrader.")
    parser.add_argument("model_path", help="Path to the saved .pkl model file.")
    parser.add_argument("price_file", help="Path to the historical price data (e.g., apple.csv).")
    parser.add_argument("--options", help="Path to the comprehensive options data CSV.")
    parser.add_argument("--use-comprehensive", action='store_true',
                        help="Signal that the model uses comprehensive options features.")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Probability threshold for entering a trade.")
    
    args = parser.parse_args()
    
    run_backtest(
        model_path=args.model_path,
        price_file=args.price_file,
        options_file=args.options,
        use_comprehensive=args.use_comprehensive,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main() 