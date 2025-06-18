import argparse
import joblib
import pandas as pd
import backtrader as bt
from features import add_technical_indicators, add_option_features, OPTION_FEATURES

FEATURE_COLS = [
    'MA_20', 'MA_50', 'EMA_20', 'EMA_50',
    'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
    'RSI_14', 'Stoch_%K', 'Stoch_%D', 'ATR_14',
    'CCI_20', 'OBV',
] + OPTION_FEATURES

class ModelStrategy(bt.Strategy):
    params = dict(threshold=0.6, stake=1.0)

    def __init__(self):
        self.prob = self.datas[0].prob

    def next(self):
        p = self.prob[0]
        if not self.position and p > self.p.threshold:
            self.buy(size=self.p.stake)
        elif self.position and p <= self.p.threshold:
            self.close()

def run_backtest(csv_path: str, model_path: str, threshold: float, stake: float, commission: float, options_path: str | None = None):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    if options_path:
        opt_df = pd.read_csv(options_path, index_col=0, parse_dates=True)
        df = add_option_features(df, opt_df)
    df.dropna(inplace=True)

    model = joblib.load(model_path)
    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df['prob'] = probs

    data = bt.feeds.PandasData(dataname=df)
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(ModelStrategy, threshold=threshold, stake=stake)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=commission)

    start_val = cerebro.broker.getvalue()
    cerebro.run()
    end_val = cerebro.broker.getvalue()
    print(f"Starting portfolio value: {start_val:.2f}")
    print(f"Final portfolio value: {end_val:.2f}")
    print(f"Return: {(end_val / start_val - 1) * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Backtest model with backtrader")
    parser.add_argument('model', help='Trained model file')
    parser.add_argument('csv', help='CSV with historical data')
    parser.add_argument('--threshold', type=float, default=0.6, help='Buy probability threshold')
    parser.add_argument('--stake', type=float, default=1.0, help='Trade size per order')
    parser.add_argument('--commission', type=float, default=0.0, help='Broker commission as decimal')
    parser.add_argument('--options', help='CSV with option features')
    args = parser.parse_args()

    run_backtest(args.csv, args.model, args.threshold, args.stake, args.commission, args.options)

if __name__ == '__main__':
    main()
