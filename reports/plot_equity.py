"""Plot equity curves from backtest CSV files saved in reports/.

Usage:
    python -m reports.plot_equity                      # all CSVs in reports/
    python -m reports.plot_equity --dir path/to/csvs  # custom directory
    python -m reports.plot_equity --out chart.png      # output path
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

REPORTS_DIR = Path(__file__).parent
BENCHMARK_SYMBOL = "AAPL"


def load_equity_curves(directory: Path) -> dict[str, pd.Series]:
    """Load all equity-curve CSVs from *directory*.

    Each CSV must have a DatetimeIndex column and an 'Equity' or 'equity'
    column (case-insensitive).  File stem becomes the series label.
    """
    curves: dict[str, pd.Series] = {}
    for csv in sorted(directory.glob("*.csv")):
        try:
            df = pd.read_csv(csv, index_col=0, parse_dates=True)
            col = next(
                (c for c in df.columns if c.lower() in {"equity", "equity_curve"}),
                df.columns[0],
            )
            series = df[col].dropna().astype(float)
            if len(series) > 1:
                curves[csv.stem] = series
        except Exception as e:
            print(f"  Skipping {csv.name}: {e}")
    return curves


def _aapl_bah(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return AAPL buy-and-hold equity curve normalised to 1.0."""
    try:
        df = yf.download(
            BENCHMARK_SYMBOL,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        close = df["Close"].dropna()
        return close / close.iloc[0]
    except Exception:
        return pd.Series(dtype=float)


def plot(
    curves: dict[str, pd.Series],
    output: Path,
    title: str = "Equity Curves",
    include_benchmark: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    all_dates = pd.concat(list(curves.values())).index
    start, end = all_dates.min(), all_dates.max()

    for label, equity in curves.items():
        normed = equity / equity.iloc[0]
        ax.plot(normed.index, normed.values, linewidth=1.5, label=label)

    if include_benchmark:
        bah = _aapl_bah(start, end)
        if not bah.empty:
            ax.plot(bah.index, bah.values, linewidth=2, linestyle="--",
                    color="black", label=f"{BENCHMARK_SYMBOL} B&H")

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised Equity (start = 1.0)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Chart saved → {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot equity curves")
    parser.add_argument("--dir", default=str(REPORTS_DIR),
                        help="Directory containing equity-curve CSVs")
    parser.add_argument("--out", default=str(REPORTS_DIR / "equity_curves.png"),
                        help="Output PNG path")
    parser.add_argument("--title", default="Equity Curves", help="Chart title")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Omit AAPL buy-and-hold benchmark")
    args = parser.parse_args()

    directory = Path(args.dir)
    curves = load_equity_curves(directory)
    if not curves:
        print(f"No equity-curve CSVs found in {directory}")
        return

    print(f"Loaded {len(curves)} equity curve(s): {list(curves.keys())}")
    plot(
        curves,
        output=Path(args.out),
        title=args.title,
        include_benchmark=not args.no_benchmark,
    )


if __name__ == "__main__":
    main()
