# Hold-Horizon Experiment — Design Spec

Date: 2026-07-01
Status: Approved by user (design + fallback rule); executed — see Results
Repo: `Trading25/`, branch `experiment/hold-horizon`

## Context

The swing strategy's honest walk-forward baseline (2021–2026H1, `python run.py
walkforward`, defaults `rel_5d`/`threshold`, 5 bps one-way costs, 30-symbol
`DEFAULT_UNIVERSE`, $50k chained capital) is **+35.6% total / CAGR +5.7% /
Sharpe 0.81 / PF 1.18 / max DD -10.9% / 2,357 trades**
(`reports/STRATEGY_FINDINGS_2026-07.md`, finding 2). All prior in-sample
backtests are invalid (finding 1). Out-of-sample probabilities are compressed:
median 0.462, 90% within 0.40–0.55 (finding 4). The `rel_5d`
relative-strength label beat the `abs_3d` absolute label head-to-head
(finding 5).

## Harness verification (prior-data-only retraining is enforced, not claimed)

- `pipeline/walkforward.py::_train_model` fetches training data with
  `end=train_end` (Dec 31 of the year before the test year) — post-cutoff
  prices physically never enter the training set.
- `pipeline/features.py::build_features` (with `drop_unlabeled=True`) drops
  rows whose 5-day forward label window runs past the data end, so the last
  ~5 trading days before each cutoff are excluded — an automatic purge
  between train and test.
- Macro (`^VIX`, `^TNX`, SPY context) is fetched with the same cutoff; a
  fresh `StockEnsemble` is built per segment; capital chains forward; test
  fills execute at the next day's open ± 5 bps.
- Residual caveats shared by baseline and experiment (comparisons stay fair):
  yfinance retro-adjusted prices, 2026-chosen universe (survivorship),
  30-symbol eval universe vs 100-symbol live universe.

## Hypothesis (pre-registered before any variant result was observed)

The model predicts **5-trading-day** relative strength, but `max_hold_days=3`
is compared against **calendar** days (`pipeline/backtest.py`, exit check),
force-exiting after ~1–3 trading days — before the predicted window
completes. Combined with 2,357 trades × 2 × 5 bps on ~5% position notional
(≈12% of starting capital in cost drag over 5.5 years), execution contradicts
the validated label. Holding to the label horizon should harvest more of the
predicted move per round trip and mechanically cut trade count.

Directional predictions: fewer trades, higher average PnL per trade, PF up.

## Experiment

One parameter changes: `max_hold_days`. Everything else is frozen —
thresholds (0.48/0.42/0.30), -7% stop, 5% sizing, sector caps, universe,
5 bps costs, `rel_5d` label, `threshold` entry mode.

| Run | max_hold_days | Role |
|-----|---------------|------|
| Control | 3 | Must reproduce the pristine `run.py walkforward` result exactly (validates shared-training shortcut; models are deterministic: fixed seeds, cached data) |
| **Primary** | **7** | 7 calendar days = exactly 5 trading days for any weekday entry (mod holidays) = the label horizon. Chosen by prior, not by scanning |
| Neighbor | 5 | Smoothness check only |
| Neighbor | 10 | Smoothness check only |

Implementation: `Trading25/analysis/hold_horizon_experiment.py` — trains each
segment's model once (training is invariant to hold horizon), reuses it
across variants, simulates through the pristine `run_swing_backtest`, chains
capital per variant, computes overall metrics identically to
`pipeline/walkforward.py::walk_forward`. No pipeline code is modified.

## Mechanism pre-check (before the sweep)

On the reproduced baseline trade log: exit-reason mix, hold-day distribution,
PnL by exit reason, realized cost drag. If max-hold exits are rare (signal
fade dominating), the premise is weakened — report that before burning
compute and reassess.

## Acceptance criteria (frozen)

Primary (hold=7) is accepted iff ALL of:

1. Total return > +35.6% AND profit factor > 1.18 on the honest walkforward;
2. Yearly return improves vs baseline in ≥4 of 6 segments;
3. Neighbors (5, 10) both post a higher total return than the control
   (hold=3) — a smooth response surface, not an isolated spike at 7;
4. Trade count drops ≥20% vs the control (the claimed mechanism must be
   visible in the trade count, not just the headline).

**Fallback rule (user-selected):** if the primary fails but a neighbor passes
criterion 1, the neighbor may be promoted only if it also (a) beats the
baseline at 10 bps one-way costs and (b) improves ≥4 of 6 years — and it is
reported with an explicit "selected post-hoc" caveat.

## Results (2026-07-01, appended after execution)

Control (hold=3) matched the pristine `run.py walkforward` output exactly
(+35.92% / PF 1.178 / 2,370 trades — the small drift vs the doc's +35.6% is
two extra days of data), validating the shared-training shortcut. The primary
was then re-confirmed end-to-end through the untouched
`pipeline.walkforward.walk_forward(max_hold_days=7)` — identical to the sweep
on every yearly row (`reports/walkforward_rel5d_threshold_hold7_confirm.csv`).

| Run | Total | PF | Sharpe | Max DD | Trades |
|-----|-------|----|--------|--------|--------|
| Control 3 | +35.92% | 1.178 | 0.81 | -10.9% | 2,370 |
| Neighbor 5 | +39.17% | 1.213 | 0.86 | -10.6% | 1,983 |
| **Primary 7** | **+41.94%** | **1.228** | **0.89** | **-8.2%** | **1,695** |
| Neighbor 10 | +39.91% | 1.238 | 0.83 | -8.6% | 1,354 |

All four pre-registered criteria passed: (1) +41.94% > +35.6% and PF 1.228 >
1.18; (2) 4 of 6 years improved (2022, 2023, 2024, 2025; 2021 and 2026H1 got
worse); (3) both neighbors beat the control — smooth response surface; (4)
trade count fell 28.5% (2,370 → 1,695). Avg PnL per trade rose from $7.49 to
$12.01. **ACCEPTED.** The fallback rule was not needed.

Honest caveats: 2026H1 degraded (+7.80% → +4.96%); the strategy still lags
SPY buy-and-hold (+118%) by a wide margin; PF 1.228 remains a modest edge.
Nothing here authorizes `--live` — that remains a separate, explicit decision.

## Reporting

All four runs reported, pass or fail. No result is claimed as a win unless it
beat +35.6%/PF 1.18 in the honest harness. Nothing touches `--live` or bot
configs; the only new files are the analysis script, experiment CSVs under
`Trading25/reports/`, and this spec.
