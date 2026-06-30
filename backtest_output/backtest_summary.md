# Backtest Summary

## Executive Summary
- Best Sharpe: **gamma_scalper** at **0.00**.
- Best net P&L: **gamma_scalper**.
- Most active strategy: **ensemble_ai**.
- Synthetic option premiums are a proxy, not a substitute for real Sensex option chain data.
- **Warning: Survivorship Bias.** This backtest only includes currently liquid symbols. Historical delistings or mergers are not modeled.
- Strategies marked NOT READY should remain paper-only until validated on broker-grade intraday data.

## ensemble_ai

| Metric | Value |
|---|---|
| Total Net Pnl | -4478.93 |
| Total Net Pnl Pct | -8.96 |
| Annualised Return Pct | -16.40 |
| Max Drawdown Inr | -4826.50 |
| Max Drawdown Pct | 9.65 |
| Sharpe | -4.76 |
| Profit Factor | 0.38 |
| Win Rate Pct | 30.16 |
| Total Trades | 63.00 |
| Avg R Multiple | -0.60 |
| Daily Loss Limit Days | 0.00 |

### Monthly Breakdown
| Month | Trades | Wins | Losses | Win Rate | Net P&L | Drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-11 | 4 | 0 | 4 | 0.00 | -599.47 | -355.78 |
| 2025-12 | 18 | 5 | 13 | 27.78 | -1472.66 | -1603.37 |
| 2026-01 | 18 | 6 | 12 | 33.33 | -1713.48 | -1538.38 |
| 2026-03 | 6 | 1 | 5 | 16.67 | -714.76 | -284.16 |
| 2026-04 | 14 | 6 | 8 | 42.86 | 300.62 | -259.93 |
| 2026-05 | 3 | 1 | 2 | 33.33 | -279.18 | -180.39 |

### Top 3 Best Trades
| date | symbol | net_pnl | outcome |
| --- | --- | --- | --- |
| 2026-04-27 | RELIANCE | 348.59 | TARGET_HIT |
| 2026-04-07 | WIPRO | 303.99 | TARGET_HIT |
| 2026-03-18 | WIPRO | 270.03 | TARGET_HIT |

### Top 3 Worst Trades
| date | symbol | net_pnl | outcome |
| --- | --- | --- | --- |
| 2026-03-09 | WIPRO | -323.21 | SL_HIT |
| 2026-04-07 | WIPRO | -306.76 | SL_HIT |
| 2026-04-02 | WIPRO | -292.29 | SL_HIT |

### Parameter Sensitivity
| Parameter Value | Current | Trades | Win Rate | Profit Factor | Sharpe | Unreliable |
| --- | --- | --- | --- | --- | --- | --- |
| 0.55 | False | 91 | 31.87 | 0.38 | -4.91 | False |
| 0.58 | False | 80 | 27.50 | 0.30 | -5.77 | False |
| 0.60 | False | 69 | 30.43 | 0.38 | -5.02 | False |
| 0.62 | True | 63 | 30.16 | 0.38 | -4.76 | False |
| 0.65 | False | 57 | 31.58 | 0.44 | -4.62 | False |
| 0.68 | False | 48 | 33.33 | 0.50 | -3.75 | False |
| 0.72 | False | 33 | 33.33 | 0.63 | -3.24 | False |

### Quant Review Flags
- **Win rate > 45%:** FAIL
- **Profit factor > 1.2:** FAIL
- **Sharpe ratio > 0.8:** FAIL
- **Max drawdown < 20% capital:** PASS
- **Trades > 30:** PASS
- **No month > 40% profit:** FAIL
- **Monthly win rate std < 15pp:** PASS
- **Sensitivity has flat region:** FAIL
- **Overall verdict:** NOT READY

### Honest Assessment
The strategy is not ready. Either trade count, risk-adjusted return, drawdown, stability, or parameter robustness is insufficient.

### Recommended Next Steps
- Validate against broker-grade 5m/15m historical candles.
- Replace simulated AI scores with walk-forward trained models.
- Re-run with realistic liquidity, taxes, and symbol-specific slippage.
- Keep live capital disabled until out-of-sample performance is stable.

## rsmb

| Metric | Value |
|---|---|
| Total Net Pnl | -4021.07 |
| Total Net Pnl Pct | -8.04 |
| Annualised Return Pct | -14.79 |
| Max Drawdown Inr | -5295.82 |
| Max Drawdown Pct | 10.59 |
| Sharpe | -2.76 |
| Profit Factor | 0.54 |
| Win Rate Pct | 26.98 |
| Total Trades | 63.00 |
| Avg R Multiple | -0.47 |
| Daily Loss Limit Days | 0.00 |

### Monthly Breakdown
| Month | Trades | Wins | Losses | Win Rate | Net P&L | Drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-12 | 1 | 0 | 1 | 0.00 | -200.52 | 0.00 |
| 2026-01 | 6 | 3 | 3 | 50.00 | 1265.71 | -490.10 |
| 2026-02 | 21 | 6 | 15 | 28.57 | -1380.60 | -957.67 |
| 2026-03 | 15 | 4 | 11 | 26.67 | -1318.10 | -945.39 |
| 2026-04 | 15 | 2 | 13 | 13.33 | -2333.45 | -1805.26 |
| 2026-05 | 5 | 2 | 3 | 40.00 | -54.11 | -159.99 |

### Top 3 Best Trades
| date | symbol | net_pnl | outcome |
| --- | --- | --- | --- |
| 2026-01-29 | AXISBANK | 884.99 | EOD_SQUARE_OFF |
| 2026-01-29 | AXISBANK | 590.93 | EOD_SQUARE_OFF |
| 2026-01-29 | AXISBANK | 574.75 | EOD_SQUARE_OFF |

### Top 3 Worst Trades
| date | symbol | net_pnl | outcome |
| --- | --- | --- | --- |
| 2026-02-19 | INFY | -354.52 | SL_HIT |
| 2026-04-10 | WIPRO | -351.60 | SL_HIT |
| 2026-02-23 | WIPRO | -343.95 | SL_HIT |

### Parameter Sensitivity
| Parameter Value | Current | Trades | Win Rate | Profit Factor | Sharpe | Unreliable |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | False | 111 | 18.02 | 0.36 | -4.82 | False |
| 1.02 | False | 86 | 19.77 | 0.35 | -4.26 | False |
| 1.05 | True | 63 | 26.98 | 0.54 | -2.76 | False |
| 1.08 | False | 44 | 34.09 | 0.81 | -1.40 | False |
| 1.10 | False | 38 | 28.95 | 0.42 | -4.29 | False |
| 1.12 | False | 37 | 32.43 | 0.45 | -3.99 | False |
| 1.15 | False | 36 | 33.33 | 0.46 | -3.97 | False |

### Quant Review Flags
- **Win rate > 45%:** FAIL
- **Profit factor > 1.2:** FAIL
- **Sharpe ratio > 0.8:** FAIL
- **Max drawdown < 20% capital:** PASS
- **Trades > 30:** PASS
- **No month > 40% profit:** FAIL
- **Monthly win rate std < 15pp:** WARN
- **Sensitivity has flat region:** FAIL
- **Overall verdict:** NOT READY

### Honest Assessment
The strategy is not ready. Either trade count, risk-adjusted return, drawdown, stability, or parameter robustness is insufficient.

### Recommended Next Steps
- Validate against broker-grade 5m/15m historical candles.
- Replace simulated AI scores with walk-forward trained models.
- Re-run with realistic liquidity, taxes, and symbol-specific slippage.
- Keep live capital disabled until out-of-sample performance is stable.

## gamma_scalper

| Metric | Value |
|---|---|
| Total Net Pnl | 0.00 |
| Total Net Pnl Pct | 0.00 |
| Annualised Return Pct | 0.00 |
| Max Drawdown Inr | 0.00 |
| Max Drawdown Pct | 0.00 |
| Sharpe | 0.00 |
| Profit Factor | 0.00 |
| Win Rate Pct | 0.00 |
| Total Trades | 0.00 |
| Avg R Multiple | 0.00 |
| Daily Loss Limit Days | 0.00 |

### Monthly Breakdown
No trades.

### Top 3 Best Trades
No trades.

### Top 3 Worst Trades
No trades.

### Parameter Sensitivity
| Parameter Value | Current | Trades | Win Rate | Profit Factor | Sharpe | Unreliable |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 9 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 10 | True | 0 | 0.00 | 0.00 | 0.00 | True |
| 11 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 12 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 13 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 14 | False | 0 | 0.00 | 0.00 | 0.00 | True |

### Quant Review Flags
- **Win rate > 45%:** FAIL
- **Profit factor > 1.2:** FAIL
- **Sharpe ratio > 0.8:** FAIL
- **Max drawdown < 20% capital:** PASS
- **Trades > 30:** FAIL
- **No month > 40% profit:** FAIL
- **Monthly win rate std < 15pp:** PASS
- **Sensitivity has flat region:** PASS
- **Overall verdict:** NOT READY

### Honest Assessment
The strategy is not ready. Either trade count, risk-adjusted return, drawdown, stability, or parameter robustness is insufficient.

### Recommended Next Steps
- Validate against broker-grade 5m/15m historical candles.
- Replace simulated AI scores with walk-forward trained models.
- Re-run with realistic liquidity, taxes, and symbol-specific slippage.
- Keep live capital disabled until out-of-sample performance is stable.

## mean_reversion

| Metric | Value |
|---|---|
| Total Net Pnl | 0.00 |
| Total Net Pnl Pct | 0.00 |
| Annualised Return Pct | 0.00 |
| Max Drawdown Inr | 0.00 |
| Max Drawdown Pct | 0.00 |
| Sharpe | 0.00 |
| Profit Factor | 0.00 |
| Win Rate Pct | 0.00 |
| Total Trades | 0.00 |
| Avg R Multiple | 0.00 |
| Daily Loss Limit Days | 0.00 |

### Monthly Breakdown
No trades.

### Top 3 Best Trades
No trades.

### Top 3 Worst Trades
No trades.

### Parameter Sensitivity
| Parameter Value | Current | Trades | Win Rate | Profit Factor | Sharpe | Unreliable |
| --- | --- | --- | --- | --- | --- | --- |
| 2.00 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 2.50 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 3.00 | True | 0 | 0.00 | 0.00 | 0.00 | True |
| 3.50 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 4.00 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 4.50 | False | 0 | 0.00 | 0.00 | 0.00 | True |
| 5.00 | False | 0 | 0.00 | 0.00 | 0.00 | True |

### Quant Review Flags
- **Win rate > 45%:** FAIL
- **Profit factor > 1.2:** FAIL
- **Sharpe ratio > 0.8:** FAIL
- **Max drawdown < 20% capital:** PASS
- **Trades > 30:** FAIL
- **No month > 40% profit:** FAIL
- **Monthly win rate std < 15pp:** PASS
- **Sensitivity has flat region:** PASS
- **Overall verdict:** NOT READY

### Honest Assessment
The strategy is not ready. Either trade count, risk-adjusted return, drawdown, stability, or parameter robustness is insufficient.

### Recommended Next Steps
- Validate against broker-grade 5m/15m historical candles.
- Replace simulated AI scores with walk-forward trained models.
- Re-run with realistic liquidity, taxes, and symbol-specific slippage.
- Keep live capital disabled until out-of-sample performance is stable.
