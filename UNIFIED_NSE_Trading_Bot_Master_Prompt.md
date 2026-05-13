# 🇮🇳 UNIFIED NSE INTRADAY TRADING BOT — MASTER BUILD PROMPT
### Covers: Nifty · BankNifty · NSE Stocks · F&O · Currency Derivatives (USDINR, EURINR, GBPINR, JPYINR)
### Architecture: Multi-Agent Swarm + AI/ML Ensemble + Live Learning Engine

---

## ⚠️ READ THIS FIRST

This is a **complete, production-grade build specification**. Do NOT start coding immediately.
Read this entire prompt, then begin **Module by Module** as instructed at the end.

Before any code:
1. Confirm broker API credentials are set in `.env`
2. Confirm Redis + PostgreSQL are running via Docker Compose (provided in Module 0)
3. Confirm historical data (2021–present) is available in Parquet format
4. Set `paper_mode: true` in `config/config.yaml` until validated

---

## 📌 SYSTEM OVERVIEW

You are a **senior quantitative developer and ML engineer** building a fully autonomous, self-learning NSE intraday trading system. The system has **two instrument domains** running in parallel on a shared infrastructure:

| Domain | Instruments | Signal Approach |
|---|---|---|
| **Equity** | Nifty, BankNifty, NSE Stocks | AI/ML Ensemble (XGBoost + LSTM + HMM + RF) |
| **Currency** | USDINR, EURINR, GBPINR, JPYINR | Rule-Based Multi-Agent Swarm |

Both domains share the same: data pipeline, risk engine, execution layer, journal, learning engine, and dashboard.

**The system NEVER forces a trade.** If no high-confidence setup exists, the output is `NO TRADE — WAIT`.

---

## 🏗️ FULL SYSTEM ARCHITECTURE

```
                    ┌─────────────────────────────────────────┐
                    │         NSE MARKET DATA LAYER           │
                    │  (WebSocket ticks → Redis → Candle DB)  │
                    └──────────────────┬──────────────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
               ▼                       ▼                       ▼
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │  FEATURE ENGINE  │   │  FEATURE ENGINE  │   │  LIVE STREAMER   │
    │  (Equity/F&O)    │   │  (Currency)      │   │  (Tick Buffer)   │
    └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
             │                      │                       │
             ▼                      ▼                       │
    ┌──────────────────┐   ┌──────────────────┐             │
    │  AI/ML ENSEMBLE  │   │  AGENT SWARM     │             │
    │  XGBoost+LSTM    │   │  5-Agent Chain   │             │
    │  +RF+HMM         │   │  (Currency)      │             │
    └────────┬─────────┘   └────────┬─────────┘             │
             │                      │                       │
             └──────────┬───────────┘                       │
                        ▼                                   │
             ┌──────────────────────┐                       │
             │  SIGNAL VALIDATION   │◄──────────────────────┘
             │  (Shared Filters)    │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │  RISK MANAGEMENT     │
             │  ENGINE              │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │  ORDER EXECUTION     │
             │  FastAPI + Broker    │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │  TRADE LIFECYCLE     │◄── tick-by-tick tracking
             │  TRACKER             │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │  JOURNAL + ACCURACY  │
             │  + LEARNING ENGINE   │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │  MONITORING DASHBOARD│
             │  + DAILY REPORTS     │
             └──────────────────────┘
```

---

## 📁 FOLDER STRUCTURE

Generate ALL folders, files, `__init__.py`, and stub `README.md` at project start:

```
nse-trading-bot/
│
├── pipeline/
│   ├── websocket_feed.py          # WebSocket connection + tick ingestion
│   ├── redis_queue.py             # Redis stream producer/consumer
│   ├── candle_builder.py          # Tick → 1m/3m/5m/15m candles
│   └── live_price_streamer.py     # Persistent tick buffer (500 ticks/symbol)
│
├── features/
│   ├── price_features.py          # EMA, VWAP, RSI, MACD, ATR, Supertrend, ADX
│   ├── volume_features.py         # Rel. vol, delta vol, spike ratio
│   ├── options_features.py        # PCR, OI buildup, IV rank, max pain
│   ├── currency_features.py       # Pivot points, BB, currency-specific filters
│   └── time_features.py           # Session flags, expiry pressure
│
├── agents/                        # Currency swarm agents (rule-based)
│   ├── market_watcher.py
│   ├── opportunity_scanner.py
│   ├── quant_validator.py
│   ├── risk_manager_agent.py
│   └── signal_publisher.py
│
├── models/                        # Equity/F&O AI models
│   ├── xgboost/
│   ├── lstm/
│   ├── random_forest/
│   └── hmm/
│
├── strategies/
│   ├── equity_signal_engine.py    # ML signal → validated trade signal
│   └── currency_signal_engine.py  # Agent swarm output → validated signal
│
├── risk/
│   └── risk_engine.py             # Shared risk engine for both domains
│
├── execution/
│   ├── order_manager.py
│   └── broker_api.py              # Abstract broker interface
│
├── tracking/
│   ├── trade_lifecycle_tracker.py # Tick-by-tick active trade monitor
│   ├── trade_journal_writer.py    # CSV + Google Sheets logging
│   └── accuracy_analyzer.py      # Post-trade breakdown analysis
│
├── learning/
│   ├── adaptive_learning_engine.py  # Nightly threshold tuning
│   └── retrain_pipeline.py          # Weekly ML model retraining
│
├── backtesting/
│   └── backtester.py              # VectorBT + walk-forward validation
│
├── dashboard/
│   └── app.py                     # Streamlit monitoring dashboard
│
├── reporting/
│   └── daily_report_publisher.py  # Telegram/email daily + weekly reports
│
├── data/
│   ├── historical/                # Parquet files
│   ├── ticks.db                   # SQLite real-time tick store
│   ├── trade_journal.csv
│   ├── signal_log.csv
│   ├── daily_summary.csv
│   ├── session_conditions.csv
│   └── parameter_history.csv
│
├── config/
│   ├── config.yaml                # All tunable parameters
│   └── adaptive_params.json       # Live thresholds (updated nightly)
│
├── logs/
├── tests/
├── notebooks/
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── main.py                        # Entry point
```

---

## MODULE 0 — INFRASTRUCTURE SETUP

### `docker-compose.yml`
Provide a ready-to-run Docker Compose file with:
- **Redis** (port 6379) with persistence enabled
- **PostgreSQL** (port 5432) with `trading_bot` database
- **Grafana** (port 3000) connected to PostgreSQL
- **SQLite** for intraday tick storage (no container needed, file-based)

### `config/config.yaml`
```yaml
broker:
  primary: "zerodha"               # or "dhan", "angelone", "fyers", "upstox"
  api_key: ""                      # from .env
  access_token: ""                 # from .env
  fallback: "dhan"

paper_mode: true                   # ALWAYS start true

capital:
  equity_total: 50000
  currency_total: 25000
  risk_per_trade_pct: 1.0          # 1% of capital per trade
  max_open_trades_equity: 2
  max_open_trades_currency: 2

instruments:
  equity: ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK"]
  currency: ["USDINR", "EURINR", "GBPINR", "JPYINR"]

equity_signal:
  min_buy_confidence: 0.72
  min_sell_confidence: 0.70
  min_relative_volume: 1.5
  max_vix: 20.0

currency_signal:
  conditions_required: 4           # out of 6
  min_quant_score: 70
  min_adx: 20
  max_sl_paise: 20
  min_rr: 1.5
  min_volume_ratio: 1.2

risk:
  daily_loss_limit_r: 3
  consecutive_loss_limit: 3
  intraday_drawdown_limit_pct: 5.0
  atr_sl_multiplier: 1.5
  rr_ratio: 1.5
  max_vix_spike: 25.0
  currency_max_daily_loss_inr: 1500
  currency_max_daily_trades: 5

model:
  xgb_weight_trending: 0.6
  lstm_weight_trending: 0.4
  retrain_schedule: "weekly"

adaptive_learning:
  min_trades_before_change: 15
  apply_changes: "next_session_only"  # never mid-session

latency:
  target_ms: 300

session:
  market_open: "09:15"
  noise_window_end: "09:30"
  chop_zone_start: "11:30"
  chop_zone_end: "13:30"
  trend_window_start: "14:30"
  currency_cutoff: "15:00"
  equity_cutoff: "15:15"

notifications:
  channel: "telegram"              # or "email", "whatsapp"
  telegram_token: ""               # from .env
  telegram_chat_id: ""
```

---

## MODULE 1 — REAL-TIME DATA PIPELINE

Build `pipeline/websocket_feed.py`, `pipeline/redis_queue.py`, `pipeline/candle_builder.py`, `pipeline/live_price_streamer.py`.

### WebSocket Feed:
- Connect to broker WebSocket at **09:14:50 IST** (10 sec before open)
- Subscribe to: tick data (LTP, bid, ask, volume, OI) + market depth (Level 2)
- Push every tick to Redis stream: `market:ticks:{symbol}`
- Auto-reconnect on drop (max 3 retries, then Telegram alert)
- Graceful shutdown at **15:05 IST** after all positions confirmed closed

### Candle Builder (from tick stream):
- Build **1m, 3m, 5m, 15m, 1h** candles in real-time from tick buffer
- On each closed candle: recalculate ALL indicators and push to feature layer
- Keep last **500 ticks per symbol** in RAM (`collections.deque`)
- Persist ticks to SQLite every 60 seconds

### Additional Data Fetched Every 5 Minutes:
- Options chain: PCR, OI buildup, IV rank, Max Pain
- Market breadth: Advance/Decline ratio, sector strength
- India VIX (every 1 minute)
- FII/DII flow (once at open)

### Latency target: **< 300ms** from tick arrival to signal output

---

## MODULE 2 — FEATURE ENGINEERING LAYER

Build `features/` with vectorised Polars/Pandas code. All features computed on rolling window. Output = flat NumPy array ready for model inference with zero transformation.

### Price Action Features (all timeframes):
| Feature | Detail |
|---|---|
| EMA 9 / 21 / 50 | Trend direction and alignment |
| VWAP (intraday reset at 09:15) | Institutional bias |
| RSI (14) | Momentum (overbought/oversold) |
| MACD Histogram | Trend acceleration |
| ATR (14) | Volatility normalisation |
| Supertrend (ATR 10, factor 3.0) | Trend direction binary |
| ADX (14) | Trend strength (> 20 = trending) |
| Bollinger Bands (20, 2.0) | Volatility bands |
| Daily Pivot Points (R1, R2, S1, S2, PP) | Key S/R levels |
| Candle body/wick ratio | Candle strength signal |
| Gap % from prev close | Open behaviour |

### Volume Features:
| Feature | Detail |
|---|---|
| Relative Volume vs 20-day avg | Breakout confirmation |
| Volume Spike Ratio | Momentum bursts |
| Delta Volume (buy pressure - sell pressure) | Order flow direction |
| Volume ratio vs 20-bar avg | Currency-specific filter |

### Options Features (equity only):
| Feature | Detail |
|---|---|
| PCR (OI-based) | Market sentiment |
| OI change % CE vs PE | Smart money positioning |
| IV rank | Volatility regime |
| Max Pain level | Options pinning zone |

### Time Features:
| Flag | Condition |
|---|---|
| `noise_window` | 09:15–09:30 IST → skip signals |
| `chop_zone` | 11:30–13:30 IST → higher threshold |
| `trend_window` | 14:30–15:15 IST → trend continuation bias |
| `currency_exit_warn` | 14:45 IST → no new currency signals |
| `minutes_to_expiry` | F&O decay pressure score |

---

## MODULE 3 — SIGNAL ENGINE A: EQUITY AI ENSEMBLE

Build `models/` and `strategies/equity_signal_engine.py`.

### ML Pipeline Architecture:
```
Feature Vector (all price + volume + options + time features)
         ↓
  [XGBoost Classifier]        → P(BUY / SELL / NEUTRAL) score
         ↓
  [LSTM Sequence Model]       → Temporal confirmation score (last 30 candles)
         ↓
  [Random Forest Vol Filter]  → HIGH_VOL / LOW_VOL regime gate
         ↓
  [HMM Regime Detector]       → TRENDING / CHOPPY / REVERSAL state
         ↓
  Adaptive Weighted Ensemble  → Final confidence %
```

### Model Specifications:

**XGBoost (`models/xgboost/`):**
- Multiclass: `{BUY=+1, NEUTRAL=0, SELL=-1}`
- Label: `+1` if `close[t+15] > close[t] + spread + slippage`, else `0` or `-1`
- Use `scale_pos_weight` for class imbalance correction
- Output: probability vector `[p_buy, p_neutral, p_sell]`

**LSTM (`models/lstm/`, PyTorch):**
- Input: sliding window of last **30 candles × feature vector**
- Architecture: `2-layer LSTM → Dropout(0.3) → Dense(64) → Softmax(3)`
- Train ONLY on HMM-confirmed TRENDING regime samples
- Output: sequence-level direction probability

**Random Forest (`models/random_forest/`):**
- Binary: `HIGH_VOL` vs `LOW_VOL`
- Gate: suppress signals entirely during `HIGH_VOL` unless VIX is stable

**HMM Regime Detector (`models/hmm/`, hmmlearn):**
- 3 hidden states: `TRENDING`, `CHOPPY`, `REVERSAL`
- Input features: returns, ATR, volume delta
- Used to set ensemble weights dynamically

**Ensemble Weighting Logic:**
```python
if hmm_state == "TRENDING":
    final_score = 0.6 * xgb_score + 0.4 * lstm_score
elif hmm_state == "CHOPPY":
    final_score = 0.9 * xgb_score + 0.1 * lstm_score
elif hmm_state == "REVERSAL":
    final_score = 0.5 * xgb_score + 0.5 * lstm_score
```

### Equity Signal Validation (ALL must pass):

**BUY Signal:**
- [ ] AI ensemble confidence ≥ **72%**
- [ ] Price above VWAP
- [ ] EMA9 > EMA21 > EMA50 (full bullish alignment)
- [ ] OI long buildup (CE OI falling OR PE OI rising)
- [ ] Relative Volume ≥ **1.5×**
- [ ] India VIX < **20**
- [ ] HMM state ≠ CHOPPY
- [ ] Not in noise window (09:15–09:30) or chop zone (11:30–13:30)
- [ ] ADX > 20

**SELL Signal:**
- [ ] AI ensemble confidence ≥ **70%**
- [ ] Price below VWAP
- [ ] EMA9 < EMA21 < EMA50 (full bearish alignment)
- [ ] OI short buildup confirmed
- [ ] Sector index weak + Advance/Decline ratio < 0.8
- [ ] ADX > 20

---

## MODULE 4 — SIGNAL ENGINE B: CURRENCY MULTI-AGENT SWARM

Build `agents/`. This is a **5-agent sequential pipeline** (AutoHedge pattern) for currency derivatives only.

```
[MarketWatcher] → [OpportunityScanner] → [QuantValidator] → [RiskManagerAgent] → [SignalPublisher]
```

### Agent 1 — MarketWatcher:
- Fetch USDINR, EURINR, GBPINR, JPYINR nearest expiry futures
- Trading window: **09:30 IST to 14:45 IST** (15 min noise buffer each side)
- Compute every 5-min candle close: EMA(9/21/50), RSI(14), VWAP, Supertrend(10, 3.0), ADX(14), BB(20, 2.0), Daily Pivots (R1/R2/S1/S2/PP), Volume ratio

### Agent 2 — OpportunityScanner:
A setup is valid only when **≥ 4 of 6 conditions** are simultaneously true.

**BUY conditions:**
1. `EMA9 > EMA21 > EMA50`
2. `RSI between 45 and 70`
3. `Supertrend = bullish`
4. `LTP > VWAP`
5. `ADX > 20`
6. `Volume ratio > 1.2`

**SELL conditions:**
1. `EMA9 < EMA21 < EMA50`
2. `RSI between 30 and 55`
3. `Supertrend = bearish`
4. `LTP < VWAP`
5. `ADX > 20`
6. `Volume ratio > 1.2`

**HARD REJECT (any one → NO TRADE):**
- RSI > 75 for BUY or RSI < 25 for SELL
- ADX < 18
- Within noise window (09:15–09:30)
- Within 30 min of RBI / CPI / NFP announcement
- Price within 3 paise of a major Pivot level

### Agent 3 — QuantValidator:
- Confirm 15-min chart trend aligns with 5-min setup direction (if opposed → REJECT)
- Check 1-hour chart: no major resistance above (BUY) or support below (SELL)
- ATR(14) on 5-min: reject if `ATR < 0.03` (too quiet) or `ATR > 0.25` (too volatile)
- Pattern bonus: Bullish/Bearish Engulfing, Inside Bar breakout, Hammer/Shooting Star
- `quant_score >= 70` → proceed | `< 70` → NO TRADE

### Agent 4 — RiskManagerAgent:
- BUY SL: `min(recent_swing_low_10c, supertrend_value) + 0.5×ATR buffer`
- SELL SL: `max(recent_swing_high_10c, supertrend_value) − 0.5×ATR buffer`
- Max SL hard cap: **20 paise**
- T1: next pivot OR 1.5× SL distance (whichever closer)
- T2: next pivot OR 2.5× SL distance
- Minimum R:R = **1.5** (reject if below)
- Lots: `floor(₹500 / (sl_paise × 1000))`, min 1, max 3

### Agent 5 — SignalPublisher:
Publishes a formatted signal card only if ALL upstream agents approved. Max **2 active signals at any time** across all currency pairs.

**Signal card format:**
```
╔══════════════════════════════════════════════╗
║  🟢 BUY SIGNAL — USDINR                      ║
║  📅 {date} | {time} IST                       ║
╠══════════════════════════════════════════════╣
║  ENTRY     : {entry}                          ║
║  STOP LOSS : {sl}  (−{sl_paise} paise)        ║
║  TARGET 1  : {t1}  (+{t1_paise} p) — 60% qty ║
║  TARGET 2  : {t2}  (+{t2_paise} p) — 40% qty ║
║  R:R RATIO : 1:{rr}                           ║
║  LOTS      : {lots} (₹{risk} at risk max)     ║
╠══════════════════════════════════════════════╣
║  SETUP     : {conditions_met} / 6 conditions ║
║  PATTERN   : {pattern}                        ║
║  STRENGTH  : {quality} (Score: {score}/100)   ║
║  ADX       : {adx}                            ║
║  MTF       : 15-min Aligned ✅               ║
╠══════════════════════════════════════════════╣
║  ⚠️  EXIT ALL positions by 15:00 IST          ║
╚══════════════════════════════════════════════╝
```

---

## MODULE 5 — RISK MANAGEMENT ENGINE (SHARED)

Build `risk/risk_engine.py`. **Every signal from both domains passes through this.** No exceptions.

### Position Sizing:

**Equity (Fixed Fractional):**
```python
risk_per_trade = capital * 0.01           # 1% of capital
sl_points = abs(entry_price - sl_price)
quantity = floor(risk_per_trade / sl_points)
```

**Currency (Lot-based):**
```python
risk_per_trade_inr = 500                  # conservative fixed amount
lots = floor(risk_per_trade_inr / (sl_paise * 1000))
lots = max(1, min(lots, 3))               # clamp between 1–3
```

### Hard Circuit Breakers:
| Rule | Action |
|---|---|
| Daily equity loss ≥ 3R | Halt equity trading for the day |
| Daily currency loss ≥ ₹1,500 | Halt currency trading for the day |
| 3 consecutive losses (either domain) | 30-min pause + reassess regime |
| Max open trades > 2 per domain | Reject new signals |
| Intraday drawdown ≥ 5% | Emergency flatten all positions |
| VIX spikes above 25 suddenly | Immediate position reduction by 50% |

### Dynamic Stop Loss:
- Equity: `SL = entry ± (ATR(14) × 1.5)`
- Currency: `SL = swing_extreme ± (ATR(14) × 0.5)`, hard cap 20 paise
- **Trail SL to breakeven once trade moves 1R in profit**
- Partial exit: 50–60% at T1, trail remainder to T2

---

## MODULE 6 — ORDER EXECUTION ENGINE

Build `execution/order_manager.py` and `execution/broker_api.py`.

### Abstract Broker Interface:
```python
class BaseBroker(ABC):
    def place_order(self, symbol: str, qty: int, direction: str,
                    order_type: str, price: float = None) -> dict: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> list: ...
    def get_order_status(self, order_id: str) -> dict: ...
    def modify_order(self, order_id: str, new_price: float) -> dict: ...
```

Implement concrete classes: `ZerodhaBroker`, `DhanBroker`, `AngelOneBroker`

### Order Flow:
```
Signal → Risk Check → Place Entry Order → Await Fill Confirmation
→ Place SL-M Order → Place Limit Target Order
→ On T1 Hit: cancel SL, place new breakeven SL, partial exit
→ On T2 Hit: cancel SL order, full exit
→ On SL Hit: cancel target order, log outcome, start cooldown timer
→ At session cutoff: force-close all via MARKET order
```

### Implementation:
- Use **FastAPI** (async) as the execution microservice
- All order events logged to PostgreSQL: timestamp, type, price, status, slippage
- Slippage tracked = `fill_price - signal_price` (stored per trade)
- Handle: order rejection, partial fills, network timeout, stale prices

---

## MODULE 7 — TRADE LIFECYCLE TRACKER

Build `tracking/trade_lifecycle_tracker.py`. Runs on every incoming tick for active trades.

### What It Tracks Per Active Trade:
```python
{
    "trade_id": "USDINR-20250508-003",
    "status": "ACTIVE",                    # ACTIVE / T1_HIT / T2_HIT / SL_HIT / TIME_EXIT
    "entry": 84.21,
    "current_ltp": 84.33,
    "unrealized_pnl_paise": 12,
    "unrealized_pnl_inr": 240,
    "mfe_paise": 15,                       # Max Favorable Excursion — best price reached
    "mae_paise": 4,                        # Max Adverse Excursion — worst dip vs entry
    "time_in_trade_min": 18,
    "sl_current": 84.21,                   # dynamic, moves to breakeven after T1
    "t1_hit": True,
    "t2_hit": False
}
```

### Entry Monitoring Window:
- If price moves > **8 paise** away from entry within 5 min of signal → mark `ENTRY_MISSED`, do NOT chase

### Exit Rules:
- On T1 hit → book 60% qty, trail SL to entry (breakeven), continue for T2
- On T2 hit → book remaining 40%, close trade
- On SL hit → close fully, start 30-min cooldown for that symbol
- At 15:00 IST (currency) / 15:15 IST (equity) → force-exit all at MARKET

---

## MODULE 8 — TRADE JOURNAL WRITER

Build `tracking/trade_journal_writer.py`. Write to `trade_journal.csv` after every trade close.

### Journal Schema (one row per signal):
`signal_id, date, day_of_week, domain, symbol, direction, signal_time, entry_time, entry_price, sl, sl_paise, target_1, target_2, rr_ratio, quant_score, setup_type, pattern, adx_at_entry, rsi_at_entry, vwap_position, mtf_aligned, volume_ratio, market_session, market_character, outcome, exit_price, exit_time, pnl_paise, pnl_inr, lots, duration_min, mfe_paise, mae_paise, sl_touched, notes`

### Additional Log Files:
- `signal_log.csv` — ALL signals including NO TRADE ones (with reason)
- `daily_summary.csv` — one row per day: total trades, win rate, net P&L
- `session_conditions.csv` — 15-min market character snapshots (trending/choppy/volatile)
- `parameter_history.csv` — every adaptive threshold change with timestamp + reason

---

## MODULE 9 — ACCURACY ANALYZER

Build `tracking/accuracy_analyzer.py`. Runs after EVERY closed trade and at end of day.

### Per-Trade Analysis:
```python
accuracy_snapshot = {
    "rolling_win_rate_10": wins_last_10 / 10,
    "rolling_win_rate_20": wins_last_20 / 20,
    "avg_rr_achieved": mean(actual_rr_last_20),     # vs planned
    "avg_mfe_paise": mean(mfe_last_20),
    "avg_mae_paise": mean(mae_last_20),
    "sl_too_tight_rate": sl_hit_within_5p / total,
    "t1_hit_rate": t1_hits / total_non_sl,
    "t2_hit_rate": t2_hits / total_non_sl,
}
```

### End-of-Day Breakdown Analysis (run at 15:30 IST):
- By **symbol**: win rate, avg P&L, # trades
- By **setup type**: e.g., "EMA+Supertrend+VWAP" win rate
- By **market session**: morning / mid / afternoon
- By **market character**: trending vs choppy vs volatile
- By **ADX range**: <20, 20–30, 30–40, >40
- By **confidence bracket**: 70–75%, 75–80%, 80%+
- By **day of week**: Monday–Friday performance patterns

---

## MODULE 10 — ADAPTIVE LEARNING ENGINE

Build `learning/adaptive_learning_engine.py`. Runs nightly at **15:30 IST**, applies changes to **next session only**.

### Threshold Tuning Rules:

```python
# Only act when: min 15 trades in the bracket AND change is within safety bounds

if rolling_win_rate_20 < 0.50:
    min_quant_score += 2                  # tighten quality threshold
    min_adx += 2                          # avoid ranging markets more

if rolling_win_rate_20 > 0.75 for 5+ days:
    min_quant_score = max(min_quant_score - 2, 65)  # slightly relax

if sl_hit_rate > 0.60:
    max_sl_paise = min(max_sl_paise + 2, 25)        # SL too tight, widen

if t1_hit_rate < 0.45:
    t1_multiplier = max(t1_multiplier - 0.1, 1.2)  # bring target closer

if avg_mae < 0.5 * avg_sl_paise:
    max_sl_paise = max(max_sl_paise - 1, 12)        # SL too wide, tighten

if symbol_win_rate[symbol] < 0.40 over_last_20:
    symbol_active[symbol] = False                   # pause that pair
    alert(f"Pausing {symbol} — win rate {symbol_win_rate[symbol]:.0%}")

if session_win_rate["10:30-12:00"] > 0.70 for 5+ days:
    session_weight["10:30-12:00"] = 1.15            # boost that window
```

### Safety Bounds (hard limits — never exceeded):
| Parameter | Min | Max |
|---|---|---|
| `min_quant_score` | 65 | 85 |
| `min_adx` | 18 | 30 |
| `max_sl_paise` | 12 | 25 |
| `min_rr` | 1.3 | 2.0 |
| `min_volume_ratio` | 1.0 | 2.0 |

**Rules:**
1. All changes logged to `parameter_history.csv` with reason
2. Announced via Telegram before next session opens
3. Never applied mid-session
4. Never adjusts capital or position sizing (only indicator thresholds)
5. Requires minimum **15 trades** per bracket before changing anything

---

## MODULE 11 — ML RETRAINING PIPELINE

Build `learning/retrain_pipeline.py`. Do NOT retrain daily.

### Schedule:
- **Weekly (Saturday):** Retrain XGBoost + Random Forest on rolling 6-month window
- **Monthly:** Re-optimise LSTM hyperparameters + HMM transition matrix

### Walk-Forward Retraining Folds:
```
Fold 1:  Train 2021–2022    → Test 2023-Q1
Fold 2:  Train 2021–2023    → Test 2023-Q2
Fold 3:  Train 2021–2023Q3  → Test 2023-Q4
Fold 4:  Train 2021–2024    → Test 2025 (shadow live)
```

### Deployment Gate:
```
New week data appended to dataset
         ↓
Walk-forward retrain
         ↓
Shadow paper trading (3 days)
         ↓
If Sharpe ratio improves → deploy new model
Else → keep old model, log degradation alert to Telegram
```

---

## MODULE 12 — BACKTESTING ENGINE

Build `backtesting/backtester.py` with **VectorBT** (primary) and Backtrader (secondary).

### Realism Requirements:
- Slippage: `0.05%` per side (equity), `1 paise` per side (currency)
- Brokerage: ₹20/order flat
- Spread cost simulation
- Delayed fill: signal at candle close, fill at NEXT candle open
- Partial fill simulation for larger lots

### Output Metrics (minimum acceptable):
| Metric | Equity Threshold | Currency Threshold |
|---|---|---|
| Accuracy | 55–65% | 55–68% |
| Sharpe Ratio | > 1.5 | > 1.5 |
| Max Drawdown | < 10% | < 8% |
| Win Rate | > 45% | > 50% |
| Avg R:R | > 1.5 | > 1.5 |
| Profit Factor | > 1.4 | > 1.4 |

> **REJECT** any backtest showing > 70% accuracy. It is almost certainly overfit. Investigate and fix before proceeding.

---

## MODULE 13 — MONITORING DASHBOARD

Build `dashboard/app.py` using **Streamlit**. Connect to PostgreSQL (logs) and Redis (live feed).

### Dashboard Panels:
1. **Live PnL** — real-time unrealised + realised (both domains)
2. **Signal Feed** — last 10 signals with confidence scores and outcomes
3. **Active Positions** — symbol, entry, current LTP, SL, target, unrealised PnL, MFE/MAE
4. **Circuit Breaker Status** — daily loss counter, cooldown timers, active halts
5. **Model Accuracy** — rolling 7-day live accuracy vs backtest baseline (equity)
6. **Agent Scores** — today's quant scores distribution (currency)
7. **Win Rate & Drawdown** — daily/weekly chart
8. **Latency Monitor** — avg tick-to-signal ms (target < 300ms)
9. **Regime Indicator** — current HMM state per equity instrument
10. **Learning Engine Status** — last parameter changes, pending changes for tomorrow
11. **Symbol Health** — active/paused status for each instrument

---

## MODULE 14 — DAILY REPORT PUBLISHER

Build `reporting/daily_report_publisher.py`. Send via Telegram at **15:30 IST daily**, weekly summary every **Friday at 16:00 IST**.

### Daily Report Format:
```
📊 DAILY TRADING REPORT — {date} ({day})

━━━━━━━ EQUITY ━━━━━━━
Signals: {n} | Entered: {n} | No Trade: {n}
✅ Winners: {n}  ❌ Losers: {n}  📈 Win Rate: {pct}%
Gross P&L: ₹{x}  |  Brokerage: ₹{y}  |  Net: ₹{z}

━━━━━━ CURRENCY ━━━━━━
Signals: {n} | Entered: {n} | No Trade: {n}
✅ Winners: {n}  ❌ Losers: {n}  📈 Win Rate: {pct}%
Gross P&L: ₹{x}  |  Brokerage: ₹{y}  |  Net: ₹{z}

━━━━━ COMBINED ━━━━━━
Net P&L Today:    ₹{total}
This Week:        ₹{week}
This Month:       ₹{month}

Signal Quality Score: {score}/100
Accuracy Trend: {📈 Improving / 📉 Declining / ➡ Stable}
ADX Environment: {avg} avg ({trending/choppy})

PARAMETER CHANGES TONIGHT:
{list of adaptive changes or "No changes"}

⚠️  Tomorrow: {any special notes — Friday caution, event days, etc.}
```

---

## MODULE 15 — ENTRY POINT & STARTUP SEQUENCE

Build `main.py`:

```python
import asyncio
from pipeline.websocket_feed import WebSocketFeed
from pipeline.live_price_streamer import LivePriceStreamer
from agents.pipeline import CurrencyAgentPipeline
from strategies.equity_signal_engine import EquitySignalEngine
from tracking.trade_lifecycle_tracker import TradeLifecycleTracker
from tracking.accuracy_analyzer import AccuracyAnalyzer
from learning.adaptive_learning_engine import AdaptiveLearningEngine
from reporting.daily_report_publisher import DailyReportPublisher
from dashboard.app import run_dashboard

async def main():
    params = load_adaptive_params("config/adaptive_params.json")

    await asyncio.gather(
        WebSocketFeed(params).run(),            # 09:14:50 — tick ingestion
        LivePriceStreamer(params).run(),         # continuous tick buffer
        EquitySignalEngine(params).run(),        # scan equity every 5 min
        CurrencyAgentPipeline(params).run(),     # scan currency every 5 min
        TradeLifecycleTracker(params).run(),     # tick-by-tick tracking
        AccuracyAnalyzer(params).run(),          # per-trade + end-of-day
        AdaptiveLearningEngine(params).run(),    # nightly at 15:30
        DailyReportPublisher(params).run(),      # 15:30 + Friday 16:00
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Daily Startup Sequence:
```
09:14:50 — WebSocket connects, LivePriceStreamer starts tick buffer
09:15:00 — Market opens
09:30:00 — First signals eligible (past noise window)
10:00:00 — Adaptive params from last night loaded into agents
...all day...
14:45:00 — No new currency signals
15:00:00 — Currency: force-exit all open positions
15:15:00 — Equity: force-exit all open positions
15:30:00 — AccuracyAnalyzer runs full day breakdown
15:30:05 — AdaptiveLearningEngine computes parameter updates
15:30:10 — DailyReportPublisher sends Telegram summary
15:31:00 — WebSocket gracefully closes
```

---

## 📋 CODING STANDARDS

1. **Type hints** on every function and class
2. **Unit tests** in `tests/` for every module
3. **Loguru** for all structured logging
4. All errors caught, logged to file + PostgreSQL — never silently swallowed
5. No hardcoded credentials — use `.env` + `python-dotenv`
6. All async I/O via `asyncio` + `aiohttp`
7. Redis uses connection pooling
8. Database writes are non-blocking (async SQLAlchemy)
9. Every parameter change logged with timestamp and reason

---

## ❌ WHAT NOT TO BUILD

- No martingale / averaging-down logic
- No "guaranteed profit" claims in code or comments
- No single-model trading (always validate with rule filters)
- No random train/test splits in backtesting (walk-forward only)
- No daily ML retraining
- No fixed-point stop losses (use ATR or swing extremes)
- No trading during 09:15–09:30 noise window
- No mid-session parameter changes by the learning engine
- No autonomous capital allocation changes (only indicator thresholds)

---

## 🏁 PERFORMANCE TARGETS (Realistic Live Expectations)

| Metric | Target | Reject If |
|---|---|---|
| Equity accuracy | 55–65% | > 70% (overfit) |
| Currency win rate | 50–65% | > 75% without 50+ trade sample |
| Sharpe ratio | > 1.5 | < 1.0 |
| Max drawdown | < 10% | > 15% |
| Avg R:R achieved | > 1.3 | < 1.0 |
| Tick-to-signal latency | < 300ms | > 500ms |

---

## 🚀 DEVELOPMENT ROADMAP

| Month | Milestone |
|---|---|
| 1 | Module 0–2: Infrastructure, data pipeline, all indicators working |
| 2 | Module 3–4: Equity ML stack + Currency agent swarm, both in paper mode |
| 3 | Module 5–8: Risk engine, execution, tracker, journal — full paper trading |
| 4 | Module 9–11: Accuracy analyzer, learning engine, backtesting validated |
| 5 | Module 12–15: Dashboard, reports, live with minimal capital |
| 6+ | Continuous optimisation, regime adaptation, portfolio expansion |

---

## 🏃 START HERE

**Begin with Module 0.** Generate the `docker-compose.yml`, `.env.example`, `requirements.txt`, and the full folder structure with stubs.

Then proceed Module by Module. After completing each module, run its unit tests before moving on.

**Never skip the paper trading phase.** The learning engine requires a minimum of **15 trades per bracket** before it can tune parameters. Run paper mode for at least **3–4 weeks** before enabling live trading.

Output **production-ready, fully commented Python code** for each module when asked.
