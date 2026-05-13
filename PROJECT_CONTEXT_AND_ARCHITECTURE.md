# 🇮🇳 UNIFIED NSE INTRADAY TRADING BOT — ARCHITECTURE & CONTEXT

This document provides a comprehensive overview of the **Unified NSE Trading Bot** project. It is designed to give an AI coding assistant full context on the system's architecture, data flow, and core logic.

---

## 📌 PROJECT OBJECTIVE
To build a fully autonomous, production-grade intraday trading system for the National Stock Exchange (NSE) of India. The system operates in two distinct domains:
1.  **Equity (Nifty, BankNifty, Stocks)**: Driven by an **AI/ML Ensemble** (XGBoost, LSTM, Random Forest, HMM).
2.  **Currency Derivatives (USDINR, EURINR, etc.)**: Driven by a **Multi-Agent Swarm Intelligence** pipeline.

---

## 🏗️ SYSTEM ARCHITECTURE (HIGH-LEVEL)

The system follows a modular, event-driven architecture using **Redis** as the central message bus.

### 1. Data Pipeline (`pipeline/`)
*   **`websocket_feed.py`**: Connects to broker WebSockets (e.g., Angel One SmartAPI). Resolves instrument tokens dynamically using a Master Contract utility. Pushes raw ticks to Redis.
*   **`redis_queue.py`**: Manages communication between ingestion and processing.
*   **`candle_builder.py`**: Consumes ticks from Redis and builds candles (1m, 3m, 5m, 15m). Triggers Signal Engines on candle close.
*   **`live_price_streamer.py`**: Maintains a sliding window of recent ticks in Redis for high-frequency checks.

### 2. Signal Engines (`strategies/` & `agents/`)
*   **Equity Signal Engine**:
    *   Adds technical indicators (EMA, VWAP, RSI, ADX, etc.).
    *   Runs **AI Ensemble Inference**: Combines XGBoost probabilities with LSTM sequence scores and HMM regime detection.
    *   Validates signals against strict rule-based filters (EMA alignment, VWAP position, Trend strength).
*   **Currency Swarm Pipeline**:
    *   A sequential chain of agents: `Watcher` -> `Scanner` -> `Validator` -> `Risk Manager` -> `Publisher`.
    *   Uses a rule-based "4-of-6" condition check for entry.

### 3. Execution & Risk (`risk/` & `execution/`)
*   **`risk_engine.py`**: Shared module to calculate position sizing, check daily drawdown limits, and enforce "circuit breakers."
*   **`order_manager.py`**: Manages the order lifecycle (Entry -> Protection -> Exit). State is persisted in Redis.
*   **`broker_api.py`**: Abstract interface for broker interactions (supports `MockBroker` for paper trading and `AngelOneBroker` for live).

### 4. Trade Lifecycle Tracking (`tracking/`)
*   **`trade_lifecycle_tracker.py`**: Monitors active trades tick-by-tick. Detects Stop-Loss (SL) or Target hits in real-time and triggers exits via the `OrderManager`.
*   **`signal_logger.py`**: Logs ALL generated signals (including rejections) to `data/signal_log.csv`.
*   **`trade_journal_writer.py`**: Logs finalized (closed) trades to `data/trade_journal.csv` for performance analysis.

### 5. Frontend & Analysis (`dashboard/` & `learning/`)
*   **`dashboard/app.py`**: A **Streamlit** dashboard for real-time monitoring. Includes tabs for:
    *   🌩️ Live Feed (Signals & Executions)
    *   📈 Performance (Equity Curve & Win Rate)
    *   🛡️ Risk Engine (Diagnostics & DD)
    *   🧠 Brain Insights (AI Recommendations)
*   **`adaptive_learning_engine.py`**: Nightly process that tunes threshold parameters based on the day's performance.

---

## 🔄 DATA FLOW
1.  **Market Open**: `main.py` starts the Swarm.
2.  **Tick Arrival**: `WebSocketFeed` captures a tick -> `Redis`.
3.  **Candle Close**: `CandleBuilder` closes a 5m candle -> triggers `EquitySignalEngine`.
4.  **Signal Generation**: Engine checks AI + Filters -> returns a `Signal`.
5.  **Execution**: `OrderManager` checks `RiskEngine` -> places `Entry Order` -> places `SL/Target Orders`.
6.  **Monitoring**: `TradeLifecycleTracker` watches live ticks -> SL/Target hit -> triggers `Exit Order`.
7.  **Logging**: `SignalLogger` and `OrderManager` update CSVs.
8.  **Visualization**: `Streamlit Dashboard` refreshes to show the new state.

---

## 🛠️ TECH STACK
*   **Language**: Python 3.10+
*   **Infrastructure**: Redis (Streams & Key-Value), PostgreSQL (Historical), Docker.
*   **Data Science**: Pandas, NumPy, Scikit-Learn, XGBoost, PyTorch.
*   **UI**: Streamlit, Plotly.
*   **Broker**: Angel One SmartAPI.

---

## ⚠️ KEY OPERATIONAL RULES & FIXES
*   **Market Hours**: Equity (09:15-15:30), Currency (09:15-17:00). The bot includes auto-shutdown logic.
*   **Windows File Locking**: Dashboard includes retry logic for reading CSVs to avoid `PermissionError` when the bot is writing.
*   **Dynamic Tokens**: The bot downloads the `Master Contract` from Angel One daily to resolve tokens for symbols like `USDINR` which change frequently.
*   **Field Alignment**: Internal field name consistency is critical (e.g., ensuring Signal Engine and Tracker both use `target` instead of `t1`).

---

## 📁 PROJECT STRUCTURE
```text
/
├── agents/            # Currency Swarm Agents
├── config/            # YAML configs & Adaptive JSON
├── data/              # CSV Logs, Parquet History, SQLite DB
├── dashboard/         # Streamlit Dashboard
├── execution/         # Order Management & Broker APIs
├── features/          # Indicator & Feature Calculation
├── learning/          # Retraining & Parameter Tuning
├── logs/              # Application Logs (Loguru)
├── pipeline/          # Ingestion & Candle Building
├── risk/              # Risk Management Engine
├── strategies/        # Signal Engines (Equity/Currency)
├── tracking/          # Lifecycle & Accuracy Analysis
├── utils/             # Broker & Master Contract Utilities
└── main.py            # Entry Point
```

---
*Created for AI Context on 2026-05-12*
