# UNIFIED NSE INTRADAY TRADING BOT

This is a complete, production-grade NSE intraday trading system. The system runs two instrument domains in parallel:

1. **Equity** (Nifty, BankNifty, Stocks) - AI/ML Ensemble (XGBoost + LSTM + RF + HMM)
2. **Currency** (USDINR, EURINR, GBPINR, JPYINR) - Multi-Agent Swarm

## Requirements
- Python 3.10+
- Docker & Docker Compose (for Redis, PostgreSQL, Grafana)

## Quick Start
1. `cp .env.example .env` and fill in API keys
2. `docker-compose up -d`
3. `pip install -r requirements.txt`
4. Train the model: `$env:PYTHONPATH="."; python learning/retrain_pipeline.py`
5. `python main.py`

*ALWAYS run in `paper_mode: true` initially.*
