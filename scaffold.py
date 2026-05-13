import os

directories = [
    "pipeline", "features", "agents", "models", "models/xgboost", "models/lstm", "models/random_forest", "models/hmm",
    "strategies", "risk", "execution", "tracking", "learning", "backtesting", "dashboard", "reporting", "data", "data/historical", "config", "logs", "tests", "notebooks"
]

files = [
    "pipeline/__init__.py", "pipeline/websocket_feed.py", "pipeline/redis_queue.py", "pipeline/candle_builder.py", "pipeline/live_price_streamer.py",
    "features/__init__.py", "features/price_features.py", "features/volume_features.py", "features/options_features.py", "features/currency_features.py", "features/time_features.py",
    "agents/__init__.py", "agents/market_watcher.py", "agents/opportunity_scanner.py", "agents/quant_validator.py", "agents/risk_manager_agent.py", "agents/signal_publisher.py",
    "models/__init__.py", "models/xgboost/__init__.py", "models/lstm/__init__.py", "models/random_forest/__init__.py", "models/hmm/__init__.py",
    "strategies/__init__.py", "strategies/equity_signal_engine.py", "strategies/currency_signal_engine.py",
    "risk/__init__.py", "risk/risk_engine.py",
    "execution/__init__.py", "execution/order_manager.py", "execution/broker_api.py",
    "tracking/__init__.py", "tracking/trade_lifecycle_tracker.py", "tracking/trade_journal_writer.py", "tracking/accuracy_analyzer.py",
    "learning/__init__.py", "learning/adaptive_learning_engine.py", "learning/retrain_pipeline.py",
    "backtesting/__init__.py", "backtesting/backtester.py",
    "dashboard/__init__.py", "dashboard/app.py",
    "reporting/__init__.py", "reporting/daily_report_publisher.py",
    "config/__init__.py", "config/adaptive_params.json",
    "main.py", "README.md", ".env.example", "requirements.txt", "docker-compose.yml"
]

for d in directories:
    os.makedirs(d, exist_ok=True)

for f in files:
    if not os.path.exists(f):
        with open(f, 'w') as file:
            file.write("")

print("Scaffolding complete.")
