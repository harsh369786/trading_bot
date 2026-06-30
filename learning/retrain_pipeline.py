import glob
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.price_features import PriceFeatures
from models.random_forest.model import RandomForestModel
from models.xgboost.model import XGBoostModel

FEATURE_COLS = [
    "dist_ema_9", "dist_ema_21", "dist_ema_50", "rsi_14", "atr_pct",
    "dist_vwap", "ADX_14", "DMP_14", "DMN_14", "bb_pct",
]



class RetrainPipeline:
    """Walk-forward ensemble retraining with guarded, atomic deployment."""

    def __init__(
        self,
        data_path: str = "data/historical/*_6m.parquet",
        status_path: str = "data/retrain_status.json",
        symbols: list[str] | None = None,
    ):
        self.data_path = data_path
        self.status_path = Path(status_path)
        if symbols is None:
            symbols = self._configured_equity_symbols() if any(ch in data_path for ch in "*?[]") else []
        self.symbols = set(symbols)
        self.xgb_model = XGBoostModel()
        self.rf_model = RandomForestModel()

    @staticmethod
    def _configured_equity_symbols() -> list[str]:
        try:
            with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            return list(config.get("instruments", {}).get("equity", []) or [])
        except Exception as exc:
            logger.warning(f"Could not load configured equity symbols: {exc}")
            return []

    @staticmethod
    def _symbol_from_path(path: str) -> str:
        name = Path(path).stem
        return name[:-3] if name.endswith("_6m") else name

    def _load_files(self) -> pd.DataFrame:
        files = sorted(glob.glob(self.data_path)) if any(ch in self.data_path for ch in "*?[]") else [self.data_path]
        frames = []
        for path in files:
            if not os.path.exists(path):
                continue
            symbol = self._symbol_from_path(path)
            if self.symbols and symbol not in self.symbols:
                continue
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                logger.warning(f"Skipping unreadable training file {path}: {exc}")
                continue
            required = {"open", "high", "low", "close", "volume"}
            if df.empty or not required.issubset(df.columns):
                logger.warning(f"Skipping invalid training file {path}")
                continue
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()
            df["symbol"] = symbol
            frames.append(df)
            logger.info(f"Loaded {len(df)} rows from {path}")
        return pd.concat(frames).sort_index(kind="stable") if frames else pd.DataFrame()

    def _prepare_symbol_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = PriceFeatures.add_indicators(df.copy())
        lookahead = 3
        df["future_close"] = df["close"].shift(-lookahead)
        forward_return = (df["future_close"] - df["close"]) / df["close"].replace(0, np.nan)

        # Use only prior labels to target about 20% BUY, 20% SELL and 60% NEUTRAL.
        lower = forward_return.rolling(500, min_periods=100).quantile(0.20).shift(lookahead + 1)
        upper = forward_return.rolling(500, min_periods=100).quantile(0.80).shift(lookahead + 1)
        fallback = (df["atr_14"] / df["close"].replace(0, np.nan)) * 0.40
        lower = lower.fillna(-fallback)
        upper = upper.fillna(fallback)
        df["label"] = np.select([forward_return > upper, forward_return < lower], [1, 2], default=0)
        df.loc[df["future_close"].isna(), "label"] = np.nan
        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(subset=FEATURE_COLS + ["label"], inplace=True)
        return df

    @staticmethod
    def _walk_forward_splits(index: pd.Index, n_splits: int = 3):
        timestamps = pd.Index(index.unique()).sort_values()
        if len(timestamps) < n_splits + 2:
            raise ValueError("Not enough unique timestamps for walk-forward validation")
        block = len(timestamps) // (n_splits + 1)
        if block < 1:
            raise ValueError("Walk-forward validation block is empty")
        for fold in range(1, n_splits + 1):
            train_end_pos = block * fold
            test_end_pos = block * (fold + 1) if fold < n_splits else len(timestamps)
            train_end = timestamps[train_end_pos - 1]
            test_start = timestamps[train_end_pos]
            test_end = timestamps[test_end_pos - 1]
            yield (
                fold,
                np.flatnonzero(index <= train_end),
                np.flatnonzero((index >= test_start) & (index <= test_end)),
            )

    @staticmethod
    def _align_probabilities(probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if probs.shape[1] != 3:
            raise ValueError(f"Expected three class probabilities, got {probs.shape}")
        return probs

    @staticmethod
    def _quality_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict:
        y_pred = np.argmax(probabilities, axis=1)
        report_dict = classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=["NEUTRAL", "BUY", "SELL"],
            output_dict=True, zero_division=0,
        )
        return {
            "accuracy": float((y_pred == y_true.to_numpy()).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "buy_f1": float(report_dict["BUY"]["f1-score"]),
            "sell_f1": float(report_dict["SELL"]["f1-score"]),
            "predictions": y_pred,
            "classification_report": classification_report(
                y_true, y_pred, labels=[0, 1, 2], target_names=["NEUTRAL", "BUY", "SELL"],
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        }

    @staticmethod
    def _passes_quality_gate(metrics: dict) -> bool:
        return bool(
            metrics["balanced_accuracy"] >= 0.38
            and metrics["macro_f1"] >= 0.32
            and metrics["buy_f1"] >= 0.20
            and metrics["sell_f1"] >= 0.20
        )

    @staticmethod
    def _average_gate_metrics(metric_rows: list[dict]) -> dict:
        if not metric_rows:
            raise ValueError("No fold metrics available for quality gate")
        return {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in ("balanced_accuracy", "macro_f1", "buy_f1", "sell_f1")
        }

    def _write_status(self, result: dict) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.status_path.with_name(
            f"{self.status_path.name}.{uuid.uuid4().hex}.tmp"
        )
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)

        for attempt in range(5):
            try:
                os.replace(temp_path, self.status_path)
                return
            except PermissionError as exc:
                if attempt == 4:
                    break
                logger.warning(
                    f"Could not replace {self.status_path} on attempt {attempt + 1}: {exc}; retrying."
                )
                time.sleep(0.25 * (attempt + 1))

        fallback_path = self.status_path.with_name(
            f"{self.status_path.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}{self.status_path.suffix}"
        )
        os.replace(temp_path, fallback_path)
        logger.warning(
            f"Could not update locked status file {self.status_path}; wrote fallback {fallback_path} instead."
        )

    def _deploy_atomically(self, X: pd.DataFrame, y: pd.Series) -> None:
        xgb_target = Path(self.xgb_model.model_path)
        rf_target = Path(self.rf_model.model_path)
        xgb_candidate = xgb_target.with_name(f"{xgb_target.stem}.candidate{xgb_target.suffix}")
        rf_candidate = rf_target.with_name(f"{rf_target.stem}.candidate{rf_target.suffix}")
        candidate_xgb = XGBoostModel(str(xgb_candidate))
        candidate_rf = RandomForestModel(str(rf_candidate))
        try:
            candidate_xgb.train(X, y, save=True)
            candidate_rf.train(X, y, save=True)
            if not candidate_xgb.load() or not candidate_rf.load():
                raise RuntimeError("Candidate ensemble failed reload validation")
            candidate_xgb.predict(X.iloc[:2])
            candidate_rf.predict(X.iloc[:2])
            xgb_target.parent.mkdir(parents=True, exist_ok=True)
            rf_target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(xgb_candidate, xgb_target)
            os.replace(rf_candidate, rf_target)
        finally:
            for candidate in (xgb_candidate, rf_candidate):
                if candidate.exists():
                    candidate.unlink()

    def run(self) -> dict:
        result = {
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "status": "failed",
            "deployed": False,
            "data_path": self.data_path,
            "symbols": sorted(self.symbols),
        }
        logger.info("Starting walk-forward ensemble retraining...")
        try:
            df_all = self._load_files()
            if df_all.empty:
                raise RuntimeError(f"No valid equity training data found at {self.data_path}")
            prepared = []
            for symbol, df_symbol in df_all.groupby("symbol", sort=False):
                if len(df_symbol) < 200:
                    logger.warning(f"Skipping {symbol}: only {len(df_symbol)} rows")
                    continue
                frame = self._prepare_symbol_frame(df_symbol.drop(columns=["symbol"]))
                if frame.empty:
                    continue
                frame["symbol"] = symbol
                prepared.append(frame)
                logger.info(f"{symbol}: prepared {len(frame)} rows; labels={frame['label'].value_counts().to_dict()}")
            if not prepared:
                raise RuntimeError("No symbols produced enough labeled rows for training")

            df = pd.concat(prepared).sort_index(kind="stable")
            X = df[FEATURE_COLS].astype(float)
            y = df["label"].astype(int)
            if len(X) < 1000:
                raise RuntimeError(f"Not enough clean training rows: {len(X)}")
            logger.info(f"Training rows: {len(X)} | labels={y.value_counts().to_dict()}")

            fold_results = []
            gate_metric_rows = []
            latest_metrics = None
            for fold, train_index, test_index in self._walk_forward_splits(X.index, n_splits=3):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                if X_train.index.max() >= X_test.index.min():
                    raise RuntimeError(f"Temporal leakage detected in fold {fold}")
                if y_train.nunique() < 3 or y_test.nunique() < 3:
                    raise RuntimeError(f"Fold {fold} does not contain all three classes")
                logger.info(
                    f"Fold {fold}: train {X_train.index.min()} -> {X_train.index.max()} ({len(X_train)} rows), "
                    f"test {X_test.index.min()} -> {X_test.index.max()} ({len(X_test)} rows)"
                )
                self.xgb_model.train(X_train, y_train, save=False)
                self.rf_model.train(X_train, y_train, save=False)
                xgb_probs = self._align_probabilities(self.xgb_model.predict(X_test))
                rf_probs = self._align_probabilities(self.rf_model.predict(X_test))
                metrics = self._quality_metrics(y_test, (xgb_probs + rf_probs) / 2.0)
                latest_metrics = metrics
                gate_metric_rows.append({
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "buy_f1": metrics["buy_f1"],
                    "sell_f1": metrics["sell_f1"],
                })
                fold_results.append({
                    "fold": fold,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "train_end": X_train.index.max(),
                    "test_start": X_test.index.min(),
                    **{key: value for key, value in metrics.items() if key != "predictions"},
                })
                logger.info(
                    f"Fold {fold}: balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
                    f"macro_f1={metrics['macro_f1']:.3f}, buy_f1={metrics['buy_f1']:.3f}, "
                    f"sell_f1={metrics['sell_f1']:.3f}"
                )

            avg_metrics = self._average_gate_metrics(gate_metric_rows)
            deploy_ok = self._passes_quality_gate(avg_metrics)
            if deploy_ok:
                self._deploy_atomically(X, y)
                logger.success("Average walk-forward quality gates passed; ensemble deployed atomically.")
            else:
                logger.warning(
                    "Average walk-forward quality gates failed; production models were left unchanged. "
                    f"avg_metrics={avg_metrics}"
                )
            result.update({
                "status": "completed",
                "deployed": deploy_ok,
                "rows": len(X),
                "feature_cols": FEATURE_COLS,
                "label_distribution": y.value_counts().to_dict(),
                "folds": fold_results,
                "latest_fold": {key: value for key, value in latest_metrics.items() if key != "predictions"},
                "average_fold_metrics": avg_metrics,
                "quality_gate_basis": "average_walk_forward_folds",
                "quality_gates": {
                    "balanced_accuracy_min": 0.38,
                    "macro_f1_min": 0.32,
                    "buy_f1_min": 0.20,
                    "sell_f1_min": 0.20,
                },
            })
        except Exception as exc:
            result["error"] = str(exc)
            logger.exception(f"Retraining failed: {exc}")
        finally:
            result["completed_at"] = datetime.now().isoformat()
            self._write_status(result)
            logger.info(f"Retraining status saved to {self.status_path}")
        return result


if __name__ == "__main__":
    outcome = RetrainPipeline().run()
    raise SystemExit(0 if outcome["status"] == "completed" else 1)
