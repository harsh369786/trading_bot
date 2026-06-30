import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.retrain_pipeline import RetrainPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain ensemble models with walk-forward validation.")
    parser.add_argument(
        "--data",
        default="data/historical/*_6m.parquet",
        help="Parquet glob or file path used for ensemble retraining.",
    )
    parser.add_argument(
        "--status",
        default="data/retrain_status.json",
        help="JSON status file written after validation/deployment.",
    )
    args = parser.parse_args()
    result = RetrainPipeline(data_path=args.data, status_path=args.status).run()
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
