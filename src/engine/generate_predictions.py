from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.engine.prediction_pipeline import (
    M4_BATCH_PREDICTION_CONFIG_PATH,
    run_m4_batch_prediction,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate official M4 batch predictions from trained baseline models.")
    parser.add_argument(
        "--training-summary",
        required=True,
        help="Path to a baseline_training_summary.json artifact produced by baseline training.",
    )
    parser.add_argument(
        "--config",
        default=str(M4_BATCH_PREDICTION_CONFIG_PATH),
        help="Path to the official M4 batch prediction config YAML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_m4_batch_prediction(
            training_summary_path=Path(args.training_summary),
            config_path=Path(args.config),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Batch prediction failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print("-" * 60)
    print("M4 batch prediction completed")
    print(f"Run id: {result['run_id']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Summary JSON: {result['summary_json_path']}")
    print(f"Predictions parquet: {result['predictions_path']}")
    print(f"Prediction rows: {result['prediction_row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
