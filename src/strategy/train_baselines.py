from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.strategy.ml_baselines import (
    M4_BASELINE_TRAINING_CONFIG_PATH,
    run_m4_baseline_training,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train official M4 tabular baseline models.")
    parser.add_argument(
        "--config",
        default=str(M4_BASELINE_TRAINING_CONFIG_PATH),
        help="Path to the official M4 baseline training config YAML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)

    try:
        result = run_m4_baseline_training(config_path=config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Baseline training failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print("-" * 60)
    print("M4 baseline training completed")
    print(f"Run id: {result['run_id']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    for model_record in result["models"]:
        print(
            f"  - {model_record['model_name']}: {model_record['artifact_path']} "
            f"(accuracy={model_record['metrics'].get('accuracy', 'n/a')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
