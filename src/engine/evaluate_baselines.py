from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.engine.model_evaluation import (
    M4_BASELINE_EVALUATION_CONFIG_PATH,
    run_m4_baseline_evaluation,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate official M4 baseline model artifacts.")
    parser.add_argument(
        "--training-summary",
        required=True,
        help="Path to a baseline_training_summary.json artifact produced by baseline training.",
    )
    parser.add_argument(
        "--config",
        default=str(M4_BASELINE_EVALUATION_CONFIG_PATH),
        help="Path to the official M4 baseline evaluation config YAML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_m4_baseline_evaluation(
            training_summary_path=Path(args.training_summary),
            config_path=Path(args.config),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Baseline evaluation failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print("-" * 60)
    print("M4 baseline evaluation completed")
    print(f"Run id: {result['run_id']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Summary JSON: {result['summary_json_path']}")
    print(f"Summary CSV: {result['summary_csv_path']}")
    for model_report in result["model_reports"]:
        print(
            f"  - {model_report['model_name']}: {model_report['report_path']} "
            f"(accuracy={model_report['metrics'].get('accuracy', 'n/a')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
