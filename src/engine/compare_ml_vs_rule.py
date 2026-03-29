from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.engine.ml_vs_rule_comparison import (
    M4_ML_VS_RULE_COMPARISON_CONFIG_PATH,
    run_m4_ml_vs_rule_comparison,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare official M4 model outputs against the rule-based momentum strategy.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a baseline_model_predictions.parquet artifact produced by the M4 batch prediction pipeline.",
    )
    parser.add_argument(
        "--metadata",
        help="Optional path to the baseline_model_predictions.metadata.json sidecar. Defaults to the official sibling filename.",
    )
    parser.add_argument(
        "--config",
        default=str(M4_ML_VS_RULE_COMPARISON_CONFIG_PATH),
        help="Path to the official M4 ML-vs-rule comparison config YAML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_m4_ml_vs_rule_comparison(
            predictions_path=Path(args.predictions),
            metadata_path=Path(args.metadata) if args.metadata else None,
            config_path=Path(args.config),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] ML-vs-rule comparison failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print("-" * 60)
    print("M4 ML-vs-rule comparison completed")
    print(f"Run id: {result['run_id']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Aligned comparison: {result['aligned_path']}")
    print(f"Summary JSON: {result['summary_json_path']}")
    print(f"Summary CSV: {result['summary_csv_path']}")
    print(f"Per-symbol summary CSV: {result['per_symbol_summary_csv_path']}")
    print(f"Rule replay manifest: {result['rule_replay_manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
