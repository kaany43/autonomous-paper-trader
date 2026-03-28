from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.modeling_dataset import (
    FEATURES_INPUT_FILE,
    M4_MODELING_DATASET_CONFIG_PATH,
    MODELING_METADATA_FILE,
    MODELING_OUTPUT_FILE,
    OfficialM4ModelingDatasetDefinition,
    build_m4_modeling_dataset_metadata,
    load_m4_modeling_dataset_definition,
    normalize_m4_modeling_dataset,
    save_m4_modeling_dataset,
    validate_m4_modeling_dataset_contract,
)
from src.data.splits import OfficialM4SplitDefinition, load_m4_split_definition
from src.data.targets import (
    TARGET_COLUMN_PREFIX,
    OfficialTargetDefinition,
    add_m4_target_columns,
    load_m4_target_definition,
)


def _strip_existing_target_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    existing_target_columns = [
        str(column) for column in df.columns if str(column).startswith(TARGET_COLUMN_PREFIX)
    ]
    if not existing_target_columns:
        return df.copy(), []
    return df.drop(columns=existing_target_columns).copy(), existing_target_columns


def prepare_m4_modeling_dataset(
    features_df: pd.DataFrame,
    definition: OfficialTargetDefinition | None = None,
    dataset_definition: OfficialM4ModelingDatasetDefinition | None = None,
    split_definition: OfficialM4SplitDefinition | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build the official M4 modeling dataset from processed daily market features.

    The returned dataframe is deterministic, normalized to the official schema,
    and contains only rows with valid future labels.
    """
    if features_df.empty:
        raise ValueError("Processed feature dataframe is empty.")

    target_definition = definition or load_m4_target_definition()
    resolved_dataset_definition = dataset_definition or load_m4_modeling_dataset_definition()
    resolved_split_definition = split_definition or load_m4_split_definition()

    stripped_features, dropped_input_target_columns = _strip_existing_target_columns(features_df)
    labeled_df = add_m4_target_columns(stripped_features, target_definition)
    modeling_df = labeled_df.loc[
        labeled_df[resolved_dataset_definition.target_valid_column]
    ].reset_index(drop=True)
    modeling_df = normalize_m4_modeling_dataset(
        modeling_df,
        dataset_definition=resolved_dataset_definition,
        target_definition=target_definition,
    )
    validate_m4_modeling_dataset_contract(
        modeling_df,
        dataset_definition=resolved_dataset_definition,
        target_definition=target_definition,
        split_definition=resolved_split_definition,
    )

    summary = {
        "input_row_count": int(len(features_df)),
        "normalized_row_count": int(len(labeled_df)),
        "dropped_duplicate_row_count": int(len(stripped_features) - len(labeled_df)),
        "output_row_count": int(len(modeling_df)),
        "dropped_invalid_row_count": int(len(labeled_df) - len(modeling_df)),
        "dropped_input_target_columns": dropped_input_target_columns,
    }
    return modeling_df, summary


def run_m4_target_preparation(
    *,
    input_path: Path = FEATURES_INPUT_FILE,
    output_path: Path = MODELING_OUTPUT_FILE,
    metadata_path: Path = MODELING_METADATA_FILE,
    definition: OfficialTargetDefinition | None = None,
    dataset_definition: OfficialM4ModelingDatasetDefinition | None = None,
    split_definition: OfficialM4SplitDefinition | None = None,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed feature dataset not found: {input_path}. "
            "Run `python -m src.data.features` first."
        )

    target_definition = definition or load_m4_target_definition()
    resolved_dataset_definition = dataset_definition or load_m4_modeling_dataset_definition()
    resolved_split_definition = split_definition or load_m4_split_definition()

    features_df = pd.read_parquet(input_path)
    modeling_df, summary = prepare_m4_modeling_dataset(
        features_df,
        definition=target_definition,
        dataset_definition=resolved_dataset_definition,
        split_definition=resolved_split_definition,
    )
    metadata = build_m4_modeling_dataset_metadata(
        input_path=input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        features_df=features_df,
        modeling_df=modeling_df,
        summary=summary,
        dataset_definition=resolved_dataset_definition,
        target_definition=target_definition,
        split_definition=resolved_split_definition,
    )
    saved_output_path, saved_metadata_path = save_m4_modeling_dataset(
        modeling_df,
        output_path=output_path,
        metadata=metadata,
        metadata_path=metadata_path,
    )

    return {
        "input_path": input_path,
        "output_path": saved_output_path,
        "metadata_path": saved_metadata_path,
        "dataframe": modeling_df,
        "metadata": metadata,
        **summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the official M4 modeling dataset.")
    parser.add_argument("--input", default=str(FEATURES_INPUT_FILE), help="Path to processed feature parquet.")
    parser.add_argument("--output", default=str(MODELING_OUTPUT_FILE), help="Path to modeling dataset parquet.")
    parser.add_argument(
        "--metadata-output",
        default=str(MODELING_METADATA_FILE),
        help="Path to the metadata JSON sidecar.",
    )
    parser.add_argument(
        "--dataset-config",
        default=str(M4_MODELING_DATASET_CONFIG_PATH),
        help="Path to the official modeling dataset config YAML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    dataset_config_path = Path(args.dataset_config)

    try:
        dataset_definition = load_m4_modeling_dataset_definition(dataset_config_path)
        result = run_m4_target_preparation(
            input_path=input_path,
            output_path=output_path,
            metadata_path=metadata_path,
            dataset_definition=dataset_definition,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Target preparation failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print("-" * 60)
    print("M4 target preparation completed")
    print(f"Input features: {result['input_path']}")
    print(f"Output dataset: {result['output_path']}")
    print(f"Metadata: {result['metadata_path']}")
    print(f"Rows saved: {result['output_row_count']}")
    print(f"Dropped duplicate rows: {result['dropped_duplicate_row_count']}")
    print(f"Dropped invalid rows: {result['dropped_invalid_row_count']}")
    print("Schema columns: " + ", ".join(result["metadata"]["schema"]["column_order"]))
    if result["dropped_input_target_columns"]:
        print("Dropped legacy input target columns: " + ", ".join(result["dropped_input_target_columns"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
