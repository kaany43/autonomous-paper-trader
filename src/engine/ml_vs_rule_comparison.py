from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_yaml
from src.data.prediction_logs import (
    load_m4_prediction_log_bundle,
    load_m4_prediction_log_definition,
)
from src.engine.broker import Broker
from src.engine.model_evaluation import load_m4_baseline_training_run_bundle
from src.engine.portfolio import Portfolio
from src.engine.run_artifacts import RunArtifactManager
from src.engine.simulator import DailySimulator
from src.strategy.ml_baselines import build_dataframe_signature, prepare_m4_baseline_training_data
from src.strategy.momentum import MomentumStrategy


REPO_ROOT = Path(__file__).resolve().parents[2]
M4_ML_VS_RULE_COMPARISON_CONFIG_PATH = REPO_ROOT / "config" / "evaluation" / "m4_ml_vs_rule_comparison.yaml"
ALIGNED_COMPARISON_FILENAME = "ml_vs_rule_aligned.parquet"
ALIGNED_COMPARISON_METADATA_FILENAME = "ml_vs_rule_aligned.metadata.json"
COMPARISON_SUMMARY_JSON_FILENAME = "ml_vs_rule_summary.json"
COMPARISON_SUMMARY_CSV_FILENAME = "ml_vs_rule_summary.csv"
PER_SYMBOL_SUMMARY_CSV_FILENAME = "ml_vs_rule_by_symbol.csv"
RULE_REPLAY_RUNS_DIRNAME = "rule_strategy_runs"
PIPELINE_VERSION = 1
ENTRYPOINT = "python -m src.engine.compare_ml_vs_rule"
COMPARISON_KEY_COLUMNS = ["symbol", "date", "target_date"]
ALIGNED_SORT_ORDER = ["model_name", "symbol", "date", "target_date"]
SUMMARY_ROW_COLUMNS = [
    "model_name",
    "estimator",
    "prediction_run_id",
    "training_run_id",
    "row_count",
    "symbol_count",
    "agreement_rate",
    "disagreement_rate",
    "shared_entry_count",
    "ml_only_entry_count",
    "rule_only_entry_count",
    "shared_non_entry_count",
    "rule_buy_count",
    "rule_sell_count",
    "rule_hold_count",
    "model_positive_rate",
    "rule_buy_rate",
    "rule_sell_rate",
    "rule_hold_rate",
    "actual_positive_rate",
    "ml_accuracy_vs_actual",
    "rule_entry_accuracy_vs_actual",
    "actual_positive_rate_both_entry",
    "actual_positive_rate_ml_only_entry",
    "actual_positive_rate_rule_only_entry",
    "actual_positive_rate_shared_non_entry",
    "mean_predicted_probability",
    "mean_predicted_probability_rule_buy",
    "mean_predicted_probability_rule_sell",
    "mean_predicted_probability_rule_hold",
]
PER_SYMBOL_SUMMARY_COLUMNS = [
    "model_name",
    "symbol",
    "row_count",
    "agreement_rate",
    "shared_entry_count",
    "ml_only_entry_count",
    "rule_only_entry_count",
    "shared_non_entry_count",
    "rule_buy_count",
    "rule_sell_count",
    "rule_hold_count",
    "actual_positive_rate",
    "ml_accuracy_vs_actual",
    "rule_entry_accuracy_vs_actual",
    "mean_predicted_probability",
]
ALIGNED_COMPARISON_COLUMNS = [
    "prediction_run_id",
    "training_run_id",
    "inference_partition",
    "model_name",
    "estimator",
    "model_artifact_path",
    "model_metadata_path",
    "symbol",
    "date",
    "target_date",
    "target_column",
    "task_type",
    "predicted_class",
    "predicted_probability",
    "actual_target",
    "actual_target_return",
    "rule_action",
    "rule_reason_code",
    "rule_score",
    "rule_target_weight",
    "rule_schedule_status",
    "rule_scheduled_execution_date",
    "rule_entry_signal",
    "rule_exit_signal",
    "comparison_outcome",
    "is_agreement",
    "is_disagreement",
    "ml_matches_actual_target",
    "rule_entry_matches_actual_target",
    "probability_distance_from_threshold",
]


@dataclass(frozen=True)
class OfficialM4MLVsRuleComparisonDefinition:
    milestone: str
    contract_name: str
    version: int
    training_config_path: str
    prediction_config_path: str
    prediction_log_config_path: str
    strategy_config_path: str
    feature_dataset_path: str
    output_dir: str
    strategy_name: str
    run_label: str
    methodology_name: str
    comparison_signal_column: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)
    return path


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value).strip())
    if not str(path):
        raise ValueError("Configured path value cannot be empty.")
    return path if path.is_absolute() else REPO_ROOT / path


def _format_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _normalize_datetime_series(series: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(series, errors="coerce")
    if getattr(normalized.dt, "tz", None) is not None:
        normalized = normalized.dt.tz_localize(None)
    return normalized.astype("datetime64[ns]")


def _normalize_symbol_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def _validate_binary_series(values: pd.Series, *, label: str) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.isna().any():
        raise ValueError(f"Comparison requires '{label}' to contain only numeric 0/1 values.")
    if not bool((numeric_values == numeric_values.round()).all()):
        raise ValueError(f"Comparison requires '{label}' to contain exact integer 0/1 values.")
    normalized = numeric_values.astype("int64")
    if not set(normalized.tolist()).issubset({0, 1}):
        raise ValueError(f"Comparison requires '{label}' to contain only 0/1 values.")
    return normalized


def _mean_or_none(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _comparison_outcome(row: pd.Series) -> str:
    if int(row["predicted_class"]) == 1 and int(row["rule_entry_signal"]) == 1:
        return "both_entry"
    if int(row["predicted_class"]) == 1 and int(row["rule_entry_signal"]) == 0:
        return "ml_only_entry"
    if int(row["predicted_class"]) == 0 and int(row["rule_entry_signal"]) == 1:
        return "rule_only_entry"
    return "shared_non_entry"


def load_m4_ml_vs_rule_comparison_definition(
    config_path: Path = M4_ML_VS_RULE_COMPARISON_CONFIG_PATH,
) -> OfficialM4MLVsRuleComparisonDefinition:
    data = load_yaml(config_path)
    comparison_cfg = data.get("comparison")
    if not isinstance(comparison_cfg, dict):
        raise ValueError(f"Missing or invalid comparison config in: {config_path}")

    definition = OfficialM4MLVsRuleComparisonDefinition(
        milestone=str(comparison_cfg.get("milestone", "")).strip(),
        contract_name=str(comparison_cfg.get("contract_name", "")).strip(),
        version=int(comparison_cfg.get("version", 0) or 0),
        training_config_path=str(comparison_cfg.get("training_config_path", "")).strip(),
        prediction_config_path=str(comparison_cfg.get("prediction_config_path", "")).strip(),
        prediction_log_config_path=str(comparison_cfg.get("prediction_log_config_path", "")).strip(),
        strategy_config_path=str(comparison_cfg.get("strategy_config_path", "")).strip(),
        feature_dataset_path=str(comparison_cfg.get("feature_dataset_path", "")).strip(),
        output_dir=str(comparison_cfg.get("output_dir", "")).strip(),
        strategy_name=str(comparison_cfg.get("strategy_name", "")).strip(),
        run_label=str(comparison_cfg.get("run_label", "")).strip(),
        methodology_name=str(comparison_cfg.get("methodology_name", "")).strip(),
        comparison_signal_column=str(comparison_cfg.get("comparison_signal_column", "")).strip(),
    )

    if definition.milestone != "M4":
        raise ValueError("Official M4 ML-vs-rule comparison milestone must be 'M4'.")
    if not definition.contract_name:
        raise ValueError("Official M4 ML-vs-rule comparison contract_name is required.")
    if definition.version < 1:
        raise ValueError("Official M4 ML-vs-rule comparison version must be >= 1.")
    if definition.training_config_path != "config/modeling/m4_baselines.yaml":
        raise ValueError(
            "Official M4 ML-vs-rule comparison training_config_path must be "
            "'config/modeling/m4_baselines.yaml'."
        )
    if definition.prediction_config_path != "config/evaluation/m4_batch_prediction.yaml":
        raise ValueError(
            "Official M4 ML-vs-rule comparison prediction_config_path must be "
            "'config/evaluation/m4_batch_prediction.yaml'."
        )
    if definition.prediction_log_config_path != "config/modeling/m4_prediction_logs.yaml":
        raise ValueError(
            "Official M4 ML-vs-rule comparison prediction_log_config_path must be "
            "'config/modeling/m4_prediction_logs.yaml'."
        )
    if definition.strategy_config_path != "config/settings.yaml":
        raise ValueError(
            "Official M4 ML-vs-rule comparison strategy_config_path must be 'config/settings.yaml'."
        )
    if definition.feature_dataset_path != "data/processed/market_features.parquet":
        raise ValueError(
            "Official M4 ML-vs-rule comparison feature_dataset_path must be "
            "'data/processed/market_features.parquet'."
        )
    if definition.output_dir != "outputs/reports/ml_vs_rule_comparisons":
        raise ValueError(
            "Official M4 ML-vs-rule comparison output_dir must be "
            "'outputs/reports/ml_vs_rule_comparisons'."
        )
    if not definition.strategy_name:
        raise ValueError("Official M4 ML-vs-rule comparison strategy_name is required.")
    if not definition.run_label:
        raise ValueError("Official M4 ML-vs-rule comparison run_label is required.")
    if definition.methodology_name != "validation_signal_alignment":
        raise ValueError(
            "Official M4 ML-vs-rule comparison methodology_name must be "
            "'validation_signal_alignment'."
        )
    if definition.comparison_signal_column != "rule_entry_signal":
        raise ValueError(
            "Official M4 ML-vs-rule comparison comparison_signal_column must be 'rule_entry_signal'."
        )

    return definition


def _build_validation_rows(
    validation_df: pd.DataFrame,
    *,
    actual_target_column: str,
    actual_return_column: str,
) -> pd.DataFrame:
    required_columns = ["symbol", "date", "target_date", actual_target_column, actual_return_column]
    missing_columns = [column for column in required_columns if column not in validation_df.columns]
    if missing_columns:
        raise ValueError(
            "Validation dataframe is missing required comparison columns: " + ", ".join(missing_columns)
        )

    rows = validation_df.loc[:, required_columns].copy()
    rows["symbol"] = _normalize_symbol_series(rows["symbol"])
    rows["date"] = _normalize_datetime_series(rows["date"])
    rows["target_date"] = _normalize_datetime_series(rows["target_date"])
    if rows["date"].isna().any() or rows["target_date"].isna().any():
        raise ValueError("Validation dataframe contains invalid comparison timestamps.")
    if not bool((rows["target_date"] > rows["date"]).all()):
        raise ValueError("Validation dataframe contains non-forward target timestamps.")

    duplicate_feature_keys = rows.duplicated(subset=["symbol", "date"])
    if duplicate_feature_keys.any():
        raise ValueError("Validation dataframe contains duplicate symbol/date rows.")
    duplicate_alignment_keys = rows.duplicated(subset=COMPARISON_KEY_COLUMNS)
    if duplicate_alignment_keys.any():
        raise ValueError("Validation dataframe contains duplicate symbol/date/target_date rows.")

    rows["actual_target"] = _validate_binary_series(rows[actual_target_column], label=actual_target_column)
    rows["actual_target_return"] = pd.to_numeric(rows[actual_return_column], errors="coerce").astype("Float64")

    rows = rows.sort_values(COMPARISON_KEY_COLUMNS).reset_index(drop=True)
    return rows.loc[:, ["symbol", "date", "target_date", "actual_target", "actual_target_return"]]


def _validate_prediction_rows(
    prediction_df: pd.DataFrame,
    *,
    validation_rows: pd.DataFrame,
    expected_target_column: str,
    expected_task_type: str,
) -> None:
    if prediction_df.empty:
        raise ValueError("Prediction log produced no comparison rows.")
    if prediction_df["prediction_run_id"].nunique(dropna=False) != 1:
        raise ValueError("Comparison requires prediction rows from a single prediction_run_id.")
    if prediction_df["training_run_id"].nunique(dropna=False) != 1:
        raise ValueError("Comparison requires prediction rows from a single training_run_id.")
    if prediction_df["inference_partition"].nunique(dropna=False) != 1:
        raise ValueError("Comparison requires prediction rows from a single inference partition.")
    if prediction_df["inference_partition"].iloc[0] != "validation":
        raise ValueError("Comparison requires validation prediction rows.")

    observed_target_columns = sorted(prediction_df["target_column"].dropna().astype(str).str.strip().unique().tolist())
    if observed_target_columns != [expected_target_column]:
        raise ValueError("Prediction log target_column does not match the official M4 comparison target.")
    observed_task_types = (
        prediction_df["task_type"].dropna().astype(str).str.strip().str.lower().unique().tolist()
    )
    if observed_task_types != [expected_task_type]:
        raise ValueError("Prediction log task_type does not match the official M4 comparison task.")

    expected_keys = validation_rows.loc[:, COMPARISON_KEY_COLUMNS].reset_index(drop=True).copy()
    expected_keys["date"] = _normalize_datetime_series(expected_keys["date"])
    expected_keys["target_date"] = _normalize_datetime_series(expected_keys["target_date"])
    expected_index = pd.MultiIndex.from_frame(expected_keys)
    expected_row_count = len(expected_keys)

    for model_name, model_rows in prediction_df.groupby("model_name", sort=True):
        observed_keys = model_rows.loc[:, COMPARISON_KEY_COLUMNS].reset_index(drop=True).copy()
        observed_keys["date"] = _normalize_datetime_series(observed_keys["date"])
        observed_keys["target_date"] = _normalize_datetime_series(observed_keys["target_date"])
        if len(observed_keys) != expected_row_count:
            raise ValueError(
                f"Prediction rows for model '{model_name}' do not match the official validation row count."
            )

        observed_index = pd.MultiIndex.from_frame(observed_keys)
        missing_keys = expected_index.difference(observed_index)
        extra_keys = observed_index.difference(expected_index)
        if len(missing_keys) or len(extra_keys):
            details: list[str] = []
            if len(missing_keys):
                details.append(f"missing_keys={len(missing_keys)}")
            if len(extra_keys):
                details.append(f"extra_keys={len(extra_keys)}")
            raise ValueError(
                f"Prediction rows for model '{model_name}' do not match official validation keys. "
                + ", ".join(details)
            )

        if not observed_keys.equals(expected_keys):
            raise ValueError(
                f"Prediction rows for model '{model_name}' are not sorted by the official validation key order."
            )


def _load_rule_settings(strategy_config_path: Path) -> dict[str, Any]:
    settings = load_yaml(strategy_config_path)
    if not isinstance(settings, dict):
        raise ValueError(f"Rule strategy config must be a mapping: {strategy_config_path}")
    for section in ("strategy", "portfolio", "execution"):
        if not isinstance(settings.get(section), dict):
            raise ValueError(f"Rule strategy config is missing required '{section}' section: {strategy_config_path}")
    return settings


def _load_rule_feature_history(
    *,
    feature_dataset_path: Path,
    comparison_symbols: list[str],
    validation_rows: pd.DataFrame,
) -> pd.DataFrame:
    if not feature_dataset_path.exists():
        raise FileNotFoundError(f"Missing rule feature dataset: {feature_dataset_path}")

    feature_df = pd.read_parquet(feature_dataset_path)
    if feature_df.empty:
        raise ValueError("Rule feature dataset is empty.")

    required_columns = {
        "date",
        "symbol",
        "adj_close",
        "ret_5d",
        "ma_20",
        "ma_50",
        "price_vs_ma20",
        "ma20_vs_ma50",
        "volume_ratio_20",
    }
    missing_columns = required_columns - set(feature_df.columns)
    if missing_columns:
        raise ValueError(
            "Rule feature dataset is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    filtered = feature_df.copy()
    filtered["date"] = _normalize_datetime_series(filtered["date"])
    filtered["symbol"] = _normalize_symbol_series(filtered["symbol"])
    if filtered["date"].isna().any():
        raise ValueError("Rule feature dataset contains invalid dates.")

    symbol_set = set(comparison_symbols)
    validation_end_date = pd.Timestamp(validation_rows["date"].max())
    filtered = filtered.loc[
        filtered["symbol"].isin(symbol_set) & (filtered["date"] <= validation_end_date)
    ].copy()
    if filtered.empty:
        raise ValueError("Rule feature dataset has no rows for the comparison symbols and date window.")

    duplicate_keys = filtered.duplicated(subset=["symbol", "date"])
    if duplicate_keys.any():
        raise ValueError("Rule feature dataset contains duplicate symbol/date rows.")

    expected_feature_keys = validation_rows.loc[:, ["symbol", "date"]].drop_duplicates().reset_index(drop=True)
    expected_index = pd.MultiIndex.from_frame(expected_feature_keys)
    observed_index = pd.MultiIndex.from_frame(filtered.loc[:, ["symbol", "date"]])
    missing_validation_keys = expected_index.difference(observed_index)
    if len(missing_validation_keys):
        raise ValueError("Rule feature dataset is missing validation rows required for comparison.")

    filtered = filtered.sort_values(["date", "symbol"]).reset_index(drop=True)
    return filtered


def _run_rule_strategy_replay(
    *,
    feature_history: pd.DataFrame,
    settings: dict[str, Any],
    comparison_run_id: str,
    comparison_dir: Path,
    config_source: Path,
) -> dict[str, Any]:
    strategy_cfg = settings.get("strategy", {})
    portfolio_cfg = settings.get("portfolio", {})
    execution_cfg = settings.get("execution", {})
    benchmark_cfg = settings.get("benchmark", {})

    portfolio = Portfolio(initial_cash=float(portfolio_cfg.get("initial_cash", 0.0)))
    broker = Broker(
        commission_rate=float(execution_cfg.get("commission_rate", 0.0)),
        slippage_rate=float(execution_cfg.get("slippage_rate", 0.0)),
        fractional_shares=bool(portfolio_cfg.get("fractional_shares", True)),
    )
    strategy = MomentumStrategy(
        max_open_positions=int(portfolio_cfg.get("max_open_positions", 1)),
        top_k=int(strategy_cfg.get("top_k", 1)),
        min_score=float(strategy_cfg.get("min_score", 0.0)),
        min_volume_ratio=float(strategy_cfg.get("min_volume_ratio", 0.8)),
    )
    benchmark_symbol = str(benchmark_cfg.get("benchmark_symbol", "")).strip().upper()
    comparison_symbols = sorted(feature_history["symbol"].dropna().astype(str).str.upper().unique().tolist())

    simulator = DailySimulator(
        market_data=feature_history,
        strategy=strategy,
        portfolio=portfolio,
        broker=broker,
        price_column="adj_close",
    )

    rule_runs_dir = comparison_dir / RULE_REPLAY_RUNS_DIRNAME
    rule_runs_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "strategy_name": "momentum_v0",
        "strategy_parameters": {
            "max_open_positions": strategy.max_open_positions,
            "top_k": strategy.top_k,
            "min_score": strategy.min_score,
            "min_volume_ratio": strategy.min_volume_ratio,
        },
        "broker": {
            "commission_rate": float(broker.commission_rate),
            "slippage_rate": float(broker.slippage_rate),
            "fractional_shares": bool(broker.fractional_shares),
        },
        "portfolio": {
            "initial_cash": float(portfolio.initial_cash),
            "max_open_positions": int(portfolio_cfg.get("max_open_positions", 1)),
        },
        "comparison": {
            "entrypoint": "src.engine.compare_ml_vs_rule",
            "comparison_run_id": comparison_run_id,
        },
        "strategy_variant": {"name": "rule_replay", "params": dict(strategy_cfg)},
    }

    import src.engine.simulator as simulator_module

    original_backtests_dir = simulator_module.BACKTEST_OUTPUTS_DIR
    simulator_module.BACKTEST_OUTPUTS_DIR = rule_runs_dir
    try:
        result = simulator.run(
            start_date=feature_history["date"].min(),
            end_date=feature_history["date"].max(),
            benchmark_symbol=benchmark_symbol,
            equal_weight_universe=comparison_symbols,
            run_config=run_config,
            config_source=str(config_source),
            run_label="rule_strategy_replay",
        )
    finally:
        simulator_module.BACKTEST_OUTPUTS_DIR = original_backtests_dir

    return result


def _build_rule_signal_rows(
    *,
    signal_history: pd.DataFrame,
    validation_rows: pd.DataFrame,
) -> pd.DataFrame:
    if signal_history.empty:
        normalized_signals = pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "rule_action",
                "rule_reason_code",
                "rule_score",
                "rule_target_weight",
                "rule_schedule_status",
                "rule_scheduled_execution_date",
            ]
        )
    else:
        required_columns = {"date", "symbol", "action"}
        missing_columns = required_columns - set(signal_history.columns)
        if missing_columns:
            raise ValueError(
                "Rule strategy replay is missing required signal columns: " + ", ".join(sorted(missing_columns))
            )

        normalized_signals = signal_history.copy()
        normalized_signals["date"] = _normalize_datetime_series(normalized_signals["date"])
        normalized_signals["symbol"] = _normalize_symbol_series(normalized_signals["symbol"])
        if normalized_signals["date"].isna().any():
            raise ValueError("Rule strategy replay contains invalid signal dates.")
        if normalized_signals["symbol"].eq("").any():
            raise ValueError("Rule strategy replay contains invalid empty symbols.")

        duplicate_keys = normalized_signals.duplicated(subset=["symbol", "date"])
        if duplicate_keys.any():
            raise ValueError("Rule strategy replay produced duplicate symbol/date signals.")

        normalized_signals["rule_action"] = normalized_signals["action"].astype(str).str.upper().str.strip()
        invalid_actions = sorted(set(normalized_signals["rule_action"]) - {"BUY", "SELL", "HOLD"})
        if invalid_actions:
            raise ValueError("Rule strategy replay produced unsupported actions: " + ", ".join(invalid_actions))

        normalized_signals["rule_reason_code"] = (
            normalized_signals["reason_code"].fillna("").astype(str).str.strip()
            if "reason_code" in normalized_signals.columns
            else ""
        )
        normalized_signals["rule_score"] = (
            pd.to_numeric(normalized_signals["score"], errors="coerce").astype("Float64")
            if "score" in normalized_signals.columns
            else pd.Series(pd.NA, index=normalized_signals.index, dtype="Float64")
        )
        normalized_signals["rule_target_weight"] = (
            pd.to_numeric(normalized_signals["target_weight"], errors="coerce").astype("Float64")
            if "target_weight" in normalized_signals.columns
            else pd.Series(pd.NA, index=normalized_signals.index, dtype="Float64")
        )
        normalized_signals["rule_schedule_status"] = (
            normalized_signals["schedule_status"].fillna("").astype(str).str.strip()
            if "schedule_status" in normalized_signals.columns
            else ""
        )
        normalized_signals["rule_scheduled_execution_date"] = (
            _normalize_datetime_series(normalized_signals["scheduled_execution_date"])
            if "scheduled_execution_date" in normalized_signals.columns
            else pd.Series(pd.NaT, index=normalized_signals.index, dtype="datetime64[ns]")
        )

        normalized_signals = normalized_signals.loc[
            :,
            [
                "symbol",
                "date",
                "rule_action",
                "rule_reason_code",
                "rule_score",
                "rule_target_weight",
                "rule_schedule_status",
                "rule_scheduled_execution_date",
            ],
        ]

    rule_rows = validation_rows.loc[:, COMPARISON_KEY_COLUMNS].copy()
    rule_rows = rule_rows.merge(normalized_signals, on=["symbol", "date"], how="left", validate="one_to_one")
    rule_rows["rule_action"] = rule_rows["rule_action"].fillna("HOLD")
    rule_rows["rule_reason_code"] = rule_rows["rule_reason_code"].fillna("")
    rule_rows["rule_schedule_status"] = rule_rows["rule_schedule_status"].fillna("NO_SIGNAL")
    rule_rows["rule_entry_signal"] = (rule_rows["rule_action"] == "BUY").astype("int64")
    rule_rows["rule_exit_signal"] = (rule_rows["rule_action"] == "SELL").astype("int64")
    rule_rows = rule_rows.sort_values(COMPARISON_KEY_COLUMNS).reset_index(drop=True)
    return rule_rows


def _build_aligned_comparison_table(
    *,
    prediction_df: pd.DataFrame,
    validation_rows: pd.DataFrame,
    rule_rows: pd.DataFrame,
) -> pd.DataFrame:
    aligned = prediction_df.merge(
        validation_rows,
        on=COMPARISON_KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    )
    if len(aligned) != len(prediction_df):
        raise ValueError("Prediction log rows do not fully align to the official validation rows.")

    aligned = aligned.merge(
        rule_rows,
        on=COMPARISON_KEY_COLUMNS,
        how="inner",
        validate="many_to_one",
    )
    if len(aligned) != len(prediction_df):
        raise ValueError("Rule strategy outputs do not fully align to the official validation rows.")
    if aligned.empty:
        raise ValueError("Comparison alignment produced no overlapping rows.")

    aligned["comparison_outcome"] = aligned.apply(_comparison_outcome, axis=1)
    aligned["is_agreement"] = (aligned["predicted_class"] == aligned["rule_entry_signal"]).astype("int64")
    aligned["is_disagreement"] = (1 - aligned["is_agreement"]).astype("int64")
    aligned["ml_matches_actual_target"] = (aligned["predicted_class"] == aligned["actual_target"]).astype("int64")
    aligned["rule_entry_matches_actual_target"] = (
        aligned["rule_entry_signal"] == aligned["actual_target"]
    ).astype("int64")
    aligned["probability_distance_from_threshold"] = (
        pd.to_numeric(aligned["predicted_probability"], errors="coerce") - 0.5
    ).abs().astype("Float64")

    aligned = aligned.sort_values(ALIGNED_SORT_ORDER).reset_index(drop=True)
    return aligned.loc[:, ALIGNED_COMPARISON_COLUMNS]


def _build_model_summary_rows(aligned_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, model_rows in aligned_df.groupby("model_name", sort=True):
        estimator = str(model_rows["estimator"].iloc[0])
        outcome_counts = model_rows["comparison_outcome"].value_counts()
        rule_action_counts = model_rows["rule_action"].value_counts()

        rows.append(
            {
                "model_name": model_name,
                "estimator": estimator,
                "prediction_run_id": str(model_rows["prediction_run_id"].iloc[0]),
                "training_run_id": str(model_rows["training_run_id"].iloc[0]),
                "row_count": int(len(model_rows)),
                "symbol_count": int(model_rows["symbol"].nunique()),
                "agreement_rate": float(model_rows["is_agreement"].mean()),
                "disagreement_rate": float(model_rows["is_disagreement"].mean()),
                "shared_entry_count": int(outcome_counts.get("both_entry", 0)),
                "ml_only_entry_count": int(outcome_counts.get("ml_only_entry", 0)),
                "rule_only_entry_count": int(outcome_counts.get("rule_only_entry", 0)),
                "shared_non_entry_count": int(outcome_counts.get("shared_non_entry", 0)),
                "rule_buy_count": int(rule_action_counts.get("BUY", 0)),
                "rule_sell_count": int(rule_action_counts.get("SELL", 0)),
                "rule_hold_count": int(rule_action_counts.get("HOLD", 0)),
                "model_positive_rate": float(model_rows["predicted_class"].mean()),
                "rule_buy_rate": float(model_rows["rule_entry_signal"].mean()),
                "rule_sell_rate": float((model_rows["rule_action"] == "SELL").mean()),
                "rule_hold_rate": float((model_rows["rule_action"] == "HOLD").mean()),
                "actual_positive_rate": float(model_rows["actual_target"].mean()),
                "ml_accuracy_vs_actual": float(model_rows["ml_matches_actual_target"].mean()),
                "rule_entry_accuracy_vs_actual": float(model_rows["rule_entry_matches_actual_target"].mean()),
                "actual_positive_rate_both_entry": _mean_or_none(
                    model_rows.loc[model_rows["comparison_outcome"] == "both_entry", "actual_target"]
                ),
                "actual_positive_rate_ml_only_entry": _mean_or_none(
                    model_rows.loc[model_rows["comparison_outcome"] == "ml_only_entry", "actual_target"]
                ),
                "actual_positive_rate_rule_only_entry": _mean_or_none(
                    model_rows.loc[model_rows["comparison_outcome"] == "rule_only_entry", "actual_target"]
                ),
                "actual_positive_rate_shared_non_entry": _mean_or_none(
                    model_rows.loc[model_rows["comparison_outcome"] == "shared_non_entry", "actual_target"]
                ),
                "mean_predicted_probability": _mean_or_none(model_rows["predicted_probability"]),
                "mean_predicted_probability_rule_buy": _mean_or_none(
                    model_rows.loc[model_rows["rule_action"] == "BUY", "predicted_probability"]
                ),
                "mean_predicted_probability_rule_sell": _mean_or_none(
                    model_rows.loc[model_rows["rule_action"] == "SELL", "predicted_probability"]
                ),
                "mean_predicted_probability_rule_hold": _mean_or_none(
                    model_rows.loc[model_rows["rule_action"] == "HOLD", "predicted_probability"]
                ),
            }
        )

    return rows


def _build_per_symbol_summary_rows(aligned_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model_name, symbol), symbol_rows in aligned_df.groupby(["model_name", "symbol"], sort=True):
        outcome_counts = symbol_rows["comparison_outcome"].value_counts()
        rule_action_counts = symbol_rows["rule_action"].value_counts()
        rows.append(
            {
                "model_name": model_name,
                "symbol": symbol,
                "row_count": int(len(symbol_rows)),
                "agreement_rate": float(symbol_rows["is_agreement"].mean()),
                "shared_entry_count": int(outcome_counts.get("both_entry", 0)),
                "ml_only_entry_count": int(outcome_counts.get("ml_only_entry", 0)),
                "rule_only_entry_count": int(outcome_counts.get("rule_only_entry", 0)),
                "shared_non_entry_count": int(outcome_counts.get("shared_non_entry", 0)),
                "rule_buy_count": int(rule_action_counts.get("BUY", 0)),
                "rule_sell_count": int(rule_action_counts.get("SELL", 0)),
                "rule_hold_count": int(rule_action_counts.get("HOLD", 0)),
                "actual_positive_rate": float(symbol_rows["actual_target"].mean()),
                "ml_accuracy_vs_actual": float(symbol_rows["ml_matches_actual_target"].mean()),
                "rule_entry_accuracy_vs_actual": float(symbol_rows["rule_entry_matches_actual_target"].mean()),
                "mean_predicted_probability": _mean_or_none(symbol_rows["predicted_probability"]),
            }
        )
    return rows


def _methodology_payload(
    definition: OfficialM4MLVsRuleComparisonDefinition,
) -> dict[str, Any]:
    return {
        "name": definition.methodology_name,
        "comparison_level": "decision_row_signal",
        "shared_key_columns": list(COMPARISON_KEY_COLUMNS),
        "sort_order": list(ALIGNED_SORT_ORDER),
        "ml_signal_definition": {
            "source_column": "predicted_class",
            "positive_label": 1,
            "probability_column": "predicted_probability",
            "meaning": "Predict next-session direction is positive for the same symbol row.",
        },
        "rule_signal_definition": {
            "source": "DailySimulator signal_history from the live momentum strategy replay.",
            "raw_action_column": "rule_action",
            "comparison_signal_column": definition.comparison_signal_column,
            "comparison_mapping": {"BUY": 1, "SELL": 0, "HOLD": 0},
            "raw_actions_retained": ["BUY", "SELL", "HOLD"],
        },
        "alignment_policy": {
            "prediction_rows_must_match_validation_rows_per_model": True,
            "rule_rows_must_cover_all_validation_symbol_dates": True,
            "duplicate_symbol_date_rows_fail": True,
        },
        "carryover_policy": (
            "The rule strategy is replayed on the shared feature history for the comparison symbols through "
            "the validation end date, then only validation decision rows are aligned to ML predictions."
        ),
        "time_safe_assumptions": [
            "ML rows use official M4 prediction logs keyed by symbol/date/target_date.",
            "Rule decisions are generated by the existing simulator using data available up to each decision date only.",
            "The simulator strips target_ columns before strategy execution.",
            "Rule orders continue to use next-session execution behavior during replay.",
            "The aligned comparison keeps target_date strictly after date for every row.",
        ],
    }


def run_m4_ml_vs_rule_comparison(
    *,
    predictions_path: Path,
    metadata_path: Path | None = None,
    config_path: Path = M4_ML_VS_RULE_COMPARISON_CONFIG_PATH,
    comparison_definition: OfficialM4MLVsRuleComparisonDefinition | None = None,
) -> dict[str, Any]:
    resolved_definition = comparison_definition or load_m4_ml_vs_rule_comparison_definition(config_path)
    prediction_log_definition = load_m4_prediction_log_definition()
    resolved_metadata_path = metadata_path or predictions_path.with_name(prediction_log_definition.metadata_filename)
    prediction_bundle = load_m4_prediction_log_bundle(
        dataset_path=Path(predictions_path),
        metadata_path=Path(resolved_metadata_path),
        validate=True,
    )
    prediction_df = prediction_bundle["dataframe"].copy()
    prediction_metadata = prediction_bundle["metadata"]
    if prediction_metadata.get("pipeline_name") != "m4_model_output_logging":
        raise ValueError("Prediction metadata is not an M4 model output logging artifact.")

    source_artifacts = prediction_metadata.get("source_artifacts") or {}
    training_summary_raw = str(source_artifacts.get("training_summary_path", "")).strip()
    if not training_summary_raw:
        raise ValueError("Prediction log metadata is missing source_artifacts.training_summary_path.")
    training_summary_path = _resolve_repo_path(training_summary_raw)
    training_bundle = load_m4_baseline_training_run_bundle(training_summary_path)

    if str(prediction_df["training_run_id"].iloc[0]).strip() != str(training_bundle["training_summary"].get("run_id", "")).strip():
        raise ValueError("Prediction log training_run_id does not match the referenced training summary.")

    prepared = prepare_m4_baseline_training_data(
        training_definition=training_bundle["training_definition"],
        split_definition=training_bundle["split_definition"],
        target_definition=training_bundle["target_definition"],
    )
    validation_rows = _build_validation_rows(
        prepared["validation_dataframe"],
        actual_target_column=training_bundle["training_definition"].target_column,
        actual_return_column=training_bundle["target_definition"].helper_return_column,
    )

    _validate_prediction_rows(
        prediction_df,
        validation_rows=validation_rows,
        expected_target_column=training_bundle["training_definition"].target_column,
        expected_task_type=str(training_bundle["target_definition"].task_type).strip().lower(),
    )

    output_root = _resolve_repo_path(resolved_definition.output_dir)
    manager = RunArtifactManager(
        base_output_dir=output_root,
        strategy_name=resolved_definition.strategy_name,
        start_date=_format_date(validation_rows["date"].min()) or "",
        end_date=_format_date(validation_rows["date"].max()) or "",
        run_label=resolved_definition.run_label,
        strategy_variant=resolved_definition.methodology_name,
    )

    config_snapshot = {
        "comparison_definition": asdict(resolved_definition),
        "predictions_path": str(Path(predictions_path).resolve()),
        "prediction_metadata_path": str(Path(resolved_metadata_path).resolve()),
        "training_summary_path": str(training_summary_path),
        "prediction_run_id": str(prediction_df["prediction_run_id"].iloc[0]),
        "training_run_id": str(prediction_df["training_run_id"].iloc[0]),
        "recreated_split_definition": asdict(training_bundle["split_definition"]),
        "recreated_target_definition": asdict(training_bundle["target_definition"]),
    }
    config_snapshot_path = manager.write_config_snapshot(config_snapshot)

    try:
        strategy_config_path = _resolve_repo_path(resolved_definition.strategy_config_path)
        feature_dataset_path = _resolve_repo_path(resolved_definition.feature_dataset_path)
        rule_settings = _load_rule_settings(strategy_config_path)
        comparison_symbols = (
            validation_rows["symbol"].dropna().astype(str).str.upper().sort_values().unique().tolist()
        )
        rule_feature_history = _load_rule_feature_history(
            feature_dataset_path=feature_dataset_path,
            comparison_symbols=comparison_symbols,
            validation_rows=validation_rows,
        )
        rule_replay_result = _run_rule_strategy_replay(
            feature_history=rule_feature_history,
            settings=rule_settings,
            comparison_run_id=manager.run_id,
            comparison_dir=manager.output_dir,
            config_source=strategy_config_path,
        )
        rule_rows = _build_rule_signal_rows(
            signal_history=rule_replay_result["signal_history"],
            validation_rows=validation_rows,
        )
        aligned_df = _build_aligned_comparison_table(
            prediction_df=prediction_df,
            validation_rows=validation_rows,
            rule_rows=rule_rows,
        )
        model_summary_rows = _build_model_summary_rows(aligned_df)
        per_symbol_summary_rows = _build_per_symbol_summary_rows(aligned_df)

        aligned_path = manager.artifact_path(ALIGNED_COMPARISON_FILENAME)
        aligned_df.to_parquet(aligned_path, index=False)
        manager.register_artifact("aligned_comparison", aligned_path)

        summary_df = pd.DataFrame(model_summary_rows, columns=SUMMARY_ROW_COLUMNS)
        summary_csv_path = manager.artifact_path(COMPARISON_SUMMARY_CSV_FILENAME)
        summary_df.to_csv(summary_csv_path, index=False)
        manager.register_artifact("summary_csv", summary_csv_path)

        per_symbol_df = pd.DataFrame(per_symbol_summary_rows, columns=PER_SYMBOL_SUMMARY_COLUMNS)
        per_symbol_csv_path = manager.artifact_path(PER_SYMBOL_SUMMARY_CSV_FILENAME)
        per_symbol_df.to_csv(per_symbol_csv_path, index=False)
        manager.register_artifact("per_symbol_summary_csv", per_symbol_csv_path)

        methodology = _methodology_payload(resolved_definition)
        validation_signature = build_dataframe_signature(validation_rows)
        aligned_signature = build_dataframe_signature(aligned_df)

        aligned_metadata = {
            "pipeline_name": "m4_ml_vs_rule_comparison",
            "pipeline_version": PIPELINE_VERSION,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "entrypoint": ENTRYPOINT,
            "comparison_run_id": manager.run_id,
            "comparison_definition": asdict(resolved_definition),
            "methodology": methodology,
            "prediction_source": {
                "predictions_path": str(Path(predictions_path).resolve()),
                "prediction_metadata_path": str(Path(resolved_metadata_path).resolve()),
                "prediction_run_id": str(prediction_df["prediction_run_id"].iloc[0]),
                "training_run_id": str(prediction_df["training_run_id"].iloc[0]),
                "prediction_output_signature": str(
                    (prediction_metadata.get("output_log") or {}).get("output_signature", "")
                ).strip(),
                "model_count": int(prediction_df["model_name"].nunique()),
                "models": [
                    {
                        "model_name": str(model_name),
                        "estimator": str(group["estimator"].iloc[0]),
                        "model_artifact_path": str(group["model_artifact_path"].iloc[0]),
                        "model_metadata_path": str(group["model_metadata_path"].iloc[0]),
                    }
                    for model_name, group in prediction_df.groupby("model_name", sort=True)
                ],
            },
            "training_source": {
                "training_summary_path": str(training_summary_path),
                "training_run_id": str(training_bundle["training_summary"].get("run_id", "")),
                "training_output_dir": str(training_bundle["training_run_dir"]),
                "feature_schema_path": str(training_bundle["feature_schema_path"]),
                "split_summary_path": str(training_bundle["split_summary_path"]),
                "validation_dataset_signature": validation_signature,
            },
            "rule_strategy": {
                "strategy_config_path": str(strategy_config_path),
                "feature_dataset_path": str(feature_dataset_path),
                "feature_history_row_count": int(len(rule_feature_history)),
                "feature_history_start_date": _format_date(rule_feature_history["date"].min()),
                "feature_history_end_date": _format_date(rule_feature_history["date"].max()),
                "strategy_name": str((rule_settings.get("strategy") or {}).get("name", "momentum_v0")),
                "strategy_parameters": {
                    "max_open_positions": int((rule_settings.get("portfolio") or {}).get("max_open_positions", 1)),
                    "top_k": int((rule_settings.get("strategy") or {}).get("top_k", 1)),
                    "min_score": float((rule_settings.get("strategy") or {}).get("min_score", 0.0)),
                    "min_volume_ratio": float((rule_settings.get("strategy") or {}).get("min_volume_ratio", 0.8)),
                },
                "portfolio_parameters": {
                    "initial_cash": float((rule_settings.get("portfolio") or {}).get("initial_cash", 0.0)),
                    "fractional_shares": bool((rule_settings.get("portfolio") or {}).get("fractional_shares", True)),
                },
                "execution_parameters": {
                    "commission_rate": float((rule_settings.get("execution") or {}).get("commission_rate", 0.0)),
                    "slippage_rate": float((rule_settings.get("execution") or {}).get("slippage_rate", 0.0)),
                    "execution_timing": str((rule_settings.get("execution") or {}).get("execution_timing", "")),
                },
                "benchmark_symbol": str((rule_settings.get("benchmark") or {}).get("benchmark_symbol", "")),
                "rule_replay_run_id": str(rule_replay_result["run_id"]),
                "rule_replay_output_dir": str(rule_replay_result["output_dir"]),
                "rule_replay_manifest_path": str(rule_replay_result["manifest_path"]),
            },
            "validation_dataset": {
                "row_count": int(len(validation_rows)),
                "symbol_count": int(validation_rows["symbol"].nunique()),
                "feature_date_start": _format_date(validation_rows["date"].min()),
                "feature_date_end": _format_date(validation_rows["date"].max()),
                "target_date_start": _format_date(validation_rows["target_date"].min()),
                "target_date_end": _format_date(validation_rows["target_date"].max()),
                "shared_key_columns": list(COMPARISON_KEY_COLUMNS),
                "actual_target_column": training_bundle["training_definition"].target_column,
                "actual_target_return_column": training_bundle["target_definition"].helper_return_column,
                "validation_dataset_signature": validation_signature,
            },
            "aligned_output": {
                "path": str(aligned_path),
                "row_count": int(len(aligned_df)),
                "columns": list(aligned_df.columns),
                "sort_order": list(ALIGNED_SORT_ORDER),
                "output_signature": aligned_signature,
                "summary_csv_path": str(summary_csv_path),
                "per_symbol_summary_csv_path": str(per_symbol_csv_path),
            },
            "summary_rows": model_summary_rows,
        }
        aligned_metadata_path = _write_json(
            manager.artifact_path(ALIGNED_COMPARISON_METADATA_FILENAME),
            aligned_metadata,
        )
        manager.register_artifact("aligned_comparison_metadata", aligned_metadata_path)

        summary_payload = {
            "pipeline_name": "m4_ml_vs_rule_comparison",
            "pipeline_version": PIPELINE_VERSION,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "entrypoint": ENTRYPOINT,
            "comparison_run_id": manager.run_id,
            "status": "completed",
            "config_snapshot_path": str(config_snapshot_path),
            "methodology": methodology,
            "prediction_run_id": str(prediction_df["prediction_run_id"].iloc[0]),
            "training_run_id": str(prediction_df["training_run_id"].iloc[0]),
            "validation_dataset": aligned_metadata["validation_dataset"],
            "rule_strategy": {
                "strategy_name": aligned_metadata["rule_strategy"]["strategy_name"],
                "strategy_config_path": aligned_metadata["rule_strategy"]["strategy_config_path"],
                "feature_dataset_path": aligned_metadata["rule_strategy"]["feature_dataset_path"],
                "rule_replay_run_id": aligned_metadata["rule_strategy"]["rule_replay_run_id"],
                "rule_replay_output_dir": aligned_metadata["rule_strategy"]["rule_replay_output_dir"],
                "rule_replay_manifest_path": aligned_metadata["rule_strategy"]["rule_replay_manifest_path"],
            },
            "artifacts": {
                "aligned_comparison_path": str(aligned_path),
                "aligned_comparison_metadata_path": str(aligned_metadata_path),
                "summary_csv_path": str(summary_csv_path),
                "per_symbol_summary_csv_path": str(per_symbol_csv_path),
            },
            "rows": model_summary_rows,
        }
        summary_json_path = _write_json(
            manager.artifact_path(COMPARISON_SUMMARY_JSON_FILENAME),
            summary_payload,
        )
        manager.register_artifact("summary_json", summary_json_path)
        manager.register_artifact("rule_replay_manifest", Path(rule_replay_result["manifest_path"]))

        manifest_path = manager.write_manifest(status="completed", config_source=str(config_path))
    except Exception as exc:
        manifest_path = manager.write_manifest(
            status="failed",
            config_source=str(config_path),
            error_message=str(exc),
        )
        raise

    return {
        "run_id": manager.run_id,
        "output_dir": manager.output_dir,
        "manifest_path": manifest_path,
        "config_path": config_snapshot_path,
        "aligned_path": aligned_path,
        "aligned_metadata_path": aligned_metadata_path,
        "summary_json_path": summary_json_path,
        "summary_csv_path": summary_csv_path,
        "per_symbol_summary_csv_path": per_symbol_csv_path,
        "rule_replay_run_id": rule_replay_result["run_id"],
        "rule_replay_output_dir": rule_replay_result["output_dir"],
        "rule_replay_manifest_path": rule_replay_result["manifest_path"],
        "rows": model_summary_rows,
    }
