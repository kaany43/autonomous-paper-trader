from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.features import add_basic_features
from src.data.loader import get_benchmark_symbol, load_market_data, load_yaml
from src.engine.broker import Broker
from src.engine.comparison_metrics import write_comparison_metrics
from src.engine.comparison_exports import write_aligned_equity_curves
from src.engine.metrics import compute_backtest_metrics, write_metrics_json
from src.engine.portfolio import Portfolio
from src.engine.run_artifacts import RunArtifactManager
from src.engine.simulator import (
    BENCHMARK_EQUITY_FILENAME,
    EQUAL_WEIGHT_EQUITY_FILENAME,
    BenchmarkComparator,
    DailySimulator,
    EqualWeightComparator,
)
from src.strategy.momentum import MomentumStrategy

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISONS_OUTPUT_DIR = REPO_ROOT / "outputs" / "backtests" / "comparisons"
COMPARISON_MANIFEST_FILENAME = "manifest.json"
COMPARISON_CONFIG_FILENAME = "comparison_config.json"
COMPARISON_SUMMARY_FILENAME = "comparison_summary.json"
ALIGNED_EQUITY_CURVES_FILENAME = "aligned_equity_curves.csv"
ALIGNED_DRAWDOWNS_FILENAME = "aligned_drawdowns.csv"


def _parse_date(value: str, flag_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"Invalid {flag_name} date format: {value!r}. Use YYYY-MM-DD.") from exc


def _normalize_variant_name(value: Any) -> str:
    name = str(value or "").strip().lower()
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _parse_momentum_variants(strategy_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    variants_raw = strategy_cfg.get("variants")
    base_params = {
        "top_k": strategy_cfg.get("top_k", 1),
        "min_score": strategy_cfg.get("min_score", 0.0),
        "min_volume_ratio": strategy_cfg.get("min_volume_ratio", 0.8),
    }

    if variants_raw is None:
        return [{"name": "baseline", "params": base_params}]

    if not isinstance(variants_raw, list) or not variants_raw:
        raise ValueError("strategy.variants must be a non-empty list when provided.")

    parsed: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for index, item in enumerate(variants_raw):
        if not isinstance(item, dict):
            raise ValueError(f"strategy.variants[{index}] must be a mapping.")
        normalized_name = _normalize_variant_name(item.get("name"))
        if not normalized_name:
            raise ValueError(f"strategy.variants[{index}].name is required.")
        if normalized_name in seen_names:
            raise ValueError(f"Duplicate strategy variant name: {normalized_name}")
        seen_names.add(normalized_name)

        params = dict(base_params)
        variant_params = item.get("params", {}) or {}
        if not isinstance(variant_params, dict):
            raise ValueError(f"strategy.variants[{index}].params must be a mapping.")
        params.update(variant_params)
        parsed.append({"name": normalized_name, "params": params})

    return parsed


def _load_universe_symbols(settings: dict[str, Any]) -> list[str]:
    symbols = settings.get("universe", {}).get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("No symbols found in universe config.")
    cleaned = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not cleaned:
        raise ValueError("Universe symbols are empty after cleaning.")
    return cleaned


def _build_unique_comparison_run_id(base_dir: Path, prefix: str = "m3-comparison") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{prefix}-{timestamp}"
    suffix = 0
    while (base_dir / run_id).exists():
        suffix += 1
        run_id = f"{prefix}-{timestamp}-{suffix:02d}"
    return run_id


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def _build_portfolio_curve_from_column(curve: pd.DataFrame, equity_column: str) -> pd.DataFrame:
    result = curve[["date", equity_column]].copy()
    result = result.rename(columns={equity_column: "total_equity"})
    return result

def _equity_input_for_run(run: dict[str, Any]) -> tuple[Path, str]:
    output_dir = Path(run["output_dir"])
    run_name = str(run.get("name", ""))

    if run_name == "buy_and_hold":
        return output_dir / BENCHMARK_EQUITY_FILENAME, "benchmark_equity"
    if run_name == "equal_weight":
        return output_dir / EQUAL_WEIGHT_EQUITY_FILENAME, "equal_weight_equity"
    return output_dir / "daily_portfolio_snapshots.csv", "total_equity"


def _load_equity_series(run: dict[str, Any]) -> pd.DataFrame:
    curve_path, equity_column = _equity_input_for_run(run)
    if not curve_path.exists():
        return pd.DataFrame(columns=["date", "run_name", "equity"])

    curve = pd.read_csv(curve_path)
    if "date" not in curve.columns or equity_column not in curve.columns:
        return pd.DataFrame(columns=["date", "run_name", "equity"])

    normalized = curve[["date", equity_column]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    normalized[equity_column] = pd.to_numeric(normalized[equity_column], errors="coerce")
    normalized = normalized.dropna(subset=["date", equity_column])
    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    normalized = normalized.rename(columns={equity_column: "equity"})
    normalized["run_name"] = str(run.get("name", ""))
    return normalized[["date", "run_name", "equity"]]


def _write_aligned_curves(
    *,
    comparison_dir: Path,
    run_records: list[dict[str, Any]],
) -> dict[str, Path]:
    curves = [_load_equity_series(run) for run in run_records]
    available_curves = [curve for curve in curves if not curve.empty]

    if not available_curves:
        aligned_equity_df = pd.DataFrame(columns=["date"])
        aligned_drawdown_df = pd.DataFrame(columns=["date"])
    else:
        combined = pd.concat(available_curves, ignore_index=True)
        aligned_equity_df = (
            combined.pivot(index="date", columns="run_name", values="equity")
            .sort_index()
            .ffill()
            .reset_index()
        )
        aligned_equity_df["date"] = pd.to_datetime(aligned_equity_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        drawdown_df = aligned_equity_df.copy()
        run_columns = [column for column in drawdown_df.columns if column != "date"]
        for column in run_columns:
            series = pd.to_numeric(drawdown_df[column], errors="coerce")
            running_peak = series.cummax()
            drawdown_df[column] = (series / running_peak) - 1.0
        aligned_drawdown_df = drawdown_df

    equity_path = comparison_dir / ALIGNED_EQUITY_CURVES_FILENAME
    drawdown_path = comparison_dir / ALIGNED_DRAWDOWNS_FILENAME
    aligned_equity_df.to_csv(equity_path, index=False)
    aligned_drawdown_df.to_csv(drawdown_path, index=False)
    return {"equity_path": equity_path, "drawdown_path": drawdown_path}



def _run_momentum_variant(
    *,
    settings: dict[str, Any],
    config_path: Path,
    run_label: str,
    variant_name: str,
    variant_params: dict[str, Any],
    features_df: pd.DataFrame,
    benchmark_symbol: str,
    symbols: list[str],
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    runs_dir: Path,
) -> dict[str, Any]:
    portfolio_cfg = settings.get("portfolio", {})
    execution_cfg = settings.get("execution", {})

    portfolio = Portfolio(initial_cash=float(portfolio_cfg.get("initial_cash", 0.0)))
    broker = Broker(
        commission_rate=float(execution_cfg.get("commission_rate", 0.0)),
        slippage_rate=float(execution_cfg.get("slippage_rate", 0.0)),
        fractional_shares=bool(portfolio_cfg.get("fractional_shares", True)),
    )
    strategy = MomentumStrategy(
        max_open_positions=int(portfolio_cfg.get("max_open_positions", 1)),
        top_k=int(variant_params.get("top_k", 1)),
        min_score=float(variant_params.get("min_score", 0.0)),
        min_volume_ratio=float(variant_params.get("min_volume_ratio", 0.8)),
    )

    simulator = DailySimulator(
        market_data=features_df,
        strategy=strategy,
        broker=broker,
        portfolio=portfolio,
        price_column="adj_close",
    )

    run_config = {
        "settings": settings,
        "comparison": {"entrypoint": "src.engine.comparison_runner", "run_label": run_label},
        "strategy_variant": {"name": variant_name, "params": variant_params},
    }

    import src.engine.simulator as simulator_module

    original_backtests_dir = simulator_module.BACKTEST_OUTPUTS_DIR
    simulator_module.BACKTEST_OUTPUTS_DIR = runs_dir
    try:
        result = simulator.run(
            start_date=start_ts,
            end_date=end_ts,
            benchmark_symbol=benchmark_symbol,
            equal_weight_universe=symbols,
            run_config=run_config,
            config_source=str(config_path),
            run_label=run_label,
        )
    finally:
        simulator_module.BACKTEST_OUTPUTS_DIR = original_backtests_dir

    return result


def run_m3_comparison(
    config_path: Path,
    output_root: Path | None = None,
    start_date_override: str | None = None,
    end_date_override: str | None = None,
) -> dict[str, Any]:
    settings = load_yaml(config_path)

    strategy_cfg = settings.get("strategy", {})
    benchmark_symbol = get_benchmark_symbol(settings)
    symbols = _load_universe_symbols(settings)
    variants = _parse_momentum_variants(strategy_cfg)

    baseline_cfg = settings.get("baselines", {})
    run_buy_and_hold = bool(baseline_cfg.get("buy_and_hold", True))
    run_equal_weight = bool(baseline_cfg.get("equal_weight", True))

    if run_buy_and_hold and not benchmark_symbol:
        raise ValueError("Buy-and-hold baseline requires benchmark.benchmark_symbol.")
    if run_equal_weight and not symbols:
        raise ValueError("Equal-weight baseline requires a non-empty universe symbols list.")

    start_date = start_date_override or settings.get("data", {}).get("start_date")
    end_date = end_date_override if end_date_override is not None else settings.get("data", {}).get("end_date")

    start_ts = _parse_date(str(start_date), "--start-date") if start_date else None
    end_ts = _parse_date(str(end_date), "--end-date") if end_date else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f"Invalid date range: start date {start_ts.date()} is after end date {end_ts.date()}.")

    output_base = Path(output_root) if output_root is not None else COMPARISONS_OUTPUT_DIR
    output_base.mkdir(parents=True, exist_ok=True)

    comparison_run_id = _build_unique_comparison_run_id(output_base)
    comparison_dir = output_base / comparison_run_id
    comparison_dir.mkdir(parents=True, exist_ok=False)
    runs_dir = comparison_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=False)

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    comparison_config = {
        "comparison_run_id": comparison_run_id,
        "created_at": created_at,
        "config_source": str(config_path),
        "data": {"start_date": str(start_date) if start_date else None, "end_date": str(end_date) if end_date else None},
        "benchmark_symbol": benchmark_symbol,
        "universe_symbols": symbols,
        "strategy_variants": variants,
        "baselines": {"buy_and_hold": run_buy_and_hold, "equal_weight": run_equal_weight},
    }
    comparison_config_path = _write_json(comparison_dir / COMPARISON_CONFIG_FILENAME, comparison_config)

    market_symbols = list(symbols)
    if benchmark_symbol and benchmark_symbol not in market_symbols:
        market_symbols.append(benchmark_symbol)
    market_df = load_market_data(symbols=market_symbols)
    features_df = add_basic_features(market_df)

    run_records: list[dict[str, Any]] = []

    for variant in variants:
        component_name = f"momentum_{variant['name']}"
        variant_result = _run_momentum_variant(
            settings=settings,
            config_path=config_path,
            run_label=component_name,
            variant_name=variant["name"],
            variant_params=variant["params"],
            features_df=features_df,
            benchmark_symbol=benchmark_symbol,
            symbols=symbols,
            start_ts=start_ts,
            end_ts=end_ts,
            runs_dir=runs_dir,
        )
        run_records.append(
            {
                "name": component_name,
                "strategy_type": "momentum",
                "variant_name": variant["name"],
                "status": "completed",
                "run_id": variant_result["run_id"],
                "output_dir": str(variant_result["output_dir"]),
                "manifest_path": str(variant_result["manifest_path"]),
                "config_path": str(variant_result["config_path"]),
                "metrics_path": str(variant_result["backtest_metrics_path"]),
            }
        )

    reference_snapshots = run_records[0]
    ref_output_dir = Path(reference_snapshots["output_dir"])
    portfolio_snapshots = pd.read_csv(ref_output_dir / "daily_portfolio_snapshots.csv")
    trade_history = pd.DataFrame()
    initial_capital = float(settings.get("portfolio", {}).get("initial_cash", 0.0))
    benchmark_curve_for_metrics = pd.DataFrame(columns=["date", "benchmark_equity"])
    if benchmark_symbol:
        benchmark_curve_for_metrics = BenchmarkComparator.build_benchmark_curve(
            market_data=features_df,
            portfolio_snapshots=portfolio_snapshots,
            benchmark_symbol=benchmark_symbol,
            initial_capital=initial_capital,
            price_column="adj_close",
        )

    if run_buy_and_hold:
        comparator = BenchmarkComparator.build_benchmark_curve(
            market_data=features_df,
            portfolio_snapshots=portfolio_snapshots,
            benchmark_symbol=benchmark_symbol,
            initial_capital=initial_capital,
            price_column="adj_close",
        )
        manager = RunArtifactManager(
            base_output_dir=runs_dir,
            strategy_name="BuyAndHoldBaseline",
            benchmark_symbol=benchmark_symbol,
            start_date=str(portfolio_snapshots.iloc[0]["date"]),
            end_date=str(portfolio_snapshots.iloc[-1]["date"]),
            run_label="buy_and_hold",
            strategy_variant="buy_and_hold",
        )
        config_payload = {"component": "buy_and_hold", "comparison_run_id": comparison_run_id, "benchmark_symbol": benchmark_symbol}
        config_path_snapshot = manager.write_config_snapshot(config_payload)
        curve_path = manager.artifact_path(BENCHMARK_EQUITY_FILENAME)
        comparator.to_csv(curve_path, index=False)
        manager.register_artifact("benchmark_curve", curve_path)
        metrics = compute_backtest_metrics(
            strategy_equity_curve=_build_portfolio_curve_from_column(comparator, "benchmark_equity"),
            benchmark_equity_curve=comparator,
            trade_history=trade_history,
        )
        metrics_path = manager.artifact_path("backtest_metrics.json")
        write_metrics_json(metrics, metrics_path)
        manager.register_artifact("backtest_metrics", metrics_path)
        manifest_path = manager.write_manifest(status="completed", config_source=str(config_path))
        run_records.append(
            {
                "name": "buy_and_hold",
                "strategy_type": "baseline",
                "variant_name": "buy_and_hold",
                "status": "completed",
                "run_id": manager.run_id,
                "output_dir": str(manager.output_dir),
                "manifest_path": str(manifest_path),
                "config_path": str(config_path_snapshot),
                "metrics_path": str(metrics_path),
            }
        )

    if run_equal_weight:
        comparator = EqualWeightComparator.build_equal_weight_curve(
            market_data=features_df,
            portfolio_snapshots=portfolio_snapshots,
            universe_symbols=symbols,
            initial_capital=initial_capital,
            price_column="adj_close",
        )
        manager = RunArtifactManager(
            base_output_dir=runs_dir,
            strategy_name="EqualWeightBaseline",
            benchmark_symbol=benchmark_symbol,
            start_date=str(portfolio_snapshots.iloc[0]["date"]),
            end_date=str(portfolio_snapshots.iloc[-1]["date"]),
            run_label="equal_weight",
            strategy_variant="equal_weight",
        )
        config_payload = {"component": "equal_weight", "comparison_run_id": comparison_run_id, "universe_symbols": symbols}
        config_path_snapshot = manager.write_config_snapshot(config_payload)
        curve_path = manager.artifact_path(EQUAL_WEIGHT_EQUITY_FILENAME)
        comparator.to_csv(curve_path, index=False)
        manager.register_artifact("equal_weight_curve", curve_path)
        metrics = compute_backtest_metrics(
            strategy_equity_curve=_build_portfolio_curve_from_column(comparator, "equal_weight_equity"),
            benchmark_equity_curve=benchmark_curve_for_metrics,
            trade_history=trade_history,
        )
        metrics_path = manager.artifact_path("backtest_metrics.json")
        write_metrics_json(metrics, metrics_path)
        manager.register_artifact("backtest_metrics", metrics_path)
        manifest_path = manager.write_manifest(status="completed", config_source=str(config_path))
        run_records.append(
            {
                "name": "equal_weight",
                "strategy_type": "baseline",
                "variant_name": "equal_weight",
                "status": "completed",
                "run_id": manager.run_id,
                "output_dir": str(manager.output_dir),
                "manifest_path": str(manifest_path),
                "config_path": str(config_path_snapshot),
                "metrics_path": str(metrics_path),
            }
        )

    run_records = sorted(run_records, key=lambda item: item["name"])
    comparison_metrics_paths = write_comparison_metrics(
        comparison_dir=comparison_dir,
        comparison_run_id=comparison_run_id,
        created_at=created_at,
        runs=run_records,
    )
    aligned_curve_paths = _write_aligned_curves(comparison_dir=comparison_dir, run_records=run_records)

    summary_payload = {
        "comparison_export_version": 1,
        "comparison_run_id": comparison_run_id,
        "created_at": created_at,
        "status": "completed",
        "comparison_config_path": str(comparison_config_path),
        "config_source": str(config_path),
        "deterministic_runs": [
            {
                "run_name": item["name"],
                "run_id": item["run_id"],
                "strategy_type": item["strategy_type"],
                "variant_name": item["variant_name"],
                "output_dir": item["output_dir"],
            }
            for item in run_records
        ],
        "artifacts": {
            "comparison_metrics_csv_path": str(comparison_metrics_paths["csv_path"]),
            "aligned_equity_curves_csv_path": str(aligned_curve_paths["equity_path"]),
            "aligned_drawdowns_csv_path": str(aligned_curve_paths["drawdown_path"]),
            "comparison_metrics_json_path": str(comparison_metrics_paths["json_path"]),
        },
    }
    comparison_summary_path = _write_json(comparison_dir / COMPARISON_SUMMARY_FILENAME, summary_payload)

    aligned_equity_csv_path = Path(
        comparison_metrics_paths.get("aligned_equity_csv_path", comparison_dir / "aligned_equity.csv")
    )
    aligned_drawdowns_csv_path = Path(
        comparison_metrics_paths.get("aligned_drawdowns_csv_path", comparison_dir / "aligned_drawdowns.csv")
    )

    manifest_payload = {
        "comparison_run_id": comparison_run_id,
        "created_at": created_at,
        "status": "completed",
        "config_snapshot_path": str(comparison_config_path),
        "comparison_metrics_json_path": str(comparison_metrics_paths["json_path"]),
        "comparison_metrics_csv_path": str(comparison_metrics_paths["csv_path"]),
        "aligned_equity_curves_csv_path": str(comparison_metrics_paths["aligned_equity_curves_csv_path"]),
        "aligned_equity_csv_path": str(comparison_metrics_paths["aligned_equity_curves_csv_path"]),
        "aligned_drawdowns_csv_path": str(comparison_metrics_paths["aligned_drawdowns_csv_path"]),
        "comparison_summary_json_path": str(comparison_metrics_paths["comparison_summary_json_path"]),
        "comparison_summary_path": str(comparison_metrics_paths["comparison_summary_json_path"]),
        "runs": run_records,
        "compared_strategies": [item["name"] for item in run_records],
    }
    manifest_path = _write_json(comparison_dir / COMPARISON_MANIFEST_FILENAME, manifest_payload)

    return {
        "comparison_run_id": comparison_run_id,
        "created_at": created_at,
        "output_dir": comparison_dir,
        "runs_dir": runs_dir,
        "comparison_manifest_path": manifest_path,
        "comparison_config_path": comparison_config_path,
        "comparison_metrics_json_path": comparison_metrics_paths["json_path"],
        "comparison_metrics_csv_path": comparison_metrics_paths["csv_path"],
        "aligned_equity_curves_csv_path": aligned_curve_paths["equity_path"],
        "aligned_drawdowns_csv_path": aligned_curve_paths["drawdown_path"],
        "comparison_summary_path": comparison_summary_path,
        "aligned_equity_csv_path": aligned_equity_csv_path,
        "aligned_equity_curves_csv_path": comparison_metrics_paths["aligned_equity_curves_csv_path"],
        "aligned_drawdowns_csv_path": comparison_metrics_paths["aligned_drawdowns_csv_path"],
        "comparison_summary_json_path": comparison_metrics_paths["comparison_summary_json_path"],
        "runs": run_records,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M3 strategy and baseline comparison workflow.")
    parser.add_argument("--config", required=True, help="Path to settings YAML file.")
    parser.add_argument("--start-date", help="Optional override start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Optional override end date (YYYY-MM-DD).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        result = run_m3_comparison(
            config_path=config_path,
            start_date_override=args.start_date,
            end_date_override=args.end_date,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[ERROR] Comparison failed unexpectedly: {exc}", file=sys.stderr)
        return 1

    print("-" * 60)
    print("M3 comparison completed")
    print(f"Comparison run id: {result['comparison_run_id']}")
    print(f"Output directory: {result['output_dir']}")
    print("Runs:")
    for run in result["runs"]:
        print(f"  - {run['name']}: {run['output_dir']}")
    print(f"Manifest: {result['comparison_manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
