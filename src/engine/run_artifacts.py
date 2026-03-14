from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "config.json"


class RunArtifactManager:
    def __init__(
        self,
        base_output_dir: Path,
        strategy_name: str = "",
        benchmark_symbol: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> None:
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        self.strategy_name = str(strategy_name or "")
        self.benchmark_symbol = str(benchmark_symbol or "")
        self.start_date = str(start_date or "")
        self.end_date = str(end_date or "")

        self.run_id = self._build_unique_run_id()
        self.output_dir = self.base_output_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self._artifacts: dict[str, str] = {}

    def _build_unique_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = timestamp
        suffix = 0
        while (self.base_output_dir / run_id).exists():
            suffix += 1
            run_id = f"{timestamp}-{suffix:02d}"
        return run_id

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): RunArtifactManager._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [RunArtifactManager._to_jsonable(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (datetime, pd.Timestamp)):
            return pd.Timestamp(value).isoformat()
        return value

    def artifact_path(self, filename: str) -> Path:
        return self.output_dir / filename

    def register_artifact(self, name: str, artifact_path: Path) -> None:
        self._artifacts[name] = Path(artifact_path).name

    def write_config_snapshot(self, config: dict[str, Any]) -> Path:
        output_path = self.artifact_path(CONFIG_FILENAME)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(self._to_jsonable(config), fh, indent=2, sort_keys=True)
        self.register_artifact("config", output_path)
        return output_path

    def write_manifest(
        self,
        status: str,
        config_source: str = "",
        error_message: str = "",
    ) -> Path:
        output_path = self.artifact_path(MANIFEST_FILENAME)
        artifacts = dict(self._artifacts)
        artifacts["manifest"] = MANIFEST_FILENAME

        payload = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "output_dir": str(self.output_dir),
            "strategy_name": self.strategy_name,
            "benchmark_symbol": self.benchmark_symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": status,
            "config_source": str(config_source or ""),
            "artifacts": dict(sorted(artifacts.items())),
            "artifact_files": sorted(set(artifacts.values())),
        }
        if error_message:
            payload["error_message"] = str(error_message)

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        self.register_artifact("manifest", output_path)
        return output_path
