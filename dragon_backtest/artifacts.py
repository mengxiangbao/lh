from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__


def write_run_artifacts(
    out_dir: str | Path,
    cfg: dict,
    data_path: str | Path | None = None,
    data_df: pd.DataFrame | None = None,
    command: str | None = None,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = out_dir / "config_snapshot.toml"
    run_metadata_path = out_dir / "run_metadata.json"
    data_manifest_path = out_dir / "data_manifest.json"

    config_text = _to_toml(cfg)
    config_snapshot_path.write_text(config_text, encoding="utf-8")
    config_hash = _sha256_bytes(config_text.encode("utf-8"))

    data_info = _build_data_manifest(data_path=data_path, data_df=data_df)
    with data_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(data_info, f, ensure_ascii=False, indent=2, default=str)

    now = datetime.now(timezone.utc)
    run_id = f"run_{now.strftime('%Y%m%dT%H%M%SZ')}_{config_hash[:8]}"
    metadata = {
        "run_id": run_id,
        "created_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "command": command or "",
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "project_version": __version__,
        "config_file": str(config_snapshot_path.name),
        "config_hash": config_hash,
        "data_manifest_file": str(data_manifest_path.name),
        "data_hash": data_info.get("data_hash"),
        "data_rows": data_info.get("rows"),
        "date_range": {
            "start": data_info.get("start_date"),
            "end": data_info.get("end_date"),
        },
    }
    with run_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    return {
        "run_id": run_id,
        "run_metadata_file": str(run_metadata_path.name),
        "config_snapshot_file": str(config_snapshot_path.name),
        "data_manifest_file": str(data_manifest_path.name),
    }


def attach_metadata_to_metrics(metrics: dict[str, Any], artifact_refs: dict[str, str]) -> dict[str, Any]:
    out = dict(metrics)
    out["run_id"] = artifact_refs.get("run_id")
    out["metadata_file"] = artifact_refs.get("run_metadata_file")
    out["config_snapshot_file"] = artifact_refs.get("config_snapshot_file")
    out["data_manifest_file"] = artifact_refs.get("data_manifest_file")
    return out


def _build_data_manifest(data_path: str | Path | None, data_df: pd.DataFrame | None) -> dict[str, Any]:
    path = Path(data_path).resolve() if data_path else None
    rows = None
    columns: list[str] = []
    start_date = None
    end_date = None
    data_hash = None
    file_size = None
    file_mtime_utc = None

    if path and path.exists():
        file_size = path.stat().st_size
        file_mtime_utc = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data_hash = _sha256_file(path)
    if data_df is not None and not data_df.empty:
        rows = int(len(data_df))
        columns = [str(c) for c in data_df.columns]
        if "date" in data_df.columns:
            date_series = pd.to_datetime(data_df["date"], errors="coerce")
            if date_series.notna().any():
                start_date = str(date_series.min().date())
                end_date = str(date_series.max().date())
        if data_hash is None:
            data_hash = _hash_dataframe(data_df)

    return {
        "data_path": str(path) if path else "",
        "rows": rows,
        "columns": columns,
        "start_date": start_date,
        "end_date": end_date,
        "file_size_bytes": file_size,
        "file_mtime_utc": file_mtime_utc,
        "data_hash": data_hash,
    }


def _hash_dataframe(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(",".join(map(str, df.columns)).encode("utf-8"))
    hasher.update(str(len(df)).encode("utf-8"))
    sample = df.head(200).copy()
    hasher.update(sample.to_csv(index=False).encode("utf-8", errors="ignore"))
    return hasher.hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _to_toml(value: Any, prefix: str = "") -> str:
    lines: list[str] = []
    if isinstance(value, dict):
        scalars = {k: v for k, v in value.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in value.items() if isinstance(v, dict)}
        for k, v in scalars.items():
            lines.append(f"{k} = {_toml_value(v)}")
        for k, v in nested.items():
            section = f"{prefix}.{k}" if prefix else k
            if lines:
                lines.append("")
            lines.append(f"[{section}]")
            lines.append(_to_toml(v, prefix=section))
        return "\n".join(lines).strip()
    return _toml_value(value)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    text = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{text}\""
