import json
from copy import deepcopy

import pandas as pd

from dragon_backtest.artifacts import attach_metadata_to_metrics, write_run_artifacts
from dragon_backtest.config import DEFAULT_CONFIG


def test_write_run_artifacts_creates_expected_files(tmp_path):
    cfg = deepcopy(DEFAULT_CONFIG)
    data_path = tmp_path / "daily.csv"
    df = pd.DataFrame(
        [
            {"date": "2024-01-02", "code": "000001", "close": 10.0},
            {"date": "2024-01-03", "code": "000001", "close": 10.2},
        ]
    )
    df.to_csv(data_path, index=False)

    refs = write_run_artifacts(
        out_dir=tmp_path,
        cfg=cfg,
        data_path=data_path,
        data_df=df,
        command="python main.py backtest --config config/default.toml",
    )

    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "config_snapshot.toml").exists()
    assert (tmp_path / "data_manifest.json").exists()
    assert refs["run_metadata_file"] == "run_metadata.json"

    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"]
    assert metadata["config_hash"]
    assert metadata["data_manifest_file"] == "data_manifest.json"


def test_attach_metadata_to_metrics():
    metrics = {"total_return": 0.12}
    refs = {
        "run_id": "run_abc",
        "run_metadata_file": "run_metadata.json",
        "config_snapshot_file": "config_snapshot.toml",
        "data_manifest_file": "data_manifest.json",
    }
    out = attach_metadata_to_metrics(metrics, refs)
    assert out["run_id"] == "run_abc"
    assert out["metadata_file"] == "run_metadata.json"
