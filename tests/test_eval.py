"""End-to-end test that drives the model through the CHAP CLI on bundled data.

The test requires the ``chap`` command-line tool (provided by chap-core, which is
a dependency of this model). If it is not on PATH the test fails with a clear
message rather than being silently skipped.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "input" / "trainData.csv"


def _chap_bin() -> str:
    chap = shutil.which("chap")
    if chap is None:
        pytest.fail("The 'chap' CLI is required for these tests but was not found on PATH. Install chap-core.")
    return chap


def test_chap_eval_produces_finite_forecasts(tmp_path: Path) -> None:
    chap = _chap_bin()
    output = tmp_path / "eval.nc"
    env = {**os.environ, "AR_N_ITER": "30"}  # fast pass; production default is 1000
    cmd = [
        chap,
        "eval",
        "--model-name",
        str(REPO),
        "--dataset-csv",
        str(DATASET),
        "--output-file",
        str(output),
        "--backtest-params.n-splits",
        "2",
        "--backtest-params.n-periods",
        "1",
    ]
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1200)

    assert result.returncode == 0, f"chap eval failed:\n{result.stderr[-3000:]}"
    assert output.exists(), "chap eval did not produce an output file"

    ds = xr.open_dataset(output)
    forecast = ds["forecast"].values
    assert forecast.size > 0, "forecast is empty"
    assert np.isfinite(forecast).all(), "forecast contains non-finite values"
