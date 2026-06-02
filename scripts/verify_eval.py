"""Verify a chap eval output NetCDF: forecasts must be present and finite.

Run via ``uv run --with xarray --with netcdf4 python scripts/verify_eval.py <file.nc>``
so that xarray/netcdf4 are available without adding them to the project.
"""

import sys

import numpy as np
import xarray as xr


def main() -> None:
    """Open the eval output and assert the forecasts are non-empty and finite."""
    path = sys.argv[1]
    dataset = xr.open_dataset(path)
    forecast = dataset["forecast"].values
    assert forecast.size > 0, "forecast is empty"
    assert np.isfinite(forecast).all(), "forecast contains non-finite values"
    print(f"verified: {int(forecast.size)} finite forecast values, dims {dict(dataset.sizes)}")


if __name__ == "__main__":
    main()
