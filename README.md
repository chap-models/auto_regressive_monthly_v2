# auto_regressive_monthly

Monthly Deep Auto Regressive model for CHAP. An experimental deep learning model
based on an RNN architecture that forecasts disease cases from auto-regressive
time series data and climate covariates.

It wraps `ARModelTV1` from
[`chap_ar`](https://github.com/mortenoh/chap_ar) and exposes the standard CHAP
`train` / `predict` command-line interface via
`chap_core.adaptors.command_line_interface.generate_app`.

## Configuration

| setting | value |
| --- | --- |
| period type | month |
| context length | 12 |
| prediction length | 3 |
| training iterations | 1000 |
| learning rate | 1e-5 |
| required covariates | rainfall, mean_temperature, population |

## Environment

This model uses [uv](https://docs.astral.sh/uv/) and Python 3.13. The pinned
environment lives in `pyproject.toml` / `uv.lock`. CHAP runs the model through
its uv runner (`uv run python main.py ...`); the committed lock file makes
environment creation deterministic and fast.

Key pins:

- Python 3.13
- `chap-core` @ git `master` (dhis2-chap/chap-core)
- `chap_ar` @ git (mortenoh/chap_ar) — the deep AR flax model, providing `ARModelTV1`
- `flax 0.12`, `jax 0.10` (resolved transitively via `chap_ar`)

Training iterations default to 1000; set `AR_N_ITER` to run a faster pass (the
test suite uses a small value).

## Development

```bash
make install   # uv sync
make check     # ruff (format + lint) + mypy + pyright, no changes
make lint      # ruff format + autofix, then type-check
make test      # run the model through the chap CLI on bundled input data
```

`make test` requires the `chap` CLI (provided by chap-core, a dependency of this
model) and fails clearly if it is missing. It runs a small backtest via
`chap eval` against `input/trainData.csv` and asserts the forecasts are finite.

## Local use

```bash
uv sync

# train
uv run python main.py train <training_data.csv> <model_output_path>

# predict
uv run python main.py predict <model_path> <historic_data.csv> <future_data.csv> <out_file.csv>
```

## Evaluating through CHAP

From a chap-core checkout:

```bash
uv run chap eval \
    --model-name /path/to/auto_regressive_monthly \
    --dataset-csv example_data/laos_subset.csv \
    --output-file /tmp/chap/ar_eval.nc \
    --backtest-params.n-splits 2 \
    --backtest-params.n-periods 1
```
