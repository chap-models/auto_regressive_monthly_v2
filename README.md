# auto_regressive_monthly

Monthly Deep Auto Regressive model for CHAP. An experimental deep learning model
based on an RNN architecture that forecasts disease cases from auto-regressive
time series data and climate covariates.

It wraps `AutoRegressiveModel` from
[`chap_auto_regressive`](https://github.com/chap-models/chap_auto_regressive) and exposes the standard CHAP
`train` / `predict` commands via small `train.py` / `predict.py` scripts that read
and write CSV with pandas. The model has no chap-core dependency at runtime.

## Configuration

| setting | value |
| --- | --- |
| period type | month |
| context length | 12 |
| prediction length | 3 |
| training iterations | 1000 |
| learning rate | 1e-5 |
| required covariates | rainfall, mean_temperature, population |
| additional covariates | any `additional_continuous_covariates` from the run config |

## Covariates

The three required covariates (`rainfall`, `mean_temperature`, `population`) are
always used. Any covariate named in a run's `additional_continuous_covariates`
is passed through to the network as an extra feature on top of those three —
`train.py` picks up every covariate column present in the training data, and the
chosen set is stored in the saved model so `predict.py` needs no matching config.

## The `full_rich` configuration

[`configs/full_rich.yaml`](configs/full_rich.yaml) is the tuned configuration that beats
the published MSTL champion on the Rwanda disease-case dataset on **every** evaluation
window (log-CRPS, `chap export-metrics --metric-ids crps_log1p`, n-splits 12 / n-periods 3
/ stride 1):

| window | champion (MSTL) | `full_rich` | gain |
| --- | --- | --- | --- |
| full dataset (2013–2026) | 0.3254 | **0.3184** | −2.1 % |
| active-spray (truncated to last spray) | 0.3596 | **0.3330** | −7.4 % |

It feeds 22 covariates — the required three plus climate, the resistance-weighted
"failed-protection" feature (`failed_total`), the champion IRS timing features, riskmap
statics, and geojson-derived spatial features (centroids + neighbour-mean climate) — and a
5-member deep ensemble. The many-covariate set only wins **with regularization**
(`dropout_rate: 0.3`, `input_dropout_rate: 0.25` feature-dropout, early stop `n_iter: 300`);
unregularized it overfits this small, noisy data. The feature CSV is built by the scripts in
[`scripts/data_prep/`](scripts/data_prep/).

```bash
chap evaluate --model-configuration-yaml configs/full_rich.yaml ...
```

## Environment

This model uses [uv](https://docs.astral.sh/uv/) and Python 3.13. The pinned
environment lives in `pyproject.toml` / `uv.lock`. CHAP runs the model through
its uv runner (`uv run python train.py …` / `predict.py …`); the committed lock file makes
environment creation deterministic and fast.

Key pins:

- Python 3.13
- `chap_auto_regressive` @ git (chap-models/chap_auto_regressive) — the deep AR flax model, providing `AutoRegressiveModel`
- `flax 0.12`, `jax 0.10` (resolved transitively via `chap_auto_regressive`)

The number of training iterations defaults to **1000**. Set the `AR_N_ITER`
environment variable to override it — CHAP passes it through to the model process,
so for example the test suite runs with `AR_N_ITER=30` to make a full `chap eval`
finish in a couple of minutes. Lower it for quick checks, leave it at the default
for production forecasts.

## Development

```bash
make install   # uv sync
make check     # ruff (format + lint) + mypy + pyright, no changes
make lint      # ruff format + autofix, then type-check
make test      # self-contained train/predict pipeline test
make eval      # chap eval backtest (chap CLI via uvx; no chap-core dependency)
```

This model has **no chap-core dependency**. Testing is split in two:

- `make test` is **self-contained** — it drives `train.py` / `predict.py` on the
  bundled `input/` data and verifies the prediction CSV (columns, finiteness, row
  count). No chap-core involved.
- `make eval` runs a real `chap eval` backtest, then reads back the output NetCDF
  and asserts the forecasts are finite. It gets the `chap` CLI on demand with
  `uvx --from chap-core` — chap-core is never added to the project. CI runs both.

## Local use

```bash
uv sync

# train
uv run python train.py <training_data.csv> <model_output_path>

# predict
uv run python predict.py <model_path> <historic_data.csv> <future_data.csv> <out_file.csv>
```

## Evaluating through CHAP

`--model-name` can point straight at the GitHub repo; CHAP clones and runs it.
From a chap-core checkout (for the example dataset):

```bash
uv run chap eval \
    --model-name https://github.com/chap-models/auto_regressive_monthly_v2 \
    --dataset-csv example_data/laos_subset.csv \
    --output-file /tmp/chap/ar_eval.nc \
    --backtest-params.n-splits 2 \
    --backtest-params.n-periods 1
```
