"""Build the configured monthly auto-regressive model."""

import os

import pandas as pd
from chap_auto_regressive import AutoRegressiveModel
from chap_auto_regressive.transforms import REQUIRED_COVARIATES

# Index/target/identifier columns that are never covariates.
_NON_COVARIATE_COLUMNS = frozenset({"time_period", "location", "disease_cases", "parent"})


def additional_covariates(data: pd.DataFrame) -> list[str]:
    """Return the additional covariate columns present in a training frame.

    Beyond the index, target and required covariates, CHAP includes the
    ``additional_continuous_covariates`` named in the run config — every such
    column is fed to the network as an extra feature, in column order. Only
    numeric columns qualify, which skips the string ``parent``/``location``
    identifiers and the unnamed index CHAP writes into the CSV.
    """
    skip = _NON_COVARIATE_COLUMNS | set(REQUIRED_COVARIATES)
    return [
        c
        for c in data.columns
        if c not in skip and not str(c).startswith("Unnamed") and pd.api.types.is_numeric_dtype(data[c])
    ]


def build_model() -> AutoRegressiveModel:
    """Return the configured monthly model.

    ``AR_N_ITER`` overrides the training-iteration count (default 1000) so the
    test suite can run a fast pass.
    """
    model = AutoRegressiveModel()
    model.n_iter = int(os.environ.get("AR_N_ITER", "1000"))
    model.context_length = 12
    model.prediction_length = 3
    model.learning_rate = 1e-5
    return model
