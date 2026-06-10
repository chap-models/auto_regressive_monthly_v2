"""Build the configured monthly auto-regressive model."""

import os
from collections.abc import Iterable

from chap_auto_regressive import AutoRegressiveModel
from chap_auto_regressive.transforms import REQUIRED_COVARIATES

# Index and target columns that are never covariates.
_NON_COVARIATE_COLUMNS = frozenset({"time_period", "location", "disease_cases"})


def additional_covariates(columns: Iterable[str]) -> list[str]:
    """Return the additional covariate columns present in a training frame.

    CHAP writes the training CSV with exactly the index columns, the target, the
    required covariates and any ``additional_continuous_covariates`` named in the
    model configuration — so every remaining column is an additional covariate to
    feed the network, in their given order.
    """
    skip = _NON_COVARIATE_COLUMNS | set(REQUIRED_COVARIATES)
    return [c for c in columns if c not in skip]


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
