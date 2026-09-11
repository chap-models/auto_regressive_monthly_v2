"""Build the configured monthly auto-regressive model."""

import logging
import os

from chap_auto_regressive import AutoRegressiveModel

logger = logging.getLogger(__name__)

# Model options a run may set via user_option_values (all optional; defaults are
# the tuned configuration baked into AutoRegressiveModel).
MODEL_OPTIONS = (
    "n_iter",
    "context_length",
    "n_ensemble",
    "learning_rate",
    "prediction_length",
    "cell",
    "rnn_features",
    "head_features",
    "rnn_layers",
    "dropout_rate",
    "recursive_decode",
    "input_dropout_rate",
)


def build_model(options: dict | None = None) -> AutoRegressiveModel:
    """Return the model, applying any run ``user_option_values``.

    Defaults are the tuned configuration baked into ``AutoRegressiveModel`` (GRU
    cells, 3-year context, 5-member ensemble). ``options`` (a run's
    ``user_option_values``) override individual knobs; unknown keys are ignored
    with a warning. ``AR_N_ITER`` still overrides the epoch count so the test
    suite can run a fast pass.
    """
    options = dict(options or {})
    if "AR_N_ITER" in os.environ:
        options["n_iter"] = int(os.environ["AR_N_ITER"])
    model = AutoRegressiveModel()
    for key, value in options.items():
        if key in MODEL_OPTIONS:
            setattr(model, key, value)
        else:
            logger.warning("Ignoring unknown model option: %s", key)
    return model
