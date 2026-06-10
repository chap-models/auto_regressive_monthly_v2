"""CHAP train entry point: ``python train.py <train_data.csv> <model_out>``."""

import argparse
import logging

import pandas as pd

from model import additional_covariates, build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Train the model on a CSV and save the predictor."""
    parser = argparse.ArgumentParser(description="Train the auto-regressive model")
    parser.add_argument("train_data", help="path to the training data CSV")
    parser.add_argument("model", help="path to write the trained model")
    args = parser.parse_args()

    data = pd.read_csv(args.train_data)
    model = build_model()
    # Any covariate column beyond the required three is fed to the network as an
    # additional feature. The chosen covariates are persisted in the saved
    # predictor, so predict.py needs no matching configuration.
    model.additional_covariates = additional_covariates(data)
    if model.additional_covariates:
        logger.info("Using additional covariates: %s", model.additional_covariates)
    predictor = model.train(data)
    predictor.save(args.model)


if __name__ == "__main__":
    main()
