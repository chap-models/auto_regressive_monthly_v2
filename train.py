"""CHAP train entry point: ``python train.py <train_data.csv> <model_out>``."""

import argparse
import logging

import pandas as pd

from model import build_model

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Train the model on a CSV and save the predictor."""
    parser = argparse.ArgumentParser(description="Train the auto-regressive model")
    parser.add_argument("train_data", help="path to the training data CSV")
    parser.add_argument("model", help="path to write the trained model")
    args = parser.parse_args()

    predictor = build_model().train(pd.read_csv(args.train_data))
    predictor.save(args.model)


if __name__ == "__main__":
    main()
