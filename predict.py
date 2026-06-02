"""CHAP predict entry point.

Usage: ``python predict.py <model> <historic.csv> <future.csv> <out.csv>``.
"""

import argparse
import logging

import pandas as pd

from model import build_model

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Load a trained model and write forecast samples to a CSV."""
    parser = argparse.ArgumentParser(description="Forecast with the auto-regressive model")
    parser.add_argument("model", help="path to the trained model")
    parser.add_argument("historic_data", help="path to the historic data CSV (with disease_cases)")
    parser.add_argument("future_data", help="path to the future covariates CSV (no disease_cases)")
    parser.add_argument("out_file", help="path to write the predictions CSV")
    args = parser.parse_args()

    predictor = build_model().load_predictor(args.model)
    forecasts = predictor.predict(pd.read_csv(args.historic_data), pd.read_csv(args.future_data))
    forecasts.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    main()
