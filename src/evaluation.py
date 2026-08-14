"""Utilities for evaluating Prophet stock-price forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import ProphetModel


def _calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    previous: np.ndarray,
) -> dict[str, float]:
    """Calculate forecast accuracy metrics."""

    errors = predicted - actual

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(
        np.mean(errors**2)
    )

    mape = np.mean(
        np.abs(errors / actual)
    ) * 100

    actual_direction = np.sign(
        actual - previous
    )

    predicted_direction = np.sign(
        predicted - previous
    )

    direction_accuracy = np.mean(
        actual_direction == predicted_direction
    ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "Direction Accuracy": float(direction_accuracy),
    }


def walk_forward_evaluation(
    price_series: pd.Series,
    test_days: int = 10,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Evaluate Prophet using walk-forward one-day predictions.
    Prophet is compared against a naive baseline where tomorrow's predicted price equals today's price.
    """

    if len(price_series) <= test_days:
        raise ValueError(
            "Price series must contain more observations than test_days."
        )

    records = []

    start_index = len(price_series) - test_days

    for i in range(start_index, len(price_series)):
        train = price_series.iloc[:i]

        actual_date = pd.Timestamp(
            price_series.index[i]
        )

        actual_price = float(
            price_series.iloc[i]
        )

        previous_price = float(
            train.iloc[-1]
        )

        model = ProphetModel()
        model.fit(train)

        if model.model is None:
            raise RuntimeError("Prophet model was not fitted.")

        future = pd.DataFrame(
            {"ds": [actual_date]}
        )

        forecast = model.model.predict(future)

        prophet_prediction = float(
            forecast["yhat"].iloc[0]
        )

        naive_prediction = previous_price

        records.append(
            {
                "Date": actual_date,
                "Previous": previous_price,
                "Actual": actual_price,
                "Prophet": prophet_prediction,
                "Naive": naive_prediction,
            }
        )

    results = pd.DataFrame(records)

    actual = results["Actual"].to_numpy()
    previous = results["Previous"].to_numpy()

    prophet_metrics = _calculate_metrics(
        actual,
        results["Prophet"].to_numpy(),
        previous,
    )

    naive_metrics = _calculate_metrics(
        actual,
        results["Naive"].to_numpy(),
        previous,
    )

    metrics = {
        "Prophet": prophet_metrics,
        "Naive": naive_metrics,
    }

    return results, metrics