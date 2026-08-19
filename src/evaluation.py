"""Utilities for evaluating Prophet stock-price forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import ProphetModel
from .return_model import ReturnRidgeModel
from .market_return_model import MarketReturnRidgeModel


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


def walk_forward_return_evaluation(
    price_series: pd.Series,
    test_days: int = 20,
    horizon: int = 1,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Evaluate the Ridge return model using walk-forward validation.

    Args:
        price_series: Historical stock prices.
        test_days: Number of forecast origins to evaluate.
        horizon: Number of trading days ahead to predict.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    if len(price_series) <= test_days + horizon:
        raise ValueError(
            "Not enough observations for the requested test period."
        )

    records = []

    # Gives exactly test_days forecasts
    start_index = len(price_series) - test_days - horizon
    end_index = len(price_series) - horizon

    for i in range(start_index, end_index):

        # Data available at the moment the forecast is made
        train = price_series.iloc[: i + 1]

        current_date = pd.Timestamp(
            price_series.index[i]
        )

        forecast_date = pd.Timestamp(
            price_series.index[i + horizon]
        )

        current_price = float(
            price_series.iloc[i]
        )

        actual_price = float(
            price_series.iloc[i + horizon]
        )

        # Train model for the requested horizon
        model = ReturnRidgeModel(
            horizon=horizon
        )

        predicted_return = model.predict_next(
            train
        )

        predicted_price = current_price * (
            1 + predicted_return
        )

        # Persistence baseline:
        # future price = today's price
        naive_prediction = current_price

        actual_return = (
            actual_price / current_price - 1
        )

        records.append(
            {
                "Origin Date": current_date,
                "Forecast Date": forecast_date,
                "Current": current_price,
                "Actual": actual_price,
                "Actual Return": actual_return,
                "Ridge": predicted_price,
                "Predicted Return": predicted_return,
                "Naive": naive_prediction,
            }
        )

    results = pd.DataFrame(records)

    actual = results["Actual"].to_numpy()
    current = results["Current"].to_numpy()

    ridge_metrics = _calculate_metrics(
        actual,
        results["Ridge"].to_numpy(),
        current,
    )

    naive_metrics = _calculate_metrics(
        actual,
        results["Naive"].to_numpy(),
        current,
    )

    metrics = {
        "Ridge": ridge_metrics,
        "Naive": naive_metrics,
    }

    return results, metrics


def walk_forward_market_evaluation(
    price_series: pd.Series,
    market_price_series: pd.Series,
    test_days: int = 20,
    horizon: int = 20,
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Compare market-aware Ridge, original Ridge, and naive baseline.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    # Make sure stock and SPY use exactly the same trading dates
    aligned = pd.concat(
        {
            "stock": price_series.astype(float),
            "market": market_price_series.astype(float),
        },
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) <= test_days + horizon:
        raise ValueError(
            "Not enough observations for the requested evaluation."
        )

    records = []

    start_index = len(aligned) - test_days - horizon
    end_index = len(aligned) - horizon

    for i in range(start_index, end_index):

        stock_train = aligned["stock"].iloc[: i + 1]
        market_train = aligned["market"].iloc[: i + 1]

        origin_date = pd.Timestamp(
            aligned.index[i]
        )

        forecast_date = pd.Timestamp(
            aligned.index[i + horizon]
        )

        current_price = float(
            aligned["stock"].iloc[i]
        )

        actual_price = float(
            aligned["stock"].iloc[i + horizon]
        )

        # -------------------------
        # Market-aware Ridge
        # -------------------------

        market_model = MarketReturnRidgeModel(
            horizon=horizon,
            alpha=alpha,
        )

        market_return_prediction = (
            market_model.predict_next(
                stock_train,
                market_train,
            )
        )

        market_price_prediction = (
            current_price
            * (1 + market_return_prediction)
        )

        # -------------------------
        # Original Ridge
        # -------------------------

        ridge_model = ReturnRidgeModel(
            horizon=horizon,
            alpha=alpha,
        )

        ridge_return_prediction = (
            ridge_model.predict_next(
                stock_train
            )
        )

        ridge_price_prediction = (
            current_price
            * (1 + ridge_return_prediction)
        )

        # -------------------------
        # Naive baseline
        # -------------------------

        naive_prediction = current_price

        records.append(
            {
                "Origin Date": origin_date,
                "Forecast Date": forecast_date,
                "Current": current_price,
                "Actual": actual_price,
                "Market Ridge": market_price_prediction,
                "Market Return": market_return_prediction,
                "Ridge": ridge_price_prediction,
                "Ridge Return": ridge_return_prediction,
                "Naive": naive_prediction,
            }
        )

    results = pd.DataFrame(records)

    actual = results["Actual"].to_numpy()
    current = results["Current"].to_numpy()

    market_metrics = _calculate_metrics(
        actual,
        results["Market Ridge"].to_numpy(),
        current,
    )

    ridge_metrics = _calculate_metrics(
        actual,
        results["Ridge"].to_numpy(),
        current,
    )

    naive_metrics = _calculate_metrics(
        actual,
        results["Naive"].to_numpy(),
        current,
    )

    metrics = {
        "Market Ridge": market_metrics,
        "Ridge": ridge_metrics,
        "Naive": naive_metrics,
    }

    return results, metrics