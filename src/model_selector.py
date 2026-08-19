"""Select the best forecasting model using validation performance."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .elastic_net_model import MarketReturnElasticNetModel
from .evaluation import (
    walk_forward_elastic_net_evaluation,
    walk_forward_market_evaluation,
)
from .market_return_model import MarketReturnRidgeModel


# Hyperparameters already selected using validation data
MARKET_RIDGE_ALPHA = 100.0

ELASTIC_NET_ALPHA = 0.01
ELASTIC_NET_L1_RATIO = 0.1

DEFAULT_HORIZON = 20


@dataclass(frozen=True)
class ModelSelectionResult:
    """Result of model selection."""

    selected_model: str
    validation_mape: dict[str, float]


def select_best_model(
    price_series: pd.Series,
    market_price_series: pd.Series,
    test_days: int = 40,
    horizon: int = DEFAULT_HORIZON,
) -> ModelSelectionResult:
    """
    Select the model with the lowest validation MAPE.

    Candidate models:
    - Naive persistence baseline
    - Market-aware Ridge
    - Market-aware Elastic Net
    """

    # Evaluate Market Ridge
    _, market_metrics = walk_forward_market_evaluation(
        price_series,
        market_price_series,
        test_days=test_days,
        horizon=horizon,
        alpha=MARKET_RIDGE_ALPHA,
    )

    # Evaluate Elastic Net
    _, elastic_metrics = (
        walk_forward_elastic_net_evaluation(
            price_series,
            market_price_series,
            test_days=test_days,
            horizon=horizon,
            alpha=ELASTIC_NET_ALPHA,
            l1_ratio=ELASTIC_NET_L1_RATIO,
        )
    )

    scores = {
        "naive": market_metrics["Naive"]["MAPE"],
        "market_ridge": market_metrics[
            "Market Ridge"
        ]["MAPE"],
        "elastic_net": elastic_metrics[
            "Elastic Net"
        ]["MAPE"],
    }

    selected_model = min(
        scores,
        key=lambda model_name: scores[model_name],
    )

    return ModelSelectionResult(
        selected_model=selected_model,
        validation_mape=scores,
    )


def predict_with_model(
    model_name: str,
    price_series: pd.Series,
    market_price_series: pd.Series,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, float | str]:
    """
    Generate a forecast using the selected model.

    Returns predicted return and predicted price.
    """

    current_price = float(
        price_series.iloc[-1]
    )

    if model_name == "naive":

        predicted_return = 0.0

    elif model_name == "market_ridge":

        model = MarketReturnRidgeModel(
            alpha=MARKET_RIDGE_ALPHA,
            horizon=horizon,
        )

        predicted_return = model.predict_next(
            price_series,
            market_price_series,
        )

    elif model_name == "elastic_net":

        model = MarketReturnElasticNetModel(
            alpha=ELASTIC_NET_ALPHA,
            l1_ratio=ELASTIC_NET_L1_RATIO,
            horizon=horizon,
        )

        predicted_return = model.predict_next(
            price_series,
            market_price_series,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    predicted_price = current_price * (
        1 + predicted_return
    )

    return {
        "model": model_name,
        "predicted_return": float(
            predicted_return
        ),
        "predicted_price": float(
            predicted_price
        ),
    }