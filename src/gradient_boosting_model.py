"""Market-aware return forecasting using gradient boosting."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .market_return_model import (
    FEATURE_COLUMNS,
    build_market_features,
)


class MarketGradientBoostingModel:
    """Gradient boosting model using stock and SPY market features."""

    def __init__(
        self,
        horizon: int = 20,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        max_leaf_nodes: int = 15,
        l2_regularization: float = 1.0,
    ) -> None:

        if horizon < 1:
            raise ValueError(
                "horizon must be at least 1"
            )

        self.horizon = horizon

        self.model = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=42,
        )

    def fit(
        self,
        price_series: pd.Series,
        market_price_series: pd.Series,
    ) -> "MarketGradientBoostingModel":
        """Fit the model using stock and SPY history."""

        dataset = build_market_features(
            price_series,
            market_price_series,
            horizon=self.horizon,
        )

        training_data = dataset.dropna(
            subset=FEATURE_COLUMNS
            + ["target_return"]
        )

        if len(training_data) < 50:
            raise ValueError(
                "Not enough historical data to train model."
            )

        X = training_data[
            FEATURE_COLUMNS
        ]

        y = training_data[
            "target_return"
        ]

        self.model.fit(
            X,
            y,
        )

        return self

    def predict_next(
        self,
        price_series: pd.Series,
        market_price_series: pd.Series,
    ) -> float:
        """Predict the future stock return."""

        self.fit(
            price_series,
            market_price_series,
        )

        dataset = build_market_features(
            price_series,
            market_price_series,
            horizon=self.horizon,
        )

        available_features = dataset[
            FEATURE_COLUMNS
        ].dropna()

        if available_features.empty:
            raise ValueError(
                "Not enough data to create prediction features."
            )

        latest_features = (
            available_features.iloc[[-1]]
        )

        prediction = self.model.predict(
            latest_features
        )

        return float(
            prediction[0]
        )