"""Market-aware return forecasting using Elastic Net."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .market_return_model import (
    FEATURE_COLUMNS,
    build_market_features,
)


class MarketReturnElasticNetModel:
    """Elastic Net model using stock and SPY market features."""

    def __init__(
        self,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        horizon: int = 20,
    ) -> None:

        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        if not 0 <= l1_ratio <= 1:
            raise ValueError(
                "l1_ratio must be between 0 and 1"
            )

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.horizon = horizon

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "elastic_net",
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=10000,
                    ),
                ),
            ]
        )

    def fit(
        self,
        price_series: pd.Series,
        market_price_series: pd.Series,
    ) -> "MarketReturnElasticNetModel":
        """Fit Elastic Net using historical stock and SPY prices."""

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

        X = training_data[FEATURE_COLUMNS]
        y = training_data["target_return"]

        self.model.fit(X, y)

        return self

    def predict_next(
        self,
        price_series: pd.Series,
        market_price_series: pd.Series,
    ) -> float:
        """Predict future stock return."""

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

        return float(prediction[0])