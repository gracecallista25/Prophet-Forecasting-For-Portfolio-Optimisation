"""Return-based forecasting model using Ridge regression."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "return_1d",
    "momentum_5",
    "momentum_10",
    "ma_5_ratio",
    "ma_10_ratio",
    "volatility_5",
    "volatility_10",
]


def build_return_features(
    price_series: pd.Series,
) -> pd.DataFrame:
    """
    Build technical features for next-day return prediction.

    Each row only uses information available up to that date.
    """

    prices = price_series.astype(float)

    returns = prices.pct_change()

    features = pd.DataFrame(
        index=prices.index
    )

    # Today's daily return
    features["return_1d"] = returns

    # Momentum
    features["momentum_5"] = prices.pct_change(5)
    features["momentum_10"] = prices.pct_change(10)

    # Price relative to recent moving averages
    features["ma_5_ratio"] = (
        prices / prices.rolling(5).mean() - 1
    )

    features["ma_10_ratio"] = (
        prices / prices.rolling(10).mean() - 1
    )

    # Recent volatility
    features["volatility_5"] = (
        returns.rolling(5).std()
    )

    features["volatility_10"] = (
        returns.rolling(10).std()
    )

    # What we want to predict:
    # next trading day's return
    features["target_return"] = returns.shift(-1)

    return features


class ReturnRidgeModel:
    """Ridge regression model for next-day stock returns."""

    def __init__(
        self,
        alpha: float = 1.0,
    ) -> None:
        self.alpha = alpha

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

    def fit(
        self,
        price_series: pd.Series,
    ) -> ReturnRidgeModel:
        """Fit the model using historical prices."""

        dataset = build_return_features(
            price_series
        )

        training_data = dataset.dropna(
            subset=FEATURE_COLUMNS + ["target_return"]
        )

        if len(training_data) < 30:
            raise ValueError(
                "Not enough historical data to train return model."
            )

        X = training_data[FEATURE_COLUMNS]

        y = training_data["target_return"]

        self.model.fit(X, y)

        return self

    def predict_next(
        self,
        price_series: pd.Series,
    ) -> float:
        """
        Predict the next trading day's return.

        Returns:
            Predicted return as a decimal.
            Example: 0.01 means +1%.
        """

        self.fit(price_series)

        dataset = build_return_features(
            price_series
        )

        available_features = dataset[
            FEATURE_COLUMNS
        ].dropna()

        if available_features.empty:
            raise ValueError(
                "Not enough data to create prediction features."
            )

        latest_features = available_features.iloc[
            [-1]
        ]

        prediction = self.model.predict(
            latest_features
        )

        return float(prediction[0])

    def predict_for_tickers(
        self,
        portfolio_data: dict[str, pd.DataFrame],
    ) -> tuple[
        dict[str, float],
        dict[str, float],
    ]:
        """
        Predict next-day prices and returns for multiple stocks.
        """

        predictions: dict[str, float] = {}

        predicted_returns: dict[str, float] = {}

        for ticker, dataframe in portfolio_data.items():

            prices = dataframe["Price"]

            current_price = float(
                prices.iloc[-1]
            )

            predicted_return = self.predict_next(
                prices
            )

            predicted_price = current_price * (
                1 + predicted_return
            )

            predictions[ticker] = predicted_price

            predicted_returns[ticker] = predicted_return

        return predictions, predicted_returns