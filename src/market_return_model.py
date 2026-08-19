"""Return forecasting model with SPY market context."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    # Stock-specific features
    "return_1d",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "ma_5_ratio",
    "ma_10_ratio",
    "volatility_5",
    "volatility_20",

    # Market/SPY features
    "market_return_1d",
    "market_momentum_5",
    "market_momentum_20",
    "market_volatility_20",

    # Relative stock-vs-market behaviour
    "relative_return_1d",
    "relative_momentum_5",
    "relative_momentum_20",
]


def build_market_features(
    price_series: pd.Series,
    market_price_series: pd.Series,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Build stock and market features for return prediction.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    # Keep only trading dates available for both stock and SPY
    aligned = pd.concat(
        {
            "stock": price_series.astype(float),
            "market": market_price_series.astype(float),
        },
        axis=1,
        join="inner",
    ).dropna()

    stock = aligned["stock"]
    market = aligned["market"]

    stock_returns = stock.pct_change()
    market_returns = market.pct_change()

    features = pd.DataFrame(index=aligned.index)

    # -----------------------
    # Stock features
    # -----------------------

    features["return_1d"] = stock_returns

    features["momentum_5"] = stock.pct_change(5)
    features["momentum_10"] = stock.pct_change(10)
    features["momentum_20"] = stock.pct_change(20)

    features["ma_5_ratio"] = (
        stock / stock.rolling(5).mean() - 1
    )

    features["ma_10_ratio"] = (
        stock / stock.rolling(10).mean() - 1
    )

    features["volatility_5"] = (
        stock_returns.rolling(5).std()
    )

    features["volatility_20"] = (
        stock_returns.rolling(20).std()
    )

    # -----------------------
    # SPY market features
    # -----------------------

    features["market_return_1d"] = market_returns

    features["market_momentum_5"] = (
        market.pct_change(5)
    )

    features["market_momentum_20"] = (
        market.pct_change(20)
    )

    features["market_volatility_20"] = (
        market_returns.rolling(20).std()
    )

    # -----------------------
    # Relative-to-market features
    # -----------------------

    features["relative_return_1d"] = (
        stock_returns - market_returns
    )

    features["relative_momentum_5"] = (
        stock.pct_change(5)
        - market.pct_change(5)
    )

    features["relative_momentum_20"] = (
        stock.pct_change(20)
        - market.pct_change(20)
    )

    # Future stock return that we want to predict
    features["target_return"] = (
        stock.shift(-horizon) / stock - 1
    )

    return features


class MarketReturnRidgeModel:
    """Ridge model using stock history and SPY market context."""

    def __init__(
        self,
        alpha: float = 1.0,
        horizon: int = 20,
    ) -> None:

        if horizon < 1:
            raise ValueError(
                "horizon must be at least 1"
            )

        self.alpha = alpha
        self.horizon = horizon

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

    def fit(
        self,
        price_series: pd.Series,
        market_price_series: pd.Series,
    ) -> MarketReturnRidgeModel:
        """Train the market-aware return model."""

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

        predicted_return = self.model.predict(
            latest_features
        )

        return float(predicted_return[0])