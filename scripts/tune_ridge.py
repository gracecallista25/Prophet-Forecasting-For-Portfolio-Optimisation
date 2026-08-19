"""Tune Ridge regularisation using an earlier validation period."""

from __future__ import annotations

import pandas as pd

from src.evaluation import walk_forward_market_evaluation
from src.extractor import extract_data


TICKERS = ["AAPL", "MSFT", "GOOG"]

ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

HORIZON = 20
TEST_DAYS = 40

# Only use data available before April 2026 for tuning.
VALIDATION_END_DATE = "2026-04-01"


def main() -> None:
    """Find the best alpha using average validation MAPE."""

    data = extract_data(
        TICKERS + ["SPY"],
        start_date="2024-01-01",
        end_date=VALIDATION_END_DATE,
    )

    market_prices = data["SPY"]["Price"]

    records = []

    for alpha in ALPHAS:

        print(f"\nTesting alpha = {alpha}")

        for ticker in TICKERS:

            _, metrics = walk_forward_market_evaluation(
                data[ticker]["Price"],
                market_prices,
                test_days=TEST_DAYS,
                horizon=HORIZON,
                alpha=alpha,
            )

            records.append(
                {
                    "Alpha": alpha,
                    "Ticker": ticker,
                    "Market Ridge MAPE": metrics[
                        "Market Ridge"
                    ]["MAPE"],
                    "Ridge MAPE": metrics[
                        "Ridge"
                    ]["MAPE"],
                    "Naive MAPE": metrics[
                        "Naive"
                    ]["MAPE"],
                }
            )

    results = pd.DataFrame(records)

    print("\nIndividual results:")
    print(results.to_string(index=False))

    summary = (
        results.groupby("Alpha")[
            [
                "Market Ridge MAPE",
                "Ridge MAPE",
                "Naive MAPE",
            ]
        ]
        .mean()
        .sort_values("Market Ridge MAPE")
    )

    print("\nAverage MAPE across AAPL, MSFT and GOOG:")
    print(summary)

    best_market_alpha = summary[
        "Market Ridge MAPE"
    ].idxmin()

    best_ridge_alpha = summary[
        "Ridge MAPE"
    ].idxmin()

    print(
        "\nBest Market Ridge alpha:",
        best_market_alpha,
    )

    print(
        "Best Original Ridge alpha:",
        best_ridge_alpha,
    )


if __name__ == "__main__":
    main()