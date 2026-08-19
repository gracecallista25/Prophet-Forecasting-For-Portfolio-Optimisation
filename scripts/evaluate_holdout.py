"""Evaluate tuned models on an unseen historical holdout period."""

from __future__ import annotations

import pandas as pd

from src.evaluation import walk_forward_market_evaluation
from src.extractor import extract_data


TICKERS = ["AAPL", "MSFT", "GOOG"]

ALPHA = 100.0
HORIZON = 20
TEST_DAYS = 40

# Tuning stopped before April 2026.
# This dataset ends before July, giving us a later unseen period.
HOLDOUT_END_DATE = "2026-07-01"


def main() -> None:

    data = extract_data(
        TICKERS + ["SPY"],
        start_date="2024-01-01",
        end_date=HOLDOUT_END_DATE,
    )

    market_prices = data["SPY"]["Price"]

    records = []

    for ticker in TICKERS:

        _, metrics = walk_forward_market_evaluation(
            data[ticker]["Price"],
            market_prices,
            test_days=TEST_DAYS,
            horizon=HORIZON,
            alpha=ALPHA,
        )

        records.append(
            {
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
                "Market Direction": metrics[
                    "Market Ridge"
                ]["Direction Accuracy"],
                "Ridge Direction": metrics[
                    "Ridge"
                ]["Direction Accuracy"],
            }
        )

    results = pd.DataFrame(records)

    print("\nHoldout results:")
    print(results.to_string(index=False))

    print("\nAverage:")
    print(
        results[
            [
                "Market Ridge MAPE",
                "Ridge MAPE",
                "Naive MAPE",
            ]
        ].mean()
    )


if __name__ == "__main__":
    main()