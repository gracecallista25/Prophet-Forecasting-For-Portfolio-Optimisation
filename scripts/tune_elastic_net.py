"""Tune Elastic Net using historical validation data."""

from __future__ import annotations

import pandas as pd

from src.evaluation import walk_forward_elastic_net_evaluation
from src.extractor import extract_data


TICKERS = ["AAPL", "MSFT", "GOOG"]

ALPHAS = [
    0.0001,
    0.001,
    0.01,
    0.1,
]

L1_RATIOS = [
    0.1,
    0.5,
    0.9,
]

HORIZON = 20
TEST_DAYS = 40

VALIDATION_END_DATE = "2026-04-01"


def main() -> None:

    data = extract_data(
        TICKERS + ["SPY"],
        start_date="2024-01-01",
        end_date=VALIDATION_END_DATE,
    )

    market_prices = data["SPY"]["Price"]

    records = []

    for alpha in ALPHAS:
        for l1_ratio in L1_RATIOS:

            print(
                f"\nTesting alpha={alpha}, "
                f"l1_ratio={l1_ratio}"
            )

            for ticker in TICKERS:

                _, metrics = (
                    walk_forward_elastic_net_evaluation(
                        data[ticker]["Price"],
                        market_prices,
                        test_days=TEST_DAYS,
                        horizon=HORIZON,
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                    )
                )

                records.append(
                    {
                        "Alpha": alpha,
                        "L1 Ratio": l1_ratio,
                        "Ticker": ticker,
                        "Elastic Net MAPE": metrics[
                            "Elastic Net"
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
        results.groupby(
            ["Alpha", "L1 Ratio"]
        )[
            [
                "Elastic Net MAPE",
                "Naive MAPE",
            ]
        ]
        .mean()
        .sort_values(
            "Elastic Net MAPE"
        )
    )

    print("\nAverage MAPE:")
    print(summary)

    best_params = (
        summary["Elastic Net MAPE"]
        .idxmin()
    )

    print(
        "\nBest Elastic Net parameters:"
    )

    print(
        "Alpha:",
        best_params[0],
    )

    print(
        "L1 ratio:",
        best_params[1],
    )

    print(
        "Best average MAPE:",
        summary.loc[
            best_params,
            "Elastic Net MAPE",
        ],
    )


if __name__ == "__main__":
    main()