"""Backfill historical forecasts for dashboard performance tracking."""

from datetime import date, timedelta

import pandas as pd

from src.database import get_supabase_client, save_results_to_supabase
from src.extractor import extract_data
from src.model_selector import DEFAULT_HORIZON, predict_with_model, select_best_model
from src.optimiser import optimize_portfolio_with_expected_returns
from src.processor import collect_recent_prices, preprocess_data
from src.settings import SUPABASE_TABLE_NAME


TICKERS = ["AAPL", "MSFT", "GOOG"]
MARKET_TICKER = "SPY"
START_DATE = "2024-01-01"

BACKFILL_DATES = [
    date(2026, 4, 30),
    date(2026, 5, 29),
    date(2026, 6, 30),
    date(2026, 7, 31),
]


def backfill_forecasts() -> None:
    """Generate historical forecasts without future-data leakage."""

    supabase = get_supabase_client()
    if supabase is None:
        raise ValueError("Supabase connection is unavailable.")

    end_date = (date.today() + timedelta(days=1)).isoformat()
    all_data = extract_data([*TICKERS, MARKET_TICKER], START_DATE, end_date)

    missing = [ticker for ticker in [*TICKERS, MARKET_TICKER] if ticker not in all_data]
    if missing:
        raise ValueError(f"Missing data for: {missing}")

    full_data = preprocess_data(all_data)
    trading_dates = pd.Index(full_data[MARKET_TICKER].index)

    existing_response = (
        supabase.table(SUPABASE_TABLE_NAME)
        .select("as_of_date")
        .execute()
    )

    existing_dates = {
        date.fromisoformat(row["as_of_date"])
        for row in existing_response.data
        if row.get("as_of_date")
    }

    for as_of_date in BACKFILL_DATES:
        if as_of_date in existing_dates:
            print(f"Skipping {as_of_date}: already exists.")
            continue

        if as_of_date not in trading_dates:
            print(f"Skipping {as_of_date}: not a trading day.")
            continue

        position = trading_dates.get_loc(as_of_date)
        target_position = position + DEFAULT_HORIZON

        if target_position >= len(trading_dates):
            print(f"Skipping {as_of_date}: target date unavailable.")
            continue

        target_date = trading_dates[target_position]

        portfolio_data = {
            ticker: full_data[ticker].loc[:as_of_date]
            for ticker in TICKERS
        }

        market_prices = full_data[MARKET_TICKER].loc[:as_of_date, "Price"]

        predictions = {}
        predicted_returns = {}
        selected_models = {}

        print(f"\nBackfilling {as_of_date} → {target_date}")

        for ticker, ticker_data in portfolio_data.items():
            price_series = ticker_data["Price"]

            selection = select_best_model(
                price_series,
                market_prices,
                horizon=DEFAULT_HORIZON,
            )

            forecast = predict_with_model(
                selection.selected_model,
                price_series,
                market_prices,
                horizon=DEFAULT_HORIZON,
            )

            selected_models[ticker] = selection.selected_model
            predictions[ticker] = float(forecast["predicted_price"])
            predicted_returns[ticker] = float(forecast["predicted_return"])

            print(
                f"{ticker}: {selection.selected_model} | "
                f"{predicted_returns[ticker] * 100:.2f}%"
            )

        actual_target_prices = {
            ticker: float(full_data[ticker].at[target_date, "Price"])
            for ticker in TICKERS
        }

        weights = optimize_portfolio_with_expected_returns(
            portfolio_data,
            predicted_returns,
            horizon=DEFAULT_HORIZON,
        )

        result = {
            "date": as_of_date,
            "forecast_target_date": target_date,
            "forecast_horizon_days": DEFAULT_HORIZON,
            "selected_models": selected_models,
            "predictions": predictions,
            "predicted_returns": predicted_returns,
            "actual_target_prices": actual_target_prices,
            "actual_prices_last_month": collect_recent_prices(portfolio_data),
            "weights": weights,
        }

        save_results_to_supabase(result)
        existing_dates.add(as_of_date)

        print(f"Saved {as_of_date}.")
        print(f"Actual prices: {actual_target_prices}")
        print(f"Weights: {weights}")

    print("\nBackfill complete.")


if __name__ == "__main__":
    backfill_forecasts()