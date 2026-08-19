"""Main entry point for portfolio optimisation."""

from __future__ import annotations

import logging
import sys
from typing import Any

import pandas as pd

from src.database import save_results_to_supabase
from src.extractor import extract_data
from src.optimiser import optimize_portfolio_with_expected_returns
from src.processor import collect_recent_prices, preprocess_data
from src.settings import END_DATE, PORTFOLIO_TICKERS, START_DATE
from src.model_selector import (
    DEFAULT_HORIZON,
    predict_with_model,
    select_best_model,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_optimisation(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, Any]:
    """
    Run the full forecasting and portfolio optimisation pipeline.
    Pipeline:
    1. Extract stock + SPY market data
    2. Preprocess portfolio stock data
    3. Select the best forecasting model for each stock
    4. Generate 20-trading-day return forecasts
    5. Optimise portfolio using forecast expected returns
    6. Return predictions, model choices, and portfolio weights

    Args:
        tickers:
            List of portfolio stock ticker symbols.

        start_date:
            Start date for historical data in YYYY-MM-DD format.

        end_date:
            End date for historical data in YYYY-MM-DD format.

    Returns:
        Dictionary containing:
        - date
        - forecast_horizon_days
        - selected_models
        - validation_mape
        - predictions
        - predicted_returns
        - actual_prices_last_month
        - weights
    """

    if not tickers:
        logger.warning("No portfolio tickers provided.")
        return {}

    # 1. Extract portfolio stocks + SPY market data
    logger.info(f"Starting portfolio optimisation for tickers: {tickers}")

    market_ticker = "SPY"
    extraction_tickers = list(dict.fromkeys([*tickers, market_ticker]))

    logger.info("Extracting historical data...")
    all_stock_data = extract_data(
        extraction_tickers,
        start_date=start_date,
        end_date=end_date,
    )

    if not all_stock_data:
        logger.warning("No data extracted. Exiting optimisation.")
        return {}

    # 2. Validate required data
    if market_ticker not in all_stock_data:
        logger.warning("SPY market data unavailable. Exiting optimisation.")
        return {}

    missing_tickers = [ticker for ticker in tickers if ticker not in all_stock_data]
    if missing_tickers:
        logger.warning(f"Missing stock data for: {missing_tickers}")
        return {}

    market_prices = all_stock_data[market_ticker]["Price"]
    portfolio_stock_data = {ticker: all_stock_data[ticker] for ticker in tickers}

    # 3. Preprocess portfolio data
    logger.info("Preprocessing portfolio data...")
    portfolio_data = preprocess_data(portfolio_stock_data)

    if not portfolio_data:
        logger.warning("No portfolio data available after preprocessing.")
        return {}

    as_of_date = portfolio_data[tickers[0]].index[-1]

    predictions: dict[str, float] = {}
    predicted_returns: dict[str, float] = {}
    selected_models: dict[str, str] = {}
    validation_mape: dict[str, dict[str, float]] = {}

    # 4. Select model + forecast each stock
    logger.info(
        f"Selecting forecasting models for {DEFAULT_HORIZON}-trading-day horizon..."
    )

    for ticker in tickers:
        price_series = portfolio_data[ticker]["Price"]

        selection = select_best_model(
            price_series,
            market_prices,
            horizon=DEFAULT_HORIZON,
        )

        selected_models[ticker] = selection.selected_model
        validation_mape[ticker] = selection.validation_mape

        forecast = predict_with_model(
            selection.selected_model,
            price_series,
            market_prices,
            horizon=DEFAULT_HORIZON,
        )

        predictions[ticker] = float(forecast["predicted_price"])
        predicted_returns[ticker] = float(forecast["predicted_return"])

        logger.info(
            f"{ticker}: {selection.selected_model} selected, "
            f"predicted return = {predicted_returns[ticker] * 100:.2f}%"
        )

    # 5. Collect recent actual prices
    actual_prices_last_month = collect_recent_prices(portfolio_data)

    # 6. Portfolio optimisation
    logger.info("Calculating optimal portfolio allocation...")
    weights_dict = optimize_portfolio_with_expected_returns(
        portfolio_data,
        predicted_returns,
        horizon=DEFAULT_HORIZON,
    )

    # 7. Log results
    logger.info("Portfolio Optimisation Results")
    logger.info(f"Date: {as_of_date}")

    logger.info("\nSelected Models:")
    for ticker, model_name in selected_models.items():
        logger.info(f"  {ticker}: {model_name}")

    logger.info(f"\nPredicted Prices ({DEFAULT_HORIZON} Trading Days):")
    for ticker, price in predictions.items():
        logger.info(f"  {ticker}: ${price:.2f}")

    logger.info(f"\nPredicted {DEFAULT_HORIZON}-Day Returns:")
    for ticker, predicted_return in predicted_returns.items():
        logger.info(f"  {ticker}: {predicted_return * 100:.2f}%")

    logger.info("\nOptimal Portfolio Weights:")
    for ticker, weight in weights_dict.items():
        logger.info(f"  {ticker}: {weight * 100:.2f}%")

    # 8. Return results
    return {
        "date": as_of_date,
        "forecast_horizon_days": DEFAULT_HORIZON,
        "selected_models": selected_models,
        "validation_mape": validation_mape,
        "predictions": predictions,
        "predicted_returns": predicted_returns,
        "actual_prices_last_month": actual_prices_last_month,
        "weights": weights_dict,
    }


def main() -> None:
    """Main CLI entry point - saves results to Supabase."""
    try:
        result = run_optimisation(tickers=PORTFOLIO_TICKERS)

        if not result:
            logger.error("Optimisation returned empty result")
            sys.exit(1)

        try:
            save_results_to_supabase(result)
            print("\nResults successfully saved to Supabase database")
        except Exception as db_error:
            logger.error(f"Failed to save to Supabase: {db_error}")
            print(f"\nWarning: Failed to save to Supabase: {db_error}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during optimisation: {e}")
        print(f"Error during optimisation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
