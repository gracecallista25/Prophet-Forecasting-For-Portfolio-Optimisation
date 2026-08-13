"""Data extraction module for fetching stock data from AKShare."""

import logging

import akshare as ak
import pandas as pd

from .settings import END_DATE, START_DATE

logger = logging.getLogger(__name__)


def _process_ticker_dataframe(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Process raw AKShare stock data.

    Args:
        df: Raw DataFrame returned by AKShare.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        Processed DataFrame with 'Price' and 'Returns' columns.
    """

    # Convert the date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Filter data to the requested date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    df = df[
        (df["date"] >= start)
        & (df["date"] < end)
    ].copy()

    # Keep only date and closing price
    df = df[["date", "close"]]

    # Rename columns to match the rest of the project
    df = df.rename(
        columns={
            "date": "Date",
            "close": "Price",
        }
    )

    # Calculate daily returns
    df["Returns"] = df["Price"].pct_change()

    # Remove the first row because it has no previous day return
    df = df.dropna()

    # Use Date as the index
    df = df.set_index("Date")

    # Match the date format expected by the original project
    df.index = df.index.date
    df.index.name = "Date"

    return df


def _extract_single_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """
    Extract historical data for a single US stock using AKShare.
    """

    try:
        logger.info(f"Downloading {ticker} data from AKShare...")

        # Download historical US stock data
        df = ak.stock_us_daily(symbol=ticker)

        if df.empty:
            logger.warning(f"No data available for ticker: {ticker}")
            return None

        df_processed = _process_ticker_dataframe(
            df,
            start_date,
            end_date,
        )

        if df_processed.empty:
            logger.warning(
                f"No data available for {ticker} "
                f"between {start_date} and {end_date}"
            )
            return None

        logger.info(
            f"Successfully downloaded {len(df_processed)} rows for {ticker}"
        )

        return df_processed

    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        return None


def extract_data(
    tickers: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """
    Extract historical stock data for multiple tickers.

    Returns:
        Dictionary mapping ticker symbols to DataFrames containing
        Price and Returns.
    """

    all_stock_data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        df_processed = _extract_single_ticker_data(
            ticker,
            start_date,
            end_date,
        )

        if df_processed is not None:
            all_stock_data[ticker] = df_processed

    return all_stock_data