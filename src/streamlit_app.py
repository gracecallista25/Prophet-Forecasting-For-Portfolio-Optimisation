"""Streamlit dashboard for portfolio forecasting and optimisation."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import get_supabase_client
from src.settings import SUPABASE_TABLE_NAME

st.set_page_config(page_title="Portfolio Forecast Dashboard", layout="wide")


@st.cache_data(ttl=300)
def load_supabase_predictions() -> pd.DataFrame:
    """Return latest Supabase rows (one per ticker per date)."""
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()

    response = (
        client.table(SUPABASE_TABLE_NAME)
        .select("*")
        .order("as_of_date", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    data = getattr(response, "data", None)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])

    df = df.sort_values(["as_of_date", "created_at"], ascending=[True, False])
    df = df.drop_duplicates(subset=["as_of_date", "ticker"], keep="first")

    if "actual_prices_last_month" in df.columns:
        df["actual_prices_last_month"] = df["actual_prices_last_month"].apply(_parse_price_history)

    return df


def _parse_price_history(raw: object) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(value) for value in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [float(value) for value in decoded]
    return []


def _latest_actual_price(row: pd.Series) -> float | None:
    prices = row.get("actual_prices_last_month", [])
    if prices:
        return float(prices[-1])
    return None


def build_price_history(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    prices = row.get("actual_prices_last_month", [])
    if not prices:
        return None

    as_of_date: date = row["as_of_date"]
    n = len(prices)

    actual_index = pd.bdate_range(end=pd.to_datetime(as_of_date), periods=n)
    actual_df = pd.DataFrame({"date": actual_index, "price": prices})

    prediction_date = pd.bdate_range(
        start=pd.to_datetime(as_of_date) + pd.Timedelta(days=1),
        periods=1,
    )[0]
    predicted_df = pd.DataFrame({"date": [prediction_date], "price": [row["predicted_price"]]})

    return actual_df, predicted_df


@lru_cache(maxsize=1)
def compute_prediction_performance(data_json: str) -> pd.DataFrame:
    """Compare past predictions against actual outcomes using successive days."""
    df = pd.read_json(data_json, orient="records", convert_dates=False)
    if df.empty:
        return df

    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    if "actual_prices_last_month" in df.columns:
        df["actual_prices_last_month"] = df["actual_prices_last_month"].apply(_parse_price_history)
    df = df.sort_values(["ticker", "as_of_date"])

    records: list[dict[str, object]] = []

    for ticker, group in df.groupby("ticker"):
        group = group.reset_index(drop=True)
        for idx in range(len(group) - 1):
            current = group.loc[idx]

            prices = current.get("actual_prices_last_month")
            if not prices:
                continue

            next_row = group.loc[idx + 1]
            actual_next_price = _latest_price_from_row(next_row)
            if actual_next_price is None:
                continue

            records.append(
                {
                    "ticker": ticker,
                    "prediction_date": current["as_of_date"],
                    "evaluation_date": next_row["as_of_date"],
                    "predicted_price": float(current["predicted_price"]),
                    "actual_price": actual_next_price,
                    "error": actual_next_price - float(current["predicted_price"]),
                }
            )

    perf_df = pd.DataFrame(records)
    if perf_df.empty:
        return perf_df

    perf_df["absolute_error"] = perf_df["error"].abs()
    perf_df["error_pct"] = perf_df["error"] / perf_df["predicted_price"]
    return perf_df


def _latest_price_from_row(row: pd.Series) -> float | None:
    prices = row.get("actual_prices_last_month")
    if isinstance(prices, list) and prices:
        return float(prices[-1])
    return None


def pie_chart(weights_df: pd.DataFrame):
    chart_df = weights_df[["ticker", "portfolio_weight"]].copy()
    chart_df["portfolio_weight"] = pd.to_numeric(chart_df["portfolio_weight"], errors="coerce")
    chart_df = chart_df.dropna(subset=["portfolio_weight"])

    total_weight = chart_df["portfolio_weight"].sum()
    if total_weight <= 0:
        return None

    fig = px.pie(
        chart_df,
        names="ticker",
        values="portfolio_weight",
        hole=0.3,
    )
    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value:.2f}")
    fig.update_layout(showlegend=True, legend_title_text="Ticker", height=360)
    return fig


def run_dashboard() -> None:
    st.title("📊 Portfolio Forecast Dashboard")
    st.caption("Model-selected return forecasts and portfolio optimisation results sourced from Supabase.")

    df = load_supabase_predictions()

    if df.empty:
        st.info("No prediction data available. Run the optimisation pipeline first.")
        return

    available_dates = sorted(df["as_of_date"].unique(), reverse=True)

    selected_date = st.selectbox(
        "Select as-of date",
        options=available_dates,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

    date_df = df[df["as_of_date"] == selected_date].copy().sort_values("ticker")

    horizon = 20

    if "forecast_horizon_days" in date_df.columns:
        horizon_values = date_df["forecast_horizon_days"].dropna()

        if not horizon_values.empty:
            horizon = int(horizon_values.iloc[0])

    target_date = None

    if "forecast_target_date" in date_df.columns:
        target_dates = date_df["forecast_target_date"].dropna()

        if not target_dates.empty:
            target_date = pd.to_datetime(target_dates.iloc[0]).date()

    if target_date:
        st.caption(
            f"Forecast horizon: {horizon} trading days | "
            f"Target date: {target_date.strftime('%Y-%m-%d')}"
        )
    else:
        st.caption(f"Forecast horizon: {horizon} trading days")

    st.subheader("Portfolio Allocation")

    weight_col, table_col = st.columns([1, 1])

    with weight_col:
        pie = pie_chart(date_df)

        if pie is None:
            st.info("Portfolio weights are unavailable.")
        else:
            st.plotly_chart(pie, use_container_width=True)

    with table_col:
        columns = ["ticker", "predicted_price", "predicted_return", "portfolio_weight"]

        if "selected_model" in date_df.columns:
            columns.insert(1, "selected_model")

        summary_table = date_df[columns].copy()
        summary_table["predicted_return"] *= 100
        summary_table["portfolio_weight"] *= 100

        summary_table = summary_table.rename(
            columns={
                "ticker": "Ticker",
                "selected_model": "Model",
                "predicted_price": "Predicted Price",
                "predicted_return": "Expected Return (%)",
                "portfolio_weight": "Weight (%)",
            }
        )

        st.dataframe(
            summary_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Predicted Price": st.column_config.NumberColumn(format="$%.2f"),
                "Expected Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Weight (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    tickers = date_df["ticker"].tolist()

    selected_ticker = st.selectbox(
        "Select ticker for detail view",
        options=tickers,
    )

    ticker_row = date_df.set_index("ticker").loc[selected_ticker]

    latest_actual = _latest_actual_price(ticker_row)
    selected_model = ticker_row.get("selected_model", "Unknown")
    predicted_price = float(ticker_row["predicted_price"])
    predicted_return = float(ticker_row["predicted_return"])
    portfolio_weight = float(ticker_row["portfolio_weight"])

    st.subheader(f"{selected_ticker} Forecast")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        latest_price = f"${latest_actual:.2f}" if latest_actual is not None else "—"
        st.metric("Latest Price", latest_price)

    with col2:
        st.metric(f"{horizon}-Day Predicted Price", f"${predicted_price:.2f}")

    with col3:
        st.metric(f"{horizon}-Day Expected Return", f"{predicted_return * 100:.2f}%")

    with col4:
        st.metric("Portfolio Weight", f"{portfolio_weight * 100:.2f}%")

    model_display = str(selected_model).replace("_", " ").title()
    st.write(f"**Selected model:** {model_display}")

    st.subheader("Prediction Accuracy")

    st.info(
        f"Accuracy tracking for {horizon}-trading-day forecasts will be evaluated "
        "after each forecast reaches its target date."
    )


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
