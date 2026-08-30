"""
Weekly Seasonality Dashboard
=============================
An interactive Streamlit app that pulls historical price data live via
yfinance and computes calendar-week (ISO week 1-53) seasonality statistics
for major indices, refreshing automatically on a daily cache cycle.

Run locally:
    streamlit run app.py

Author: Quant Dev (built with Claude)
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Weekly Seasonality Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
DEFAULT_TICKERS = {
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq-100 (^NDX)": "^NDX",
    "Nasdaq-100 ETF (QQQ)": "QQQ",
    "Dow Jones (^DJI)": "^DJI",
    "Russell 2000 (^RUT)": "^RUT",
    "S&P 500 ETF (SPY)": "SPY",
    "Custom ticker…": "CUSTOM",
}

LOOKBACK_OPTIONS = {
    "10 Years": 10,
    "20 Years": 20,
    "30 Years": 30,
    "Max available": None,
}

CACHE_TTL_SECONDS = 3600 * 24  # 24 hours


# ----------------------------------------------------------------------------
# DATA FETCHING (cached daily)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_price_history(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Pull daily OHLCV history for a ticker from yfinance.
    Cached for 24h so the app refreshes automatically each new session/day
    without hammering the API on every rerun.
    """
    df = yf.download(
        ticker,
        start=start_date,
        end=None,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns even for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ----------------------------------------------------------------------------
# STATISTICAL ENGINE
# ----------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def build_weekly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily closes into one row per (ISO year, ISO week) using the
    last close of each week, then compute the week-over-week % return.
    """
    if price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()
    iso = df.index.isocalendar()
    df["iso_year"] = iso["year"].values
    df["iso_week"] = iso["week"].values

    weekly = (
        df.groupby(["iso_year", "iso_week"], as_index=False)["Close"]
        .last()
        .sort_values(["iso_year", "iso_week"])
        .reset_index(drop=True)
    )

    weekly["return_pct"] = weekly["Close"].pct_change() * 100.0
    weekly = weekly.dropna(subset=["return_pct"]).reset_index(drop=True)
    return weekly


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def compute_seasonality_stats(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly returns by ISO week number (1-53) across all years
    to build the seasonality statistics table.
    """
    if weekly_df.empty:
        return pd.DataFrame()

    grouped = weekly_df.groupby("iso_week")["return_pct"]

    stats = grouped.agg(
        avg_return="mean",
        median_return="median",
        volatility="std",
        max_gain="max",
        max_drawdown="min",
        sample_size="count",
    ).reset_index()

    win_rate = grouped.apply(lambda x: (x > 0).mean() * 100.0).reset_index(name="win_rate")
    stats = stats.merge(win_rate, on="iso_week")

    stats = stats.rename(columns={"iso_week": "week"})
    stats["volatility"] = stats["volatility"].fillna(0.0)

    # simple relative risk rating based on volatility tercile
    vol_terciles = stats["volatility"].quantile([1 / 3, 2 / 3]).values
    def risk_label(v):
        if v <= vol_terciles[0]:
            return "Low"
        elif v <= vol_terciles[1]:
            return "Medium"
        return "High"
    stats["risk_rating"] = stats["volatility"].apply(risk_label)

    stats = stats.sort_values("week").reset_index(drop=True)
    return stats


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
def risk_color(rating: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(rating, "⚪")


def render_current_week_banner(stats: pd.DataFrame, current_week: int, ticker_label: str):
    row = stats[stats["week"] == current_week]
    st.subheader(f"📍 Current Calendar Week: Week {current_week} — {ticker_label}")

    if row.empty:
        st.warning("No historical sample exists for this week number yet (e.g. rare week 53).")
        return

    row = row.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Return", f"{row['avg_return']:.2f}%")
    c2.metric("Median Return", f"{row['median_return']:.2f}%")
    c3.metric("Win Rate", f"{row['win_rate']:.1f}%")
    c4.metric("Volatility (σ)", f"{row['volatility']:.2f}%")
    c5.metric("Risk Rating", f"{risk_color(row['risk_rating'])} {row['risk_rating']}")

    st.caption(
        f"Based on {int(row['sample_size'])} historical observations of week {current_week} "
        f"| Max Gain: {row['max_gain']:.2f}% | Max Drawdown: {row['max_drawdown']:.2f}%"
    )


def render_seasonality_bar_chart(stats: pd.DataFrame, current_week: int):
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in stats["avg_return"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stats["week"],
            y=stats["avg_return"],
            marker_color=colors,
            customdata=np.stack(
                [
                    stats["median_return"],
                    stats["win_rate"],
                    stats["volatility"],
                    stats["max_gain"],
                    stats["max_drawdown"],
                    stats["sample_size"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>Week %{x}</b><br>"
                "Avg Return: %{y:.2f}%<br>"
                "Median Return: %{customdata[0]:.2f}%<br>"
                "Win Rate: %{customdata[1]:.1f}%<br>"
                "Volatility: %{customdata[2]:.2f}%<br>"
                "Max Gain: %{customdata[3]:.2f}%<br>"
                "Max Drawdown: %{customdata[4]:.2f}%<br>"
                "Sample size: %{customdata[5]}yrs"
                "<extra></extra>"
            ),
            name="Avg Weekly Return",
        )
    )

    if current_week in stats["week"].values:
        fig.add_vline(
            x=current_week,
            line_width=2,
            line_dash="dash",
            line_color="#2563eb",
            annotation_text="Current Week",
            annotation_position="top",
        )

    fig.update_layout(
        title="52-Week Seasonality — Average Return by Calendar Week",
        xaxis_title="ISO Week of Year",
        yaxis_title="Average Return (%)",
        template="plotly_white",
        height=480,
        bargap=0.15,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(weekly_df: pd.DataFrame):
    pivot = weekly_df.pivot_table(
        index="iso_year", columns="iso_week", values="return_pct", aggfunc="mean"
    )
    pivot = pivot.sort_index(ascending=False)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"W{w}" for w in pivot.columns],
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            hovertemplate="Year: %{y}<br>%{x}<br>Return: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="Return %"),
        )
    )
    fig.update_layout(
        title="Multi-Year Heatmap — Weekly Returns by Year",
        xaxis_title="Week of Year",
        yaxis_title="Year",
        template="plotly_white",
        height=max(400, 22 * len(pivot.index)),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_data_table(stats: pd.DataFrame, current_week: int):
    display = stats.copy()
    display["Current"] = np.where(display["week"] == current_week, "👉", "")
    display = display.rename(
        columns={
            "week": "Week",
            "avg_return": "Avg Return %",
            "median_return": "Median Return %",
            "win_rate": "Win Rate %",
            "volatility": "Volatility %",
            "max_gain": "Max Gain %",
            "max_drawdown": "Max Drawdown %",
            "sample_size": "Years of Data",
            "risk_rating": "Risk",
        }
    )
    cols_order = [
        "Current", "Week", "Avg Return %", "Median Return %", "Win Rate %",
        "Volatility %", "Max Gain %", "Max Drawdown %", "Years of Data", "Risk",
    ]
    display = display[cols_order]

    num_cols = [c for c in display.columns if display[c].dtype != object and c != "Week"]
    st.dataframe(
        display.style.format({c: "{:.2f}" for c in num_cols}).background_gradient(
            subset=["Avg Return %"], cmap="RdYlGn"
        ),
        use_container_width=True,
        height=520,
    )


# ----------------------------------------------------------------------------
# SIDEBAR — CONTROLS
# ----------------------------------------------------------------------------
def sidebar_controls():
    st.sidebar.title("⚙️ Controls")

    label = st.sidebar.selectbox("Select Index / Ticker", list(DEFAULT_TICKERS.keys()), index=0)
    ticker = DEFAULT_TICKERS[label]
    if ticker == "CUSTOM":
        ticker = st.sidebar.text_input("Enter custom ticker symbol", value="AAPL").strip().upper()
        label = ticker

    lookback_label = st.sidebar.selectbox("Lookback Window", list(LOOKBACK_OPTIONS.keys()), index=1)
    years = LOOKBACK_OPTIONS[lookback_label]

    if years is None:
        start_date = "1900-01-01"  # yfinance will clip to actual max history
    else:
        start_date = (dt.date.today() - dt.timedelta(days=365 * years + 5)).isoformat()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data auto-refreshes once every 24h via caching. "
        "Use the button below to force an immediate refresh."
    )
    if st.sidebar.button("🔄 Force refresh data now"):
        st.cache_data.clear()
        st.rerun()

    return ticker, label, start_date


# ----------------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------------
def main():
    st.title("📈 Weekly Seasonality Dashboard")
    st.caption(
        "Live, auto-updating historical weekly seasonality analytics — no manual CSV exports required."
    )

    ticker, label, start_date = sidebar_controls()

    with st.spinner(f"Fetching live data for {ticker}…"):
        price_df = fetch_price_history(ticker, start_date)

    if price_df.empty:
        st.error(
            f"No data returned for ticker '{ticker}'. "
            "Double-check the symbol (Yahoo Finance format, e.g. ^GSPC, QQQ, AAPL)."
        )
        st.stop()

    weekly_df = build_weekly_returns(price_df)
    stats = compute_seasonality_stats(weekly_df)

    if stats.empty:
        st.warning("Not enough data to compute seasonality statistics.")
        st.stop()

    current_week = dt.date.today().isocalendar()[1]

    first_date = price_df.index.min().date()
    last_date = price_df.index.max().date()
    n_years = weekly_df["iso_year"].nunique()
    st.info(
        f"**{label}** — history from **{first_date}** to **{last_date}** "
        f"({n_years} years, {len(weekly_df)} weekly observations). Cached for 24h."
    )

    render_current_week_banner(stats, current_week, label)
    st.markdown("---")

    render_seasonality_bar_chart(stats, current_week)
    st.markdown("---")

    render_heatmap(weekly_df)
    st.markdown("---")

    st.subheader("📋 Full Seasonality Statistics Table")
    render_data_table(stats, current_week)

    st.markdown("---")
    st.caption(
        "⚠️ For informational and educational purposes only — not financial advice. "
        "Past seasonality patterns do not guarantee future results."
    )


if __name__ == "__main__":
    main()
