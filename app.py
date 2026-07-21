"""
Investment Return-Expectation Engine — Streamlit web app.

Shows historical rolling-window return statistics for stocks/ETFs and crypto,
for whatever amount and holding period the user picks. Single data source
(Yahoo Finance) for both asset classes to keep it simple and reliable —
crypto tickers use the BTC-USD / ETH-USD / SOL-USD format.

Deploy for free on Streamlit Community Cloud (share.streamlit.io) — see
DEPLOY.md for step-by-step instructions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Backtest math
# ---------------------------------------------------------------------------

@dataclass
class ReturnStats:
    asset: str
    holding_days: int
    num_windows: int
    median_return_pct: float
    mean_return_pct: float
    win_rate_pct: float
    best_case_pct: float
    worst_case_pct: float
    volatility_pct: float
    risk_adjusted: float
    median_value: float
    best_value: float
    worst_value: float


def calendar_to_trading_days(calendar_days: int, is_crypto: bool) -> int:
    if is_crypto:
        return calendar_days
    return max(1, round(calendar_days * (252 / 365)))


def compute_stats(price_series: pd.Series, holding_days: int, amount: float) -> ReturnStats:
    closes = price_series.dropna()
    start_prices = closes.iloc[:-holding_days].values
    end_prices = closes.iloc[holding_days:].values
    window_returns_pct = (end_prices - start_prices) / start_prices * 100

    median = float(np.median(window_returns_pct))
    vol = float(np.std(window_returns_pct))
    risk_adj = median / vol if vol > 0 else 0.0

    return ReturnStats(
        asset="",
        holding_days=holding_days,
        num_windows=len(window_returns_pct),
        median_return_pct=median,
        mean_return_pct=float(np.mean(window_returns_pct)),
        win_rate_pct=float((window_returns_pct > 0).mean() * 100),
        best_case_pct=float(np.percentile(window_returns_pct, 95)),
        worst_case_pct=float(np.percentile(window_returns_pct, 5)),
        volatility_pct=vol,
        risk_adjusted=risk_adj,
        median_value=round(amount * (1 + median / 100), 2),
        best_value=round(amount * (1 + float(np.percentile(window_returns_pct, 95)) / 100), 2),
        worst_value=round(amount * (1 + float(np.percentile(window_returns_pct, 5)) / 100), 2),
    )


# ---------------------------------------------------------------------------
# Data fetching — single source (Yahoo Finance), with explicit failure modes
# ---------------------------------------------------------------------------

def is_crypto_ticker(ticker: str) -> bool:
    t = ticker.upper()
    return "-" in t and t.endswith(("USD", "USDT"))


def fetch_and_analyze(ticker: str, amount: float, calendar_days: int, timeout_sec: int = 15):
    """
    Returns (ReturnStats, note) on success, or (None, error_message) on failure.
    Every failure mode is caught and turned into a plain-English message —
    nothing raises up to crash the app.
    """
    import yfinance as yf

    ticker = ticker.strip().upper()
    if not ticker:
        return None, "Empty ticker."

    crypto = is_crypto_ticker(ticker)
    trading_days = calendar_to_trading_days(calendar_days, crypto)

    try:
        hist = yf.Ticker(ticker).history(period="5y", auto_adjust=True, timeout=timeout_sec)
    except Exception as e:
        return None, f"Could not reach data source for '{ticker}' (network/timeout issue). Try again shortly."

    if hist is None or hist.empty:
        return None, f"'{ticker}' not found — check the symbol (e.g. AAPL, RELIANCE.NS, BTC-USD)."

    closes = hist["Close"].dropna()

    if len(closes) <= trading_days:
        available_days = len(closes)
        return None, (
            f"'{ticker}' only has {available_days} days of history available, which isn't "
            f"enough for a {calendar_days}-day holding period. Try a shorter period."
        )

    try:
        stats = compute_stats(closes, trading_days, amount)
        stats.asset = ticker
    except Exception as e:
        return None, f"Unexpected error analyzing '{ticker}': {e}"

    note = "crypto (7-day weeks)" if crypto else "stock/ETF (trading-day weeks)"
    return stats, note


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Investment Return Expectation Engine", layout="wide")
st.title("Investment Return-Expectation Engine")
st.caption(
    "Shows **historical** rolling-window statistics for your chosen amount, holding period, "
    "and assets — not a prediction. Past performance does not guarantee future results. "
    "This is not financial advice."
)

with st.form("analysis_form"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Investment amount", min_value=0.0, value=5000.0, step=100.0)
    with col2:
        days = st.number_input("Holding period (calendar days)", min_value=0, value=30, step=1)

    tickers_raw = st.text_input(
        "Assets (comma-separated)",
        value="AAPL, MSFT, BTC-USD",
        help="Stock tickers (AAPL, RELIANCE.NS, TCS.BO) and/or crypto (BTC-USD, ETH-USD, SOL-USD)",
    )
    submitted = st.form_submit_button("Run Analysis")

if submitted:
    # ---- Input validation ----
    errors = []
    if amount <= 0:
        errors.append("Amount must be greater than 0.")
    if days <= 0:
        errors.append("Holding period must be at least 1 day.")
    if days > 3650:
        errors.append("Holding period is unreasonably large (max 3650 days / 10 years).")

    raw_tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not raw_tickers:
        errors.append("Enter at least one ticker.")

    # Deduplicate while preserving order
    seen = set()
    tickers = []
    for t in raw_tickers:
        if t not in seen:
            seen.add(t)
            tickers.append(t)
    if len(raw_tickers) != len(tickers):
        st.info(f"Removed {len(raw_tickers) - len(tickers)} duplicate ticker(s).")

    if errors:
        for e in errors:
            st.error(e)
    else:
        results = []
        failures = []
        with st.spinner(f"Fetching data and running analysis for {len(tickers)} asset(s)..."):
            for t in tickers:
                stats, note_or_error = fetch_and_analyze(t, amount, int(days))
                if stats is None:
                    failures.append((t, note_or_error))
                else:
                    results.append((stats, note_or_error))

        if failures:
            st.warning("Some assets couldn't be analyzed:")
            for t, msg in failures:
                st.write(f"- **{t}**: {msg}")

        if not results:
            st.error("No assets could be analyzed. Check your tickers and try again.")
        else:
            st.success(f"Analysis complete for {len(results)} of {len(tickers)} asset(s).")

            rows = []
            for stats, note in results:
                rows.append({
                    "Asset": stats.asset,
                    "Type": note,
                    "Windows": stats.num_windows,
                    "Median %": round(stats.median_return_pct, 2),
                    "Win Rate %": round(stats.win_rate_pct, 1),
                    "Best %": round(stats.best_case_pct, 2),
                    "Worst %": round(stats.worst_case_pct, 2),
                    "Volatility %": round(stats.volatility_pct, 2),
                    "Risk-Adjusted": round(stats.risk_adjusted, 3),
                    f"Median Value ($)": stats.median_value,
                    f"Best Case ($)": stats.best_value,
                    f"Worst Case ($)": stats.worst_value,
                })

            df = pd.DataFrame(rows).sort_values("Risk-Adjusted", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.caption(
                f"Statistics computed from historical {days}-calendar-day rolling windows "
                f"(converted to trading days for stocks/ETFs). "
                "Median/Best/Worst reflect the 50th/95th/5th percentiles of all historical "
                "windows of this length for each asset."
            )

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool shows historical statistics only. It is not a forecast, "
    "not investment advice, and does not guarantee future returns. Markets can and do behave "
    "differently than their history. Consult a qualified financial professional before investing."
)
