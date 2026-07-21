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
import plotly.graph_objects as go
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
    raw_returns_pct: list


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
        raw_returns_pct=window_returns_pct.tolist(),
    )


# ---------------------------------------------------------------------------
# Data fetching — single source (Yahoo Finance), with explicit failure modes
# ---------------------------------------------------------------------------

def is_crypto_ticker(ticker: str) -> bool:
    t = ticker.upper()
    return "-" in t and t.endswith(("USD", "USDT"))


def fetch_and_analyze(ticker: str, amount: float, calendar_days: int, timeout_sec: int = 15, retries: int = 3):
    """
    Returns (ReturnStats, note) on success, or (None, error_message) on failure.
    Every failure mode is caught and turned into a plain-English message —
    nothing raises up to crash the app.

    Retries a few times with a short delay: Yahoo Finance occasionally
    rate-limits requests coming from shared cloud IPs (like Streamlit Cloud),
    and a retry usually clears it.
    """
    import time
    import yfinance as yf

    ticker = ticker.strip().upper()
    if not ticker:
        return None, "Empty ticker."

    crypto = is_crypto_ticker(ticker)
    trading_days = calendar_to_trading_days(calendar_days, crypto)

    hist = None
    had_exception = False
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period="5y", auto_adjust=True, timeout=timeout_sec)
            had_exception = False
            if hist is not None and not hist.empty:
                break
        except Exception:
            had_exception = True
        time.sleep(1.5 * (attempt + 1))  # small backoff between attempts

    if hist is None or hist.empty:
        if had_exception:
            return None, (
                f"Could not reach data source for '{ticker}' after {retries} attempts "
                f"(network/rate-limit issue). Try again shortly."
            )
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
# Natural-language prompt parsing
# ---------------------------------------------------------------------------

# Common company/coin names -> tickers, so "apple" or "bitcoin" work in plain English
NAME_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "nvidia": "NVDA", "meta": "META", "facebook": "META",
    "tesla": "TSLA", "netflix": "NFLX", "jpmorgan": "JPM", "visa": "V",
    "spy": "SPY", "s&p 500": "SPY", "sp500": "SPY", "nasdaq": "QQQ",
    "bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD", "binance coin": "BNB-USD", "bnb": "BNB-USD",
    "dogecoin": "DOGE-USD", "doge": "DOGE-USD", "ripple": "XRP-USD", "xrp": "XRP-USD",
}


def parse_prompt(text: str):
    """
    Best-effort extraction of amount, holding-period days, and tickers from a
    free-text prompt. Returns (amount, days, tickers, warnings) — any piece it
    can't confidently find comes back as None, with a warning explaining what's
    missing, so the UI can ask the user to fill the gap rather than guessing.
    """
    import re

    warnings = []
    lower = text.lower()

    # ---- amount ----
    amount = None
    m = re.search(r'[\$₹]?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|m|million)?', lower)
    # Only treat this as the amount if it's near investment-y words, to avoid grabbing "30 days" as $30
    amt_context = re.search(
        r'(?:invest|put in|amount|with|of)\s*[\$₹]?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|m|million)?',
        lower,
    )
    candidate = amt_context or (m if "$" in text or "₹" in text else None)
    if candidate:
        num = float(candidate.group(1).replace(",", ""))
        unit = candidate.group(2)
        if unit in ("k", "thousand"):
            num *= 1_000
        elif unit in ("m", "million"):
            num *= 1_000_000
        amount = num
    else:
        warnings.append("Couldn't find an investment amount — please add one (e.g. '$5000').")

    # ---- days ----
    days = None
    d = re.search(r'(\d+)\s*(day|days|week|weeks|month|months|year|years)', lower)
    if d:
        n = int(d.group(1))
        unit = d.group(2)
        if unit.startswith("week"):
            days = n * 7
        elif unit.startswith("month"):
            days = n * 30
        elif unit.startswith("year"):
            days = n * 365
        else:
            days = n
    else:
        warnings.append("Couldn't find a holding period — please add one (e.g. '30 days' or '4 months').")

    # ---- tickers ----
    # Collect (start_index, ticker) candidates from both named companies/coins and
    # explicit caps tickers, then sort by where they appear in the original text
    # so the result order matches what the user typed.
    candidates = []
    for name, tick in NAME_TO_TICKER.items():
        m2 = re.search(r'\b' + re.escape(name) + r'\b', lower)
        if m2:
            candidates.append((m2.start(), tick))
    for tok_match in re.finditer(r'\b[A-Z]{2,10}(?:[-.][A-Z]{1,4})?\b', text):
        tok = tok_match.group()
        if tok not in ("USD", "ETF"):
            candidates.append((tok_match.start(), tok.upper()))

    candidates.sort(key=lambda x: x[0])
    tickers = []
    for _, tick in candidates:
        if tick not in tickers:
            tickers.append(tick)

    if not tickers:
        warnings.append("Couldn't find any assets — please name a ticker (e.g. AAPL) or company/coin (e.g. Bitcoin).")

    return amount, days, tickers, warnings


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Investment Return Expectation Engine", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
    .stMetric { background-color: rgba(255,255,255,0.03); border-radius: 10px; padding: 10px 14px; }
    div[data-testid="stExpander"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Investment Return-Expectation Engine")
st.caption(
    "Shows **historical** rolling-window statistics for your chosen amount, holding period, "
    "and assets — not a prediction. Past performance does not guarantee future results. "
    "This is not financial advice."
)

amount, days, tickers_raw_list = None, None, []

st.subheader("Ask in plain English")
prompt_text = st.text_input(
    "e.g. \"Invest $5000 in Apple, Microsoft and Bitcoin for 30 days\"",
    placeholder="Invest $5000 in Apple, Microsoft and Bitcoin for 30 days",
    key="prompt_box",
    label_visibility="collapsed",
)
prompt_submitted = st.button("🔍 Analyze", type="primary")

with st.expander("Or fill in the fields manually"):
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        with col1:
            m_amount = st.number_input("Investment amount", min_value=0.0, value=5000.0, step=100.0)
        with col2:
            m_days = st.number_input("Holding period (calendar days)", min_value=0, value=30, step=1)
        m_tickers_raw = st.text_input(
            "Assets (comma-separated)",
            value="AAPL, MSFT, BTC-USD",
            help="Stock tickers (AAPL, RELIANCE.NS, TCS.BO) and/or crypto (BTC-USD, ETH-USD, SOL-USD)",
        )
        manual_submitted = st.form_submit_button("Run Analysis (manual)")

submitted = False

if prompt_submitted:
    if not prompt_text or not prompt_text.strip():
        st.error("Type something first — e.g. \"Invest $5000 in Apple for 30 days\".")
    else:
        p_amount, p_days, p_tickers, p_warnings = parse_prompt(prompt_text)
        if p_warnings:
            for w in p_warnings:
                st.warning(w)
        st.info(
            f"**Understood:** Amount = {f'${p_amount:,.0f}' if p_amount else '—'} | "
            f"Holding period = {f'{p_days} days' if p_days else '—'} | "
            f"Assets = {', '.join(p_tickers) if p_tickers else '—'}\n\n"
            "If that's wrong, use the manual fields below instead."
        )
        if p_amount and p_days and p_tickers:
            amount, days, tickers_raw_list = p_amount, p_days, p_tickers
            submitted = True

if manual_submitted:
    amount, days = m_amount, m_days
    tickers_raw_list = [t.strip().upper() for t in m_tickers_raw.split(",") if t.strip()]
    submitted = True

if submitted:
    # ---- Input validation ----
    errors = []
    if amount is None or amount <= 0:
        errors.append("Amount must be greater than 0.")
    if days is None or days <= 0:
        errors.append("Holding period must be at least 1 day.")
    if days is not None and days > 3650:
        errors.append("Holding period is unreasonably large (max 3650 days / 10 years).")

    raw_tickers = tickers_raw_list
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
            with st.expander(f"⚠️ {len(failures)} asset(s) couldn't be analyzed", expanded=True):
                for t, msg in failures:
                    st.write(f"- **{t}**: {msg}")

        if not results:
            st.error("No assets could be analyzed. Check your tickers and try again.")
        else:
            ranked = sorted(results, key=lambda r: r[0].risk_adjusted, reverse=True)
            st.success(f"✅ Analysis complete for {len(results)} of {len(tickers)} asset(s), on ${amount:,.0f} over {days} days.")

            # ---- Comparison chart across assets (if more than one) ----
            if len(ranked) > 1:
                st.markdown("### How they compare")
                names = [s.asset for s, _ in ranked]
                medians = [s.median_return_pct for s, _ in ranked]
                colors = ["#2ecc71" if m >= 0 else "#e74c3c" for m in medians]

                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    x=names, y=medians, marker_color=colors,
                    text=[f"{m:+.2f}%" for m in medians], textposition="outside",
                    name="Median return",
                ))
                fig_compare.update_layout(
                    height=320, margin=dict(t=10, b=10, l=10, r=10),
                    yaxis_title="Median historical return (%)",
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                fig_compare.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig_compare, use_container_width=True)

            # ---- Per-asset cards ----
            st.markdown("### Asset breakdown")
            for stats, note in ranked:
                is_positive = stats.median_return_pct >= 0
                emoji = "🟢" if is_positive else "🔴"
                with st.container(border=True):
                    top_col1, top_col2 = st.columns([3, 1])
                    with top_col1:
                        st.markdown(f"#### {emoji} {stats.asset}  \n*{note} · {stats.num_windows} historical windows analyzed*")
                    with top_col2:
                        st.metric("Win rate", f"{stats.win_rate_pct:.0f}%")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Median return", f"{stats.median_return_pct:+.2f}%")
                    m2.metric("Best case (95th pct)", f"{stats.best_case_pct:+.2f}%")
                    m3.metric("Worst case (5th pct)", f"{stats.worst_case_pct:+.2f}%")
                    m4.metric("Volatility", f"{stats.volatility_pct:.2f}%")

                    # ---- Range visual: worst -> median -> best ----
                    fig_range = go.Figure()
                    fig_range.add_trace(go.Bar(
                        x=[stats.best_case_pct - stats.worst_case_pct],
                        y=[stats.asset], base=[stats.worst_case_pct],
                        orientation="h",
                        marker=dict(color="rgba(100,149,237,0.35)"),
                        showlegend=False, hoverinfo="skip",
                    ))
                    fig_range.add_trace(go.Scatter(
                        x=[stats.median_return_pct], y=[stats.asset],
                        mode="markers+text",
                        marker=dict(size=16, color="#2ecc71" if is_positive else "#e74c3c", symbol="diamond"),
                        text=[f"  Median {stats.median_return_pct:+.1f}%"], textposition="top center",
                        showlegend=False,
                    ))
                    fig_range.add_vline(x=0, line_dash="dot", line_color="gray")
                    fig_range.update_layout(
                        height=140, margin=dict(t=30, b=20, l=10, r=10),
                        xaxis_title="Historical return over this holding period (%)",
                        yaxis=dict(showticklabels=False),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_range, use_container_width=True, key=f"range_{stats.asset}")

                    # ---- Dollar projection ----
                    d1, d2, d3 = st.columns(3)
                    d1.metric("If worst case", f"${stats.worst_value:,.0f}", f"{stats.worst_case_pct:+.1f}%")
                    d2.metric("If median case", f"${stats.median_value:,.0f}", f"{stats.median_return_pct:+.1f}%")
                    d3.metric("If best case", f"${stats.best_value:,.0f}", f"{stats.best_case_pct:+.1f}%")

                    # ---- Full distribution (optional deep-dive) ----
                    with st.expander("See full historical distribution"):
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=stats.raw_returns_pct, nbinsx=40,
                            marker_color="rgba(100,149,237,0.6)",
                        ))
                        fig_hist.add_vline(x=stats.median_return_pct, line_color="#2ecc71",
                                            annotation_text="Median", line_width=2)
                        fig_hist.add_vline(x=0, line_dash="dot", line_color="gray")
                        fig_hist.update_layout(
                            height=280, margin=dict(t=20, b=20, l=10, r=10),
                            xaxis_title=f"Return over {days}-day holding windows (%)",
                            yaxis_title="Number of historical windows",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{stats.asset}")
                        st.caption(
                            "Each bar = how many historical windows of this length landed in that return "
                            "range. A wide, spread-out shape means high variance; a tall narrow shape "
                            "near the median means more consistent historical outcomes."
                        )

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool shows historical statistics only. It is not a forecast, "
    "not investment advice, and does not guarantee future returns. Markets can and do behave "
    "differently than their history. Consult a qualified financial professional before investing."
)
