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

import risk_metrics as rm
import benchmark as bm
import strategies as strat


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
# Verdict / "should I follow this" analyser
#
# This is a transparent, rules-based read of the stats already computed above
# — not a prediction, not personalized advice, and not a signal to act on.
# It exists to summarize what the historical numbers say in plain language,
# with the reasoning always shown alongside so nothing is a black box.
# ---------------------------------------------------------------------------

VERDICT_FAVORABLE = "Historically Favorable"
VERDICT_MIXED = "Mixed / Uncertain"
VERDICT_UNFAVORABLE = "Historically Unfavorable"


def generate_verdict(stats: "ReturnStats"):
    """
    Returns (label, color, score_0_100, reasons: list[str]).

    The score is a simple composite of win rate and risk-adjusted return,
    scaled to 0-100 for easy comparison across assets. It is NOT a
    probability, confidence level, or forecast — just a ranking aid.
    """
    reasons = []
    points = 0
    max_points = 4

    # Factor 1: win rate
    if stats.win_rate_pct >= 55:
        points += 1
        reasons.append(f"✅ Win rate is {stats.win_rate_pct:.0f}% — historically profitable more often than not.")
    elif stats.win_rate_pct <= 45:
        reasons.append(f"❌ Win rate is only {stats.win_rate_pct:.0f}% — historically unprofitable more often than not.")
    else:
        reasons.append(f"➖ Win rate is {stats.win_rate_pct:.0f}% — close to a coin flip historically.")

    # Factor 2: median return sign and magnitude
    if stats.median_return_pct > 0.3:
        points += 1
        reasons.append(f"✅ Median historical return is {stats.median_return_pct:+.2f}% — meaningfully positive.")
    elif stats.median_return_pct < 0:
        reasons.append(f"❌ Median historical return is {stats.median_return_pct:+.2f}% — the typical outcome was a loss.")
    else:
        reasons.append(f"➖ Median historical return is {stats.median_return_pct:+.2f}% — close to flat.")

    # Factor 3: risk-adjusted return (median / volatility)
    if stats.risk_adjusted >= 0.12:
        points += 1
        reasons.append(f"✅ Risk-adjusted score ({stats.risk_adjusted:.2f}) is solid — return has been decent relative to how much it swings.")
    elif stats.risk_adjusted <= 0:
        reasons.append(f"❌ Risk-adjusted score ({stats.risk_adjusted:.2f}) is negative or zero — the swings haven't historically been worth it.")
    else:
        reasons.append(f"➖ Risk-adjusted score ({stats.risk_adjusted:.2f}) is modest — some reward for the risk, but not strong.")

    # Factor 4: downside severity
    if stats.worst_case_pct >= -12:
        points += 1
        reasons.append(f"✅ Worst-case historical scenario ({stats.worst_case_pct:+.1f}%) has been relatively contained.")
    elif stats.worst_case_pct <= -25:
        reasons.append(f"❌ Worst-case historical scenario ({stats.worst_case_pct:+.1f}%) has been severe — meaningful capital loss was possible.")
    else:
        reasons.append(f"➖ Worst-case historical scenario ({stats.worst_case_pct:+.1f}%) has been moderate.")

    score = round((points / max_points) * 100)

    if points >= 3:
        label, color = VERDICT_FAVORABLE, "#2ecc71"
    elif points <= 1:
        label, color = VERDICT_UNFAVORABLE, "#e74c3c"
    else:
        label, color = VERDICT_MIXED, "#f39c12"

    return label, color, score, reasons


# ---------------------------------------------------------------------------
# Data fetching — single source (Yahoo Finance), with explicit failure modes
# ---------------------------------------------------------------------------

def is_crypto_ticker(ticker: str) -> bool:
    t = ticker.upper()
    return "-" in t and t.endswith(("USD", "USDT"))


def is_indian_ticker(ticker: str) -> bool:
    t = ticker.upper()
    return t.endswith((".NS", ".BO")) or t in ("^NSEI", "^BSESN")


def currency_symbol(ticker: str) -> str:
    """
    Yahoo Finance returns prices in the listing exchange's native currency --
    NSE/BSE-listed stocks are priced in INR, not USD. Dollar-sign labels on
    Indian tickers would be actively misleading, so this picks the right symbol.
    Does not perform currency conversion -- the amount you enter is treated as
    being in the displayed asset's own currency.
    """
    return "₹" if is_indian_ticker(ticker) else "$"


def _get_browser_session():
    """
    Yahoo Finance increasingly blocks requests that look bot-like (default
    Python requests library TLS/user-agent fingerprint), especially from
    shared cloud IPs like Streamlit Cloud's. curl_cffi impersonates a real
    Chrome browser's network fingerprint, which reliably avoids this.
    Falls back to None (plain yfinance session) if curl_cffi isn't available
    or fails to initialize, so the app never breaks over this.
    """
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_history(ticker: str, timeout_sec: int = 20, retries: int = 4):
    """
    Returns (closes: pd.Series or None, error_message: str or None).
    Shared by fetch_and_analyze and the advanced-analysis features (risk
    metrics, benchmark comparison, strategy backtests) so there's a single,
    tested source of price data with the same retry/error handling everywhere.
    """
    import time
    import random
    import yfinance as yf

    ticker = ticker.strip().upper()
    if not ticker:
        return None, "Empty ticker."

    session = _get_browser_session()

    hist = None
    had_exception = False
    for attempt in range(retries):
        try:
            ticker_obj = yf.Ticker(ticker, session=session) if session is not None else yf.Ticker(ticker)
            hist = ticker_obj.history(period="5y", auto_adjust=True, timeout=timeout_sec)
            had_exception = False
            if hist is not None and not hist.empty:
                break
        except Exception:
            had_exception = True
        # exponential backoff with jitter, so retries don't all collide on the same rate-limit window
        time.sleep((1.5 * (2 ** attempt)) + random.uniform(0, 1))

    if hist is None or hist.empty:
        if had_exception:
            return None, (
                f"Could not reach data source for '{ticker}' after {retries} attempts "
                f"(network/rate-limit issue). Try again shortly."
            )
        return None, f"'{ticker}' not found — check the symbol (e.g. AAPL, RELIANCE.NS, BTC-USD)."

    closes = hist["Close"].dropna()
    return closes, None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_quote(ticker: str):
    """
    Returns a dict with the latest available quote, or None if unavailable.
    Uses yfinance's fast_info (lighter weight than the full .info scrape).
    Note: Yahoo Finance's free tier is typically delayed 15-20 minutes for
    stocks during market hours (crypto is closer to real-time) — this is NOT
    a live tick-by-tick feed, and the UI must label it accurately.
    Falls back to None on any failure so the caller can fall back further
    (e.g. to the last close from the 5-year history) rather than break.
    """
    import yfinance as yf

    ticker = ticker.strip().upper()
    if not ticker:
        return None

    try:
        session = _get_browser_session()
        t = yf.Ticker(ticker, session=session) if session is not None else yf.Ticker(ticker)
        fi = t.fast_info
        last_price = fi.get("last_price") if hasattr(fi, "get") else fi.last_price
        prev_close = fi.get("previous_close") if hasattr(fi, "get") else fi.previous_close
        if last_price is None or prev_close is None or prev_close == 0:
            return None
        change_pct = (last_price - prev_close) / prev_close * 100
        return {
            "last_price": float(last_price),
            "previous_close": float(prev_close),
            "change_pct": float(change_pct),
            "fetched_at": pd.Timestamp.now(),
        }
    except Exception:
        return None


def fetch_and_analyze(ticker: str, amount: float, calendar_days: int, timeout_sec: int = 20, retries: int = 4):
    """
    Returns (ReturnStats, note) on success, or (None, error_message) on failure.
    Every failure mode is caught and turned into a plain-English message —
    nothing raises up to crash the app.
    """
    ticker = ticker.strip().upper()
    crypto = is_crypto_ticker(ticker)
    trading_days = calendar_to_trading_days(calendar_days, crypto)

    closes, err = fetch_price_history(ticker, timeout_sec, retries)
    if closes is None:
        return None, err

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

    note = "crypto (7-day weeks)" if crypto else (
        "Indian stock/ETF, NSE/BSE (trading-day weeks)" if is_indian_ticker(ticker) else "stock/ETF (trading-day weeks)"
    )
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
    # Indian stocks (NSE tickers, .NS suffix)
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS", "infy": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS", "hdfc": "HDFCBANK.NS", "icici bank": "ICICIBANK.NS",
    "icici": "ICICIBANK.NS", "sbi": "SBIN.NS", "state bank of india": "SBIN.NS",
    "itc": "ITC.NS", "bharti airtel": "BHARTIARTL.NS", "airtel": "BHARTIARTL.NS",
    "wipro": "WIPRO.NS", "hcl tech": "HCLTECH.NS", "hcltech": "HCLTECH.NS",
    "maruti": "MARUTI.NS", "maruti suzuki": "MARUTI.NS", "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS", "adani enterprises": "ADANIENT.NS", "adani": "ADANIENT.NS",
    "asian paints": "ASIANPAINT.NS", "bajaj finance": "BAJFINANCE.NS",
    "kotak bank": "KOTAKBANK.NS", "kotak mahindra bank": "KOTAKBANK.NS",
    "larsen": "LT.NS", "larsen and toubro": "LT.NS", "l&t": "LT.NS",
    "sun pharma": "SUNPHARMA.NS", "nifty": "^NSEI", "nifty 50": "^NSEI",
    "sensex": "^BSESN", "bse sensex": "^BSESN",
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

TOP5_US = "AAPL, MSFT, GOOGL, AMZN, NVDA"
TOP5_CRYPTO = "BTC-USD, ETH-USD, BNB-USD, SOL-USD, XRP-USD"
TOP5_INDIA = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, ICICIBANK.NS, INFY.NS"


TICKER_TO_NAME = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corporation",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "BNB",
    "SOL-USD": "Solana", "XRP-USD": "XRP",
    "RELIANCE.NS": "Reliance Industries Ltd", "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank Ltd", "ICICIBANK.NS": "ICICI Bank Ltd", "INFY.NS": "Infosys Ltd",
}

# Deterministic avatar colors -- avoids depending on external logo CDNs (Clearbit/CoinCap),
# which are unreliable to hotlink from arbitrary deployed apps and were failing to load.
AVATAR_PALETTE = ["#4A90D9", "#50B87C", "#D9784A", "#B05FC7", "#D94A7A", "#4AAFD9", "#8FB84A"]


def get_display_name(ticker: str) -> str:
    return TICKER_TO_NAME.get(ticker, ticker)


def get_avatar(ticker: str):
    """
    Returns (initials, hex_color) for a deterministic, locally-rendered avatar
    circle -- no external network request, so it always renders reliably.
    """
    name = get_display_name(ticker)
    words = [w for w in name.replace(".", " ").split() if w]
    if len(words) >= 2:
        letters = (words[0][0] + words[1][0]).upper()
    elif len(words) == 1:
        letters = words[0][:2].upper()
    else:
        letters = ticker[:2].upper()
    color = AVATAR_PALETTE[hash(ticker) % len(AVATAR_PALETTE)]
    return letters, color


def get_market_status(ticker: str) -> str:
    """
    Best-effort market open/closed check using exchange trading hours in the
    exchange's local timezone. Does NOT account for public holidays (that
    would need a market-calendar data source this tool doesn't have) — the
    UI notes that limitation. Crypto trades 24/7 so it's always "Open".
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if is_crypto_ticker(ticker):
        return "Open (24/7)"

    if is_indian_ticker(ticker):
        tz, open_hm, close_hm = ZoneInfo("Asia/Kolkata"), (9, 15), (15, 30)
    else:
        tz, open_hm, close_hm = ZoneInfo("America/New_York"), (9, 30), (16, 0)

    now = datetime.now(tz)
    if now.weekday() >= 5:  # Saturday/Sunday
        return "Closed"
    now_min = now.hour * 60 + now.minute
    open_min = open_hm[0] * 60 + open_hm[1]
    close_min = close_hm[0] * 60 + close_hm[1]
    return "Open" if open_min <= now_min <= close_min else "Closed"


def build_top5_rate_table(tickers_csv: str) -> list:
    """
    Builds quick-glance data per asset: live rate, today's change (for color
    coding), market open/closed status, last close, and the expected 30-day
    low/high range based on historical 5th/95th percentile returns applied
    to the current price. Returns a list of dicts (not a plain DataFrame) so
    the caller can render it with custom per-cell coloring.
    """
    rows = []
    for ticker in [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]:
        cur = currency_symbol(ticker)
        market_status = get_market_status(ticker)
        stats, err = fetch_and_analyze(ticker, amount=1, calendar_days=30)
        if stats is None:
            rows.append({
                "ticker": ticker, "ok": False, "error": err, "market_status": market_status,
            })
            continue

        quote = fetch_live_quote(ticker)
        if quote:
            live_price = quote["last_price"]
            change_pct = quote["change_pct"]
            last_close = quote["previous_close"]
        else:
            fallback_closes, _ = fetch_price_history(ticker)
            if fallback_closes is not None and len(fallback_closes) >= 2:
                live_price = float(fallback_closes.iloc[-1])
                last_close = float(fallback_closes.iloc[-2])
                change_pct = (live_price - last_close) / last_close * 100
            else:
                live_price = None
                change_pct = None
                last_close = None

        if live_price is None:
            rows.append({
                "ticker": ticker, "ok": False, "error": "Price unavailable", "market_status": market_status,
            })
            continue

        lowest = live_price * (1 + stats.worst_case_pct / 100)
        highest = live_price * (1 + stats.best_case_pct / 100)
        rows.append({
            "ticker": ticker, "ok": True, "currency": cur,
            "live_price": live_price, "change_pct": change_pct,
            "last_close": last_close, "market_status": market_status,
            "lowest": lowest, "highest": highest,
        })
    return rows


def render_top5_table(rows: list):
    """Renders the Top-5 table as HTML with logo, full name, color-coded live rate, and market-status badge."""
    html = (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<tr style='border-bottom:1px solid rgba(128,128,128,0.4);'>"
        "<td style='padding:8px 4px;color:gray;'>Stock</td>"
        "<td style='padding:8px 4px;color:gray;text-align:right;'>Live rate</td>"
        "<td style='padding:8px 4px;color:gray;text-align:center;'>Market</td>"
        "<td style='padding:8px 4px;color:gray;text-align:right;'>Last close</td>"
        "<td style='padding:8px 4px;color:gray;text-align:right;'>Lowest expected (30d)</td>"
        "<td style='padding:8px 4px;color:gray;text-align:right;'>Highest expected (30d)</td>"
        "</tr>"
    )
    for r in rows:
        ticker = r["ticker"]
        name = get_display_name(ticker)
        initials, avatar_color = get_avatar(ticker)
        logo_html = (
            f"<span style='display:inline-flex;align-items:center;justify-content:center;"
            f"width:26px;height:26px;border-radius:50%;background:{avatar_color};"
            f"color:white;font-size:11px;font-weight:600;vertical-align:middle;"
            f"margin-right:8px;flex-shrink:0;'>{initials}</span>"
        )
        name_block = (
            f"<div style='display:flex;align-items:center;'>{logo_html}"
            f"<div><div style='font-weight:600;'>{name}</div>"
            f"<div style='font-size:11px;color:gray;'>{ticker}</div></div></div>"
        )

        if not r.get("ok"):
            html += (
                f"<tr style='border-bottom:1px solid rgba(128,128,128,0.2);'>"
                f"<td style='padding:8px 4px;'>{name_block}</td>"
                f"<td colspan='5' style='padding:8px 4px;color:gray;'>Unavailable: {r.get('error','')}</td>"
                f"</tr>"
            )
            continue
        chg = r["change_pct"]
        rate_color = "#2ecc71" if chg is not None and chg >= 0 else "#e74c3c"
        arrow = "▲" if chg is not None and chg >= 0 else "▼"
        chg_txt = f" {arrow} {chg:+.2f}%" if chg is not None else ""
        status_color = "#2ecc71" if "Open" in r["market_status"] else "#888"
        html += (
            f"<tr style='border-bottom:1px solid rgba(128,128,128,0.2);'>"
            f"<td style='padding:10px 4px;'>{name_block}</td>"
            f"<td style='padding:10px 4px;text-align:right;color:{rate_color};font-weight:600;'>"
            f"{r['currency']}{r['live_price']:,.2f}{chg_txt}</td>"
            f"<td style='padding:10px 4px;text-align:center;'>"
            f"<span style='color:{status_color};font-size:12px;'>● {r['market_status']}</span></td>"
            f"<td style='padding:10px 4px;text-align:right;'>{r['currency']}{r['last_close']:,.2f}</td>"
            f"<td style='padding:10px 4px;text-align:right;color:#e74c3c;'>{r['currency']}{r['lowest']:,.2f}</td>"
            f"<td style='padding:10px 4px;text-align:right;color:#2ecc71;'>{r['currency']}{r['highest']:,.2f}</td>"
            f"</tr>"
        )
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


if "manual_tickers_input" not in st.session_state:
    st.session_state["manual_tickers_input"] = "AAPL, MSFT, BTC-USD"
if "expand_manual" not in st.session_state:
    st.session_state["expand_manual"] = False
if "top5_table_group" not in st.session_state:
    st.session_state["top5_table_group"] = None

st.caption("Quick pick a curated list (major companies/coins by market cap — not live-ranked):")
q1, q2, q3 = st.columns(3)
with q1:
    if st.button("🇺🇸 Top 5 US Stocks", use_container_width=True):
        st.session_state["manual_tickers_input"] = TOP5_US
        st.session_state["expand_manual"] = True
        st.session_state["top5_table_group"] = TOP5_US
with q2:
    if st.button("₿ Top 5 Crypto", use_container_width=True):
        st.session_state["manual_tickers_input"] = TOP5_CRYPTO
        st.session_state["expand_manual"] = True
        st.session_state["top5_table_group"] = TOP5_CRYPTO
with q3:
    if st.button("🇮🇳 Top 5 Indian Stocks", use_container_width=True):
        st.session_state["manual_tickers_input"] = TOP5_INDIA
        st.session_state["expand_manual"] = True
        st.session_state["top5_table_group"] = TOP5_INDIA

@st.fragment(run_every=30)
def _render_top5_fragment():
    if st.session_state["top5_table_group"]:
        top5_rows = build_top5_rate_table(st.session_state["top5_table_group"])
        render_top5_table(top5_rows)
        st.caption(
            f"Auto-refreshes every 30s (last pulled {pd.Timestamp.now().strftime('%H:%M:%S')}) · "
            "Lowest/Highest expected = the 5th/95th percentile of historical 30-day returns applied "
            "to today's price — not a guarantee. Rates shown in each stock's own listing currency. "
            "Market status doesn't account for public holidays."
        )


if st.session_state["top5_table_group"]:
    _render_top5_fragment()

st.subheader("Ask in plain English")
prompt_text = st.text_input(
    "e.g. \"Invest $5000 in Apple, Microsoft and Bitcoin for 30 days\"",
    placeholder="Invest $5000 in Apple, Microsoft and Bitcoin for 30 days",
    key="prompt_box",
    label_visibility="collapsed",
)
prompt_submitted = st.button("🔍 Analyze", type="primary")

with st.expander("Or fill in the fields manually / review your quick pick", expanded=st.session_state["expand_manual"]):
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        with col1:
            m_amount = st.number_input("Investment amount", min_value=0.0, value=5000.0, step=100.0)
        with col2:
            m_days = st.number_input("Holding period (calendar days)", min_value=0, value=30, step=1)
        m_tickers_raw = st.text_input(
            "Assets (comma-separated)",
            key="manual_tickers_input",
            help="Stock tickers (AAPL, RELIANCE.NS, TCS.BO) and/or crypto (BTC-USD, ETH-USD, SOL-USD). "
                 "For Indian stocks use the NSE (.NS) or BSE (.BO) suffix.",
        )
        manual_submitted = st.form_submit_button("Run Analysis (manual)")

adv_col1, adv_col2 = st.columns([2, 1])
with adv_col1:
    benchmark_choice = st.selectbox(
        "Compare against benchmark",
        ["SPY (S&P 500)", "QQQ (Nasdaq 100)", "GLD (Gold)", "BTC-USD",
         "NIFTY 50 (India)", "SENSEX (India)", "None"],
        index=0,
        help="Note: comparing a USD-priced asset against an INR-priced benchmark (or vice versa) "
             "mixes currencies — the % and ratio metrics (beta, correlation) are still valid since "
             "they're computed on returns, not absolute price levels, but keep the currency mismatch in mind.",
    )
with adv_col2:
    run_advanced = st.checkbox("Include risk/strategy analysis", value=True,
                                help="Adds risk metrics, benchmark comparison, and a strategy backtest leaderboard for each asset — takes a bit longer.")

CURRENCY_PLACEHOLDER = "— Select a currency — (required)"
CURRENCY_SYMBOL_MAP = {
    "USD ($)": "$", "AED (د.إ)": "د.إ", "INR (₹)": "₹", "EUR (€)": "€",
    "GBP (£)": "£", "SAR (﷼)": "﷼", "JPY (¥)": "¥", "SGD (S$)": "S$",
}


def resolve_forced_currency_symbol(currency_choice: str, custom_symbol: str = None):
    """
    Returns None if no fixed currency should be forced (either nothing has been
    selected yet, or the user explicitly chose Auto — meaning fall back to
    per-asset detection: ₹ for Indian tickers, $ otherwise). Returns a fixed
    symbol string for every other explicit choice, including Custom.
    """
    if not currency_choice or currency_choice == CURRENCY_PLACEHOLDER:
        return None
    if currency_choice.startswith("Auto"):
        return None
    if currency_choice == "Custom":
        return (custom_symbol or "").strip() or "$"
    return CURRENCY_SYMBOL_MAP.get(currency_choice)


cur_col1, cur_col2 = st.columns([2, 1])
with cur_col1:
    currency_choice = st.selectbox(
        "Display currency for the amount you enter *",
        [CURRENCY_PLACEHOLDER, "Auto (₹ for Indian stocks, $ for everything else)", "USD ($)", "AED (د.إ)",
         "INR (₹)", "EUR (€)", "GBP (£)", "SAR (﷼)", "JPY (¥)", "SGD (S$)", "Custom"],
        index=0,
        help="Required before running an analysis. This does NOT convert currencies — it just "
             "labels your amount and the projected results in the currency you pick, and applies "
             "the historical % return directly to that number. It does not account for "
             "exchange-rate movements.",
    )
with cur_col2:
    custom_currency_symbol = None
    if currency_choice == "Custom":
        custom_currency_symbol = st.text_input("Symbol/code", value="CHF", max_chars=6)

forced_currency_symbol = resolve_forced_currency_symbol(currency_choice, custom_currency_symbol)
currency_was_selected = currency_choice != CURRENCY_PLACEHOLDER


def display_currency(ticker: str) -> str:
    """Uses the user's chosen display currency if set, otherwise auto-detects per asset."""
    return forced_currency_symbol if forced_currency_symbol else currency_symbol(ticker)


BENCHMARK_MAP = {
    "SPY (S&P 500)": "SPY", "QQQ (Nasdaq 100)": "QQQ", "GLD (Gold)": "GLD",
    "BTC-USD": "BTC-USD", "NIFTY 50 (India)": "^NSEI", "SENSEX (India)": "^BSESN", "None": None,
}
benchmark_ticker = BENCHMARK_MAP[benchmark_choice]

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
            f"**Understood:** Amount = {f'{p_amount:,.0f}' if p_amount else '—'} | "
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
    if not currency_was_selected:
        errors.append("Please select a display currency above before running the analysis.")
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
            summary_cur = forced_currency_symbol if forced_currency_symbol else ""
            st.success(f"✅ Analysis complete for {len(results)} of {len(tickers)} asset(s), on {summary_cur}{amount:,.0f} over {days} days.")
            if forced_currency_symbol:
                st.caption(
                    f"ℹ️ All amounts shown in {forced_currency_symbol} as selected — this tool doesn't convert "
                    "currencies, it applies the historical % return directly to your entered amount and "
                    "labels the result in your chosen currency. Exchange-rate movements aren't modeled."
                )
            elif any(is_indian_ticker(s.asset) for s, _ in results):
                st.caption(
                    "ℹ️ Indian stocks (NSE/BSE) are shown in ₹ — this tool does not convert currencies, "
                    "the amount you entered is applied as-is per asset in that asset's own currency."
                )

            # ---- Quick summary table: win rate, expected (median), minimum (worst case) ----
            st.markdown("### Quick summary")
            summary_rows = []
            for s, note in ranked:
                cur = display_currency(s.asset)
                summary_rows.append({
                    "Asset": s.asset,
                    "Win Rate %": round(s.win_rate_pct, 1),
                    "Expected (median)": f"{cur}{s.median_value:,.0f}",
                    "Minimum (worst case)": f"{cur}{s.worst_value:,.0f}",
                    "Maximum (best case)": f"{cur}{s.best_value:,.0f}",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Win Rate = % of historical windows of this length that were profitable. "
                "Expected = median historical outcome. Minimum/Maximum = 5th/95th percentile "
                "historical outcomes (not absolute floors/ceilings — worse or better has happened)."
            )

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
            hdr_col1, hdr_col2 = st.columns([4, 1])
            with hdr_col1:
                st.markdown("### Asset breakdown")
            with hdr_col2:
                if st.button("🔄 Refresh prices", use_container_width=True):
                    fetch_live_quote.clear()
                    st.rerun()

            for stats, note in ranked:
                is_positive = stats.median_return_pct >= 0
                emoji = "🟢" if is_positive else "🔴"
                verdict_label, verdict_color, verdict_score, verdict_reasons = generate_verdict(stats)
                with st.container(border=True):
                    top_col1, top_col2 = st.columns([3, 1])
                    with top_col1:
                        st.markdown(f"#### {emoji} {stats.asset}  \n*{note} · {stats.num_windows} historical windows analyzed*")
                    with top_col2:
                        st.metric("Win rate", f"{stats.win_rate_pct:.0f}%")

                    # ---- Current market price (delayed, not live tick-by-tick) ----
                    cur = display_currency(stats.asset)
                    quote = fetch_live_quote(stats.asset)
                    if quote:
                        chg = quote["change_pct"]
                        chg_color = "var(--text-success)" if chg >= 0 else "var(--text-danger)"
                        chg_arrow = "▲" if chg >= 0 else "▼"
                        st.markdown(
                            f"**Current price:** {cur}{quote['last_price']:,.2f} "
                            f"<span style='color:{chg_color}'>{chg_arrow} {chg:+.2f}% today</span>  \n"
                            f"<span style='font-size:0.8em;color:gray'>Quotes are typically delayed 15-20 min for stocks "
                            f"(not real-time) · fetched {quote['fetched_at'].strftime('%H:%M:%S')}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        fallback_closes, _ = fetch_price_history(stats.asset)
                        if fallback_closes is not None and len(fallback_closes) > 0:
                            last_close = fallback_closes.iloc[-1]
                            last_date = fallback_closes.index[-1]
                            st.caption(
                                f"Live quote unavailable right now — showing last available close: "
                                f"{cur}{last_close:,.2f} (as of {last_date.strftime('%Y-%m-%d')})"
                            )
                        else:
                            st.caption("Live quote unavailable right now.")

                    # ---- Verdict badge ----
                    st.markdown(
                        f"""
                        <div style="background-color:{verdict_color}22; border-left:4px solid {verdict_color};
                                    border-radius:6px; padding:10px 14px; margin:8px 0;">
                            <span style="font-size:1.05em; font-weight:600; color:{verdict_color};">
                                {verdict_label}
                            </span>
                            <span style="opacity:0.7;"> · Historical Favorability Score: {verdict_score}/100</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("Why this verdict?"):
                        for r in verdict_reasons:
                            st.write(r)
                        st.caption(
                            "This is a rules-based summary of the historical statistics above, based on "
                            "win rate, median return, risk-adjusted return, and worst-case severity for "
                            "this exact holding period. It is not a prediction, not personalized financial "
                            "advice, and not a signal to buy or sell — just a plain-language read of what "
                            "the history shows."
                        )

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
                    cur = display_currency(stats.asset)
                    d1.metric("If worst case", f"{cur}{stats.worst_value:,.0f}", f"{stats.worst_case_pct:+.1f}%")
                    d2.metric("If median case", f"{cur}{stats.median_value:,.0f}", f"{stats.median_return_pct:+.1f}%")
                    d3.metric("If best case", f"{cur}{stats.best_value:,.0f}", f"{stats.best_case_pct:+.1f}%")

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

                    # ---- Advanced analysis: risk metrics, benchmark, strategy leaderboard ----
                    if run_advanced:
                        with st.expander("📈 Advanced analysis: risk, benchmark & strategy backtest"):
                            closes_full, adv_err = fetch_price_history(stats.asset)
                            if closes_full is None or len(closes_full) < 30:
                                st.warning(f"Couldn't load enough data for advanced analysis: {adv_err or 'insufficient history'}")
                            else:
                                risk_report = rm.full_risk_report(closes_full)

                                st.markdown("**Risk metrics** (computed from full 5-year daily history)")
                                r1, r2, r3, r4 = st.columns(4)
                                r1.metric("Sharpe Ratio", f"{risk_report['sharpe_ratio']:.2f}")
                                r2.metric("Sortino Ratio", f"{risk_report['sortino_ratio']:.2f}")
                                r3.metric("Max Drawdown", f"{risk_report['max_drawdown_pct']:.1f}%")
                                r4.metric("Calmar Ratio", f"{risk_report['calmar_ratio']:.2f}")
                                r5, r6, r7, r8 = st.columns(4)
                                r5.metric("VaR (95%, daily)", f"{risk_report['var_95_pct']:.2f}%")
                                r6.metric("CVaR (95%, daily)", f"{risk_report['cvar_95_pct']:.2f}%")
                                r7.metric("Ulcer Index", f"{risk_report['ulcer_index']:.2f}")
                                r8.metric("Skewness", f"{risk_report['skewness']:.2f}")

                                fig_dd = go.Figure()
                                fig_dd.add_trace(go.Scatter(
                                    x=risk_report["drawdown_series"].index, y=risk_report["drawdown_series"],
                                    fill="tozeroy", line=dict(color="#e74c3c"), name="Drawdown",
                                ))
                                fig_dd.update_layout(
                                    height=200, title="Drawdown over time (5y)",
                                    margin=dict(t=30, b=10, l=10, r=10),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(fig_dd, use_container_width=True, key=f"dd_{stats.asset}")

                                # ---- Benchmark comparison ----
                                if benchmark_ticker and benchmark_ticker.upper() != stats.asset.upper():
                                    bench_closes, bench_err = fetch_price_history(benchmark_ticker)
                                    if bench_closes is None:
                                        st.info(f"Couldn't load benchmark ({benchmark_ticker}): {bench_err}")
                                    else:
                                        try:
                                            bench_report = bm.full_benchmark_report(closes_full, bench_closes)
                                            st.markdown(f"**Benchmark comparison vs {benchmark_ticker}**")
                                            b1, b2, b3 = st.columns(3)
                                            b1.metric("Alpha (annualized)", f"{bench_report['alpha_pct']:+.2f}%")
                                            b2.metric("Beta", f"{bench_report['beta']:.2f}")
                                            b3.metric("Correlation", f"{bench_report['correlation']:.2f}")
                                            b4, b5, b6 = st.columns(3)
                                            b4.metric("Tracking Error", f"{bench_report['tracking_error_pct']:.2f}%")
                                            b5.metric("Information Ratio", f"{bench_report['information_ratio']:.2f}")
                                            b6.metric("Relative Return", f"{bench_report['relative_return_pct']:+.2f}%")
                                        except ValueError as e:
                                            st.info(str(e))

                                # ---- Strategy leaderboard ----
                                st.markdown("**Strategy leaderboard** (Buy & Hold vs. simple systematic rules, no lookahead bias)")
                                leaderboard = strat.run_starter_leaderboard(closes_full)
                                lb_rows = [{
                                    "Strategy": r.name,
                                    "Ann. Return %": round(r.annualized_return_pct, 2),
                                    "Win Rate %": round(r.win_rate_pct, 1),
                                    "Sharpe": round(r.sharpe_ratio, 2),
                                    "Sortino": round(r.sortino_ratio, 2),
                                    "Profit Factor": (round(r.profit_factor, 2) if r.profit_factor != float("inf") else "∞"),
                                    "Max Drawdown %": round(r.max_drawdown_pct, 2),
                                    "Trades": r.trade_count,
                                } for r in leaderboard]
                                st.dataframe(pd.DataFrame(lb_rows), use_container_width=True, hide_index=True)

                                buy_hold = next((r for r in leaderboard if r.name == "Buy & Hold"), None)
                                best = leaderboard[0]
                                if buy_hold and best.name != "Buy & Hold" and best.sharpe_ratio > buy_hold.sharpe_ratio:
                                    st.success(
                                        f"📌 Historically, **{best.name}** had a better risk-adjusted return "
                                        f"(Sharpe {best.sharpe_ratio:.2f}) than simple Buy & Hold "
                                        f"(Sharpe {buy_hold.sharpe_ratio:.2f}) for this asset."
                                    )
                                else:
                                    st.info("📌 Historically, simple Buy & Hold performed as well as or better than the other rules tested here.")

                                fig_eq = go.Figure()
                                for r in leaderboard:
                                    fig_eq.add_trace(go.Scatter(x=r.equity_curve.index, y=r.equity_curve, name=r.name))
                                fig_eq.update_layout(
                                    height=300, title="Strategy equity curves (start = 100)",
                                    margin=dict(t=30, b=10, l=10, r=10),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(fig_eq, use_container_width=True, key=f"eq_{stats.asset}")

                                st.caption(
                                    "Strategy backtests use next-day execution (today's signal, tomorrow's return) "
                                    "to avoid lookahead bias, and do not account for trading fees, slippage, taxes, "
                                    "or bid/ask spreads. Past strategy performance does not guarantee future results."
                                )

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool shows historical statistics only. It is not a forecast, "
    "not investment advice, and does not guarantee future returns. Markets can and do behave "
    "differently than their history. Consult a qualified financial professional before investing."
)
