"""
risk_metrics.py — deterministic, documented risk statistics.

Every function here takes a series of periodic returns (as decimals, e.g. 0.01
for 1%) or a price series, and returns a single float or a small dict. No
randomness, no ML, no black boxes — every formula is a standard, citable
financial calculation. Docstrings state the formula used.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


TRADING_DAYS_PER_YEAR = 252


def daily_returns(price_series: pd.Series) -> pd.Series:
    """Simple daily % returns (not log returns), as decimals."""
    return price_series.pct_change().dropna()


def max_drawdown(price_series: pd.Series):
    """
    Max Drawdown = the largest peak-to-trough decline in the price series.
    Formula: drawdown(t) = price(t) / running_max(price up to t) - 1
    Returns (max_drawdown_pct, drawdown_series) — the series is used for charting.
    """
    running_max = price_series.cummax()
    drawdown_series = (price_series / running_max - 1) * 100
    return float(drawdown_series.min()), drawdown_series


def recovery_time_days(price_series: pd.Series) -> int:
    """
    Days between the trough of the worst drawdown and the point price first
    recovered to its prior peak. Returns -1 if it never recovered within
    the available history.
    """
    running_max = price_series.cummax()
    drawdown_series = price_series / running_max - 1
    trough_idx = drawdown_series.idxmin()
    peak_before_trough = running_max.loc[trough_idx]

    after_trough = price_series.loc[trough_idx:]
    recovered = after_trough[after_trough >= peak_before_trough]
    if recovered.empty:
        return -1
    recovery_idx = recovered.index[0]
    return int((recovery_idx - trough_idx).days)


def annualized_return(price_series: pd.Series) -> float:
    """CAGR = (end_price / start_price) ^ (365 / days_held) - 1, as a percent."""
    if isinstance(price_series.index, pd.DatetimeIndex):
        days_held = (price_series.index[-1] - price_series.index[0]).days
    else:
        # Fall back to assuming one trading day per row (~365/252 calendar days each)
        days_held = (len(price_series) - 1) * (365 / TRADING_DAYS_PER_YEAR)
    if days_held <= 0:
        return 0.0
    total_return = price_series.iloc[-1] / price_series.iloc[0]
    cagr = total_return ** (365 / days_held) - 1
    return float(cagr * 100)


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """
    Sharpe Ratio = (mean daily excess return / std of daily returns) * sqrt(252)
    Standard formula (Sharpe, 1966/1994). risk_free_annual is a decimal (e.g. 0.04 for 4%).
    """
    daily_rf = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """
    Sortino Ratio = (mean daily excess return / downside deviation) * sqrt(252)
    Like Sharpe, but only penalizes downside volatility, not upside swings.
    """
    daily_rf = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < 0]
    dd = downside.std()
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float((excess.mean() / dd) * np.sqrt(TRADING_DAYS_PER_YEAR))


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """
    Downside Deviation = std dev of returns falling below a minimum acceptable
    return (MAR, default 0). Annualized, as a percent.
    """
    downside = returns[returns < mar] - mar
    if len(downside) == 0:
        return 0.0
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)


def calmar_ratio(price_series: pd.Series) -> float:
    """Calmar Ratio = CAGR / |Max Drawdown|."""
    cagr = annualized_return(price_series)
    mdd, _ = max_drawdown(price_series)
    if mdd == 0:
        return 0.0
    return float(cagr / abs(mdd))


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level: the loss threshold that
    returns did not exceed in (confidence * 100)% of historical periods.
    e.g. 95% VaR of -2% means: historically, 95% of periods lost less than 2%.
    Returned as a percent (negative = a loss).
    """
    return float(np.percentile(returns, (1 - confidence) * 100) * 100)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) = average return in the worst
    (1-confidence) tail of historical outcomes. More conservative than VaR
    since it captures how bad the bad days actually were, not just the cutoff.
    """
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return float(threshold * 100)
    return float(tail.mean() * 100)


def ulcer_index(price_series: pd.Series) -> float:
    """
    Ulcer Index = sqrt(mean(drawdown^2)) — penalizes both the depth AND
    duration of drawdowns, unlike max drawdown which only looks at the worst point.
    """
    running_max = price_series.cummax()
    drawdown_pct = (price_series / running_max - 1) * 100
    return float(np.sqrt((drawdown_pct ** 2).mean()))


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega Ratio = sum of gains above threshold / sum of losses below threshold.
    >1 means historical gains outweighed losses at that threshold.
    """
    gains = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns < threshold]).sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def skewness(returns: pd.Series) -> float:
    """
    Skewness of the return distribution. Positive = long right tail (occasional
    big gains); negative = long left tail (occasional big losses) — often more
    concerning for risk since losses hurt more than equivalent gains help.
    """
    return float(scipy_stats.skew(returns.dropna()))


def kurtosis(returns: pd.Series) -> float:
    """
    Excess kurtosis (relative to a normal distribution, which has kurtosis 0
    under this convention). High kurtosis = fatter tails = more extreme moves
    than a normal distribution would suggest.
    """
    return float(scipy_stats.kurtosis(returns.dropna()))


def full_risk_report(price_series: pd.Series, risk_free_annual: float = 0.0) -> dict:
    """Convenience wrapper: computes every metric above and returns a dict."""
    rets = daily_returns(price_series)
    mdd, dd_series = max_drawdown(price_series)
    return {
        "annualized_return_pct": annualized_return(price_series),
        "max_drawdown_pct": mdd,
        "drawdown_series": dd_series,
        "recovery_time_days": recovery_time_days(price_series),
        "sharpe_ratio": sharpe_ratio(rets, risk_free_annual),
        "sortino_ratio": sortino_ratio(rets, risk_free_annual),
        "downside_deviation_pct": downside_deviation(rets),
        "calmar_ratio": calmar_ratio(price_series),
        "var_95_pct": value_at_risk(rets, 0.95),
        "cvar_95_pct": conditional_var(rets, 0.95),
        "ulcer_index": ulcer_index(price_series),
        "omega_ratio": omega_ratio(rets),
        "skewness": skewness(rets),
        "kurtosis": kurtosis(rets),
    }
