"""
benchmark.py — compares an asset's returns against a benchmark (e.g. S&P 500).

All formulas are standard CAPM / portfolio-theory calculations.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def align_returns(asset_prices: pd.Series, benchmark_prices: pd.Series):
    """
    Align two price series on their common dates and compute daily returns
    for both. Returns (asset_returns, benchmark_returns) as aligned Series.
    """
    df = pd.DataFrame({"asset": asset_prices, "benchmark": benchmark_prices}).dropna()
    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} overlapping trading days between asset and benchmark — "
            "not enough for a reliable comparison (need at least 30)."
        )
    asset_returns = df["asset"].pct_change().dropna()
    benchmark_returns = df["benchmark"].pct_change().dropna()
    # re-align after pct_change drops the first row
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    return asset_returns.loc[common_idx], benchmark_returns.loc[common_idx]


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta = Cov(asset, benchmark) / Var(benchmark) — sensitivity to benchmark moves."""
    cov_matrix = np.cov(asset_returns, benchmark_returns)
    benchmark_var = cov_matrix[1, 1]
    if benchmark_var == 0:
        return 0.0
    return float(cov_matrix[0, 1] / benchmark_var)


def alpha(asset_returns: pd.Series, benchmark_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """
    Jensen's Alpha (annualized %) = asset's actual annualized return minus what
    CAPM would predict given its beta and the benchmark's return.
    alpha = R_asset - [R_f + beta * (R_benchmark - R_f)]
    """
    b = beta(asset_returns, benchmark_returns)
    daily_rf = risk_free_annual / TRADING_DAYS_PER_YEAR
    asset_annual = (1 + asset_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
    bench_annual = (1 + benchmark_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
    expected = risk_free_annual + b * (bench_annual - risk_free_annual)
    return float((asset_annual - expected) * 100)


def correlation(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Pearson correlation coefficient between asset and benchmark daily returns."""
    return float(np.corrcoef(asset_returns, benchmark_returns)[0, 1])


def tracking_error(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Tracking Error (annualized %) = std dev of (asset return - benchmark return).
    How much the asset's returns deviate from the benchmark, day to day.
    """
    diff = asset_returns - benchmark_returns
    return float(diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)


def information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Information Ratio = annualized excess return / tracking error.
    How much extra return per unit of active risk taken relative to the benchmark.
    """
    diff = asset_returns - benchmark_returns
    te = diff.std()
    if te == 0:
        return 0.0
    return float((diff.mean() / te) * np.sqrt(TRADING_DAYS_PER_YEAR))


def relative_return(asset_prices: pd.Series, benchmark_prices: pd.Series) -> float:
    """Simple total-return comparison over the overlapping period, in percentage points."""
    df = pd.DataFrame({"asset": asset_prices, "benchmark": benchmark_prices}).dropna()
    asset_total = (df["asset"].iloc[-1] / df["asset"].iloc[0] - 1) * 100
    bench_total = (df["benchmark"].iloc[-1] / df["benchmark"].iloc[0] - 1) * 100
    return float(asset_total - bench_total)


def full_benchmark_report(asset_prices: pd.Series, benchmark_prices: pd.Series, risk_free_annual: float = 0.0) -> dict:
    """Convenience wrapper: computes every benchmark metric and returns a dict."""
    a_ret, b_ret = align_returns(asset_prices, benchmark_prices)
    return {
        "alpha_pct": alpha(a_ret, b_ret, risk_free_annual),
        "beta": beta(a_ret, b_ret),
        "correlation": correlation(a_ret, b_ret),
        "tracking_error_pct": tracking_error(a_ret, b_ret),
        "information_ratio": information_ratio(a_ret, b_ret),
        "relative_return_pct": relative_return(asset_prices, benchmark_prices),
    }
