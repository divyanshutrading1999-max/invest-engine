"""
strategies.py — simple, transparent, rule-based strategy backtests.

Every strategy generates a position series (1 = fully invested, 0 = flat) from
a deterministic rule applied to historical prices. No optimization, no
curve-fitting, no lookahead: each day's position is decided using only data
available up to and including that day, and returns are realized the
following day (position.shift(1) * daily_return) to avoid lookahead bias.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


TRADING_DAYS_PER_YEAR = 252


@dataclass
class StrategyResult:
    name: str
    annualized_return_pct: float
    win_rate_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    max_drawdown_pct: float
    avg_trade_pct: float
    trade_count: int
    equity_curve: pd.Series  # normalized to start at 100


# ---------------------------------------------------------------------------
# Position rules — each returns a Series of 0/1 (or -1..1 in principle, but
# these three are long-only/flat for simplicity and to avoid short-selling
# assumptions that don't apply cleanly to all asset types).
# ---------------------------------------------------------------------------

def positions_buy_and_hold(prices: pd.Series) -> pd.Series:
    return pd.Series(1, index=prices.index)


def positions_ma_crossover(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """Long when fast MA > slow MA, flat otherwise."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    pos = (fast_ma > slow_ma).astype(int)
    pos[fast_ma.isna() | slow_ma.isna()] = 0
    return pos


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI (Wilder's smoothing)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def positions_rsi_mean_reversion(prices: pd.Series, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """
    Long when RSI drops below `oversold` (expecting a bounce), exit when RSI
    rises above `overbought`. Holds position between signals (doesn't flip-flop
    every day RSI is mid-range).
    """
    rsi = _rsi(prices, period)
    pos = pd.Series(0, index=prices.index)
    holding = False
    for i in range(len(rsi)):
        if not holding and rsi.iloc[i] < oversold:
            holding = True
        elif holding and rsi.iloc[i] > overbought:
            holding = False
        pos.iloc[i] = 1 if holding else 0
    return pos


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def backtest(prices: pd.Series, positions: pd.Series, name: str, risk_free_annual: float = 0.0) -> StrategyResult:
    daily_ret = prices.pct_change().fillna(0)
    # Position decided on day t is realized as a return on day t+1 -- no lookahead.
    strat_ret = positions.shift(1).fillna(0) * daily_ret

    equity_curve = 100 * (1 + strat_ret).cumprod()

    total_days = len(prices)
    years = total_days / TRADING_DAYS_PER_YEAR
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    ann_return = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    active_days = strat_ret[positions.shift(1).fillna(0) != 0]
    win_rate = float((active_days > 0).mean() * 100) if len(active_days) > 0 else 0.0

    std = strat_ret.std()
    sharpe = float((strat_ret.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)) if std and not np.isnan(std) and std != 0 else 0.0

    downside = strat_ret[strat_ret < 0]
    dstd = downside.std()
    sortino = float((strat_ret.mean() / dstd) * np.sqrt(TRADING_DAYS_PER_YEAR)) if dstd and not np.isnan(dstd) and dstd != 0 else 0.0

    gains = strat_ret[strat_ret > 0].sum()
    losses = -strat_ret[strat_ret < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

    running_max = equity_curve.cummax()
    mdd = float((equity_curve / running_max - 1).min() * 100)

    # Count discrete trades = number of times position flips from 0->1 (entries)
    pos_shifted = positions.shift(1).fillna(0)
    entries = ((pos_shifted == 1) & (pos_shifted.shift(1).fillna(0) == 0)).sum()
    trade_count = int(entries)

    avg_trade = float(active_days.mean() * 100) if len(active_days) > 0 else 0.0

    return StrategyResult(
        name=name,
        annualized_return_pct=ann_return,
        win_rate_pct=win_rate,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        profit_factor=profit_factor,
        max_drawdown_pct=mdd,
        avg_trade_pct=avg_trade,
        trade_count=trade_count,
        equity_curve=equity_curve,
    )


def run_starter_leaderboard(prices: pd.Series, risk_free_annual: float = 0.0) -> list:
    """
    Runs the Phase-1 starter set of strategies: Buy & Hold, 50/200 MA Crossover,
    RSI Mean Reversion. Returns a list of StrategyResult sorted by Sharpe ratio
    (best first). More strategies (MACD, Bollinger, Donchian, etc.) are planned
    for a later phase.
    """
    results = []

    results.append(backtest(prices, positions_buy_and_hold(prices), "Buy & Hold", risk_free_annual))

    if len(prices) > 210:
        results.append(backtest(
            prices, positions_ma_crossover(prices, 50, 200), "50/200 MA Crossover", risk_free_annual
        ))
    if len(prices) > 60:
        results.append(backtest(
            prices, positions_ma_crossover(prices, 20, 50), "20/50 MA Crossover", risk_free_annual
        ))

    results.append(backtest(
        prices, positions_rsi_mean_reversion(prices), "RSI Mean Reversion (14, 30/70)", risk_free_annual
    ))

    return sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)
