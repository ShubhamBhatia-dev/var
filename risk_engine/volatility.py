"""
risk_engine/volatility.py
--------------------------
Rolling volatility analysis and regime detection.

METHODOLOGY:
============
We calculate rolling portfolio volatility using the standard deviation
of daily realized P&L over a rolling window.

Two windows are provided:
  - Short window (20 days): Captures recent volatility regime changes
  - Long window (60 days): Provides a smoother medium-term trend

ANNUALIZATION:
==============
Daily volatility is annualized using:
  annualized_vol = daily_vol × sqrt(252)

This assumes 252 trading days per year.

VOLATILITY REGIMES:
===================
We identify high-volatility periods where the rolling 20-day volatility
exceeds the historical Nth percentile (configurable, default: 80th).

These periods are flagged for the dashboard and used in the residual
risk add-on calculation.

RISK CONTRIBUTION METHODOLOGY:
================================
For marginal risk contribution, we use a simple component volatility approach:
  - Portfolio variance: σ²_P = Σᵢ Σⱼ wᵢ wⱼ σᵢⱼ
  - Marginal contribution: ∂σ_P / ∂wᵢ ∝ Σⱼ wⱼ σᵢⱼ
  - Component volatility: wᵢ × (∂σ_P / ∂wᵢ) / σ_P

These are approximations using sample covariances and do not constitute
rigorous Marginal VaR calculations.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_portfolio_returns(
    log_returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Compute daily portfolio log returns as a weighted sum.

    For log returns, this is an approximation (exact for simple returns):
      r_portfolio ≈ Σᵢ wᵢ × rᵢ

    For small returns, this approximation is accurate. For large returns
    (e.g., daily moves > 5%), there is a small Jensen's inequality effect.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns per asset.
    weights : dict
        Asset weights.

    Returns
    -------
    pd.Series
        Daily portfolio log returns.
    """
    assets = [a for a in weights if a in log_returns.columns]
    w = np.array([weights[a] for a in assets])
    returns_matrix = log_returns[assets].values
    portfolio_returns = returns_matrix @ w

    return pd.Series(portfolio_returns, index=log_returns.index, name="Portfolio_Returns")


def rolling_volatility(
    portfolio_returns: pd.Series,
    short_window: int = 20,
    long_window: int = 60,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Calculate rolling portfolio volatility.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns.
    short_window : int
        Short rolling window (e.g., 20 days).
    long_window : int
        Long rolling window (e.g., 60 days).
    annualization_factor : int
        Trading days per year for annualization (default: 252).

    Returns
    -------
    pd.DataFrame
        Columns: Vol_20D (annualized), Vol_60D (annualized)
    """
    ann_factor = np.sqrt(annualization_factor)

    # Rolling standard deviation of portfolio daily returns
    vol_short = portfolio_returns.rolling(window=short_window, min_periods=max(5, short_window // 2)).std()
    vol_long = portfolio_returns.rolling(window=long_window, min_periods=max(10, long_window // 2)).std()

    # Annualize
    vol_df = pd.DataFrame(
        {
            f"Vol_{short_window}D": vol_short * ann_factor,
            f"Vol_{long_window}D": vol_long * ann_factor,
        },
        index=portfolio_returns.index,
    )

    logger.info(
        f"Rolling volatility computed: {short_window}D and {long_window}D windows, "
        f"latest {short_window}D vol = {vol_short.iloc[-1]:.4f} ({vol_short.iloc[-1] * ann_factor:.2%} annualized)"
    )

    return vol_df


def identify_high_volatility_periods(
    vol_series: pd.Series,
    threshold_pct: float = 80.0,
) -> pd.Series:
    """
    Flag dates as high-volatility regime.

    A date is flagged if its rolling volatility exceeds the
    historical `threshold_pct` percentile of the entire vol series.

    Parameters
    ----------
    vol_series : pd.Series
        Rolling volatility series (e.g., 20-day annualized).
    threshold_pct : float
        Percentile threshold (e.g., 80.0 for 80th percentile).

    Returns
    -------
    pd.Series
        Boolean series: True = high volatility period.
    """
    threshold = float(np.nanpercentile(vol_series.dropna().values, threshold_pct))
    high_vol = vol_series > threshold

    n_high = high_vol.sum()
    logger.info(
        f"High-volatility regime: threshold={threshold:.4f} ({threshold_pct:.0f}th pct), "
        f"{n_high} days flagged ({n_high / len(vol_series):.1%} of total)"
    )

    return high_vol


def compute_asset_volatilities(
    log_returns: pd.DataFrame,
    window: int = 60,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Calculate rolling (or point-in-time) annualized volatility per asset.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns per asset.
    window : int
        Lookback window for volatility estimate.
    annualization_factor : int
        Trading days per year.

    Returns
    -------
    pd.Series
        Current annualized volatility per asset (using last `window` obs).
    """
    recent_returns = log_returns.tail(window)
    ann_factor = np.sqrt(annualization_factor)
    vols = recent_returns.std() * ann_factor
    vols.name = "Annualized_Volatility"
    return vols


def compute_risk_contributions(
    log_returns: pd.DataFrame,
    weights: Dict[str, float],
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Compute approximate risk contributions using component volatility.

    METHODOLOGY:
    For a portfolio with weights w and asset covariance matrix Σ:
      Portfolio variance: σ²_P = w' Σ w
      Portfolio std dev: σ_P = sqrt(w' Σ w)

    Marginal contribution of asset i to portfolio variance:
      ∂σ²_P / ∂wᵢ = 2 × (Σw)ᵢ

    Component (absolute) contribution to portfolio std dev:
      CC_i = wᵢ × (Σw)ᵢ / σ_P

    This satisfies: Σᵢ CC_i = σ_P (Euler's theorem for homogeneous functions)

    Parameters
    ----------
    log_returns : pd.DataFrame
        Historical log returns.
    weights : dict
        Portfolio weights.
    annualization_factor : int
        For annualizing the results.

    Returns
    -------
    pd.DataFrame
        One row per asset with weight, standalone vol, and component contribution.
    """
    assets = [a for a in weights if a in log_returns.columns]
    w = np.array([weights[a] for a in assets])

    # Sample covariance matrix
    cov_matrix = log_returns[assets].cov().values

    # Annualized covariance
    cov_matrix_ann = cov_matrix * annualization_factor

    # Portfolio variance and standard deviation
    port_var = float(w @ cov_matrix_ann @ w)
    port_std = np.sqrt(port_var) if port_var > 0 else 0.0

    # Marginal contribution: (Σw)_i
    marginal_var = cov_matrix_ann @ w

    # Component contribution: w_i × (Σw)_i / σ_P
    if port_std > 0:
        component_contributions = w * marginal_var / port_std
    else:
        component_contributions = np.zeros_like(w)

    # Standalone volatility
    standalone_vols = np.sqrt(np.diag(cov_matrix_ann))

    rows = []
    for i, asset in enumerate(assets):
        rows.append({
            "asset": asset,
            "weight": float(weights[asset]),
            "standalone_vol_ann": float(standalone_vols[i]),
            "component_vol_contribution": float(component_contributions[i]),
            "pct_risk_contribution": float(
                component_contributions[i] / port_std
            ) if port_std > 0 else None,
        })

    result = pd.DataFrame(rows)
    logger.info(f"Risk contributions computed for {len(assets)} assets. Portfolio σ = {port_std:.4f}")
    return result


def get_scenario_analysis(
    log_returns: pd.DataFrame,
    portfolio_pnl_1d: pd.Series,
) -> list:
    """
    Extract predefined historical scenarios for the dashboard.

    Identifies:
    - Worst single day
    - Worst 5-day period
    - Best single day
    - Most recent major drawdown

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    portfolio_pnl_1d : pd.Series
        1-day portfolio P&L series.

    Returns
    -------
    list of dict
        Each scenario includes date, shocks, and P&L.
    """
    scenarios = []

    if len(portfolio_pnl_1d) == 0:
        return scenarios

    # Worst single day
    worst_day_idx = portfolio_pnl_1d.idxmin()
    scenarios.append(
        _build_scenario(
            "Worst Single Day",
            worst_day_idx,
            log_returns,
            portfolio_pnl_1d,
        )
    )

    # Best single day
    best_day_idx = portfolio_pnl_1d.idxmax()
    scenarios.append(
        _build_scenario(
            "Best Single Day",
            best_day_idx,
            log_returns,
            portfolio_pnl_1d,
        )
    )

    # Worst 5-day rolling period
    if len(portfolio_pnl_1d) >= 5:
        rolling_5d = portfolio_pnl_1d.rolling(5).sum()
        worst_5d_end = rolling_5d.idxmin()
        scenarios.append(
            _build_scenario(
                "Worst 5-Day Window (end date)",
                worst_5d_end,
                log_returns,
                portfolio_pnl_1d,
                cumulative_5d=float(rolling_5d[worst_5d_end]),
            )
        )

    # Highest volatility day (largest absolute move)
    abs_pnl = portfolio_pnl_1d.abs()
    max_vol_day = abs_pnl.idxmax()
    scenarios.append(
        _build_scenario(
            "Largest Absolute Move",
            max_vol_day,
            log_returns,
            portfolio_pnl_1d,
        )
    )

    return scenarios


def _build_scenario(
    name: str,
    date,
    log_returns: pd.DataFrame,
    portfolio_pnl: pd.Series,
    cumulative_5d: Optional[float] = None,
) -> dict:
    """Build a scenario record for the dashboard."""
    scenario = {"scenario_name": name, "date": str(date.date())}

    if date in log_returns.index:
        for col in log_returns.columns:
            scenario[f"{col}_shock_pct"] = float(log_returns.loc[date, col] * 100)

    if date in portfolio_pnl.index:
        scenario["portfolio_pnl"] = float(portfolio_pnl[date])

    if cumulative_5d is not None:
        scenario["cumulative_5d_pnl"] = cumulative_5d

    return scenario
