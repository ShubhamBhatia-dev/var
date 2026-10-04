"""
risk_engine/preprocessing.py
-----------------------------
Data cleaning and preprocessing pipeline.

Steps:
1. Remove duplicate dates.
2. Forward-fill missing observations (e.g., FX holidays vs equity holidays).
3. Align all assets to a common trading-date index.
4. Drop rows where too many assets have missing data.
5. Calculate daily log returns.

Design philosophy:
  We use log returns (ln(P_t / P_{t-1})) rather than simple returns because:
  - Log returns are additive across time (compounding is handled correctly).
  - Log returns are more symmetric and better-behaved statistically.
  - For historical simulation, the difference is small but log returns
    are the standard in quantitative risk management.

For FX instruments:
  Yahoo Finance returns FX spot rates directly as prices.
  We treat them identically to equity prices in this engine.
  The portfolio assumes USD-base with the FX pair expressed as USD per unit.

Assumptions:
  - Missing data is forward-filled (last-observation-carried-forward).
  - A maximum of forward_fill_limit days of consecutive missing data is allowed.
  - Dates with more than max_missing_pct of assets missing are dropped.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_prices(
    prices: pd.DataFrame,
    forward_fill_limit: int = 5,
    max_missing_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Clean raw price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw price DataFrame (DatetimeIndex, one column per asset).
    forward_fill_limit : int
        Maximum consecutive NaNs to forward-fill.
    max_missing_pct : float
        Drop rows where this fraction or more of assets are missing.

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame.
    """
    logger.info(f"Cleaning prices: input shape {prices.shape}")

    # Step 1: Ensure datetime index and sort
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    # Step 2: Remove duplicate dates (keep last occurrence)
    n_dupes = prices.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"Removing {n_dupes} duplicate date(s).")
        prices = prices[~prices.index.duplicated(keep="last")]

    # Step 3: Remove weekends (should not appear in market data, but just in case)
    prices = prices[prices.index.dayofweek < 5]

    # Step 4: Replace zeros with NaN (zero price indicates missing data)
    prices = prices.replace(0, np.nan)

    # Step 5: Forward-fill up to forward_fill_limit consecutive NaNs
    # This handles cases where some markets are closed while others trade
    prices = prices.ffill(limit=forward_fill_limit)

    # Step 6: Drop rows where too many assets are missing
    n_assets = prices.shape[1]
    missing_threshold = int(np.ceil(max_missing_pct * n_assets))
    rows_with_many_missing = prices.isnull().sum(axis=1) >= missing_threshold
    n_dropped = rows_with_many_missing.sum()
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} rows with >= {missing_threshold} missing assets.")
        prices = prices[~rows_with_many_missing]

    # Step 7: Drop any remaining rows with any NaN (after forward fill)
    prices = prices.dropna()

    logger.info(f"Cleaned prices: output shape {prices.shape}")
    logger.info(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily log returns for all assets.

    Log return formula:
        r_t = ln(P_t / P_{t-1})

    where P_t is the adjusted closing price at time t.

    Log returns are preferred because:
    - They are time-additive: multi-period return = sum of single-period log returns
    - They are more normally distributed than simple returns
    - They cannot fall below -1 (avoiding mathematical issues)

    Parameters
    ----------
    prices : pd.DataFrame
        Clean price DataFrame.

    Returns
    -------
    pd.DataFrame
        Daily log returns. First row is NaN and will be dropped.
    """
    # np.log(P_t / P_{t-1}) == np.log(P_t) - np.log(P_{t-1})
    returns = np.log(prices / prices.shift(1))

    # Drop the first row which is always NaN (no prior price)
    returns = returns.dropna()

    logger.info(f"Computed log returns: shape {returns.shape}")

    return returns


def align_to_common_index(
    prices: pd.DataFrame,
    window: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure all assets share the same trading-date index.

    After alignment:
    - All assets have prices on exactly the same set of dates.
    - We take the most recent `window` days plus extra for return calculation.

    Parameters
    ----------
    prices : pd.DataFrame
        Clean prices.
    window : int
        Historical simulation window (number of trading days).

    Returns
    -------
    Tuple of (aligned_prices, log_returns)
    """
    # Use inner join — only dates where ALL assets have prices
    # Forward filling in clean_prices should have handled most gaps already
    aligned = prices.dropna()

    if len(aligned) < window + 1:
        logger.warning(
            f"Only {len(aligned)} clean observations available. "
            f"Requested window: {window}. Proceeding with available data."
        )

    log_returns = compute_log_returns(aligned)

    return aligned, log_returns


def get_data_quality_report(prices: pd.DataFrame) -> dict:
    """
    Generate a data quality summary for the dashboard.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame to inspect.

    Returns
    -------
    dict
        Data quality metrics per asset.
    """
    report = {
        "n_observations": int(len(prices)),
        "date_start": str(prices.index[0].date()) if len(prices) > 0 else None,
        "date_end": str(prices.index[-1].date()) if len(prices) > 0 else None,
        "assets": {},
    }

    for col in prices.columns:
        series = prices[col]
        report["assets"][col] = {
            "n_missing": int(series.isnull().sum()),
            "pct_missing": float(series.isnull().mean()),
            "min_price": float(series.min()),
            "max_price": float(series.max()),
            "latest_price": float(series.iloc[-1]) if len(series) > 0 else None,
        }

    return report
