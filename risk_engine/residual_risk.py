"""
risk_engine/residual_risk.py
-----------------------------
Simplified Residual-Risk Add-On.

⚠️  IMPORTANT DISCLAIMER:
==========================
This is an EDUCATIONAL APPROXIMATION only.

This module does NOT implement:
- The Basel FRTB Residual Risk Add-On (RRAO) methodology
- Any regulatory capital calculation
- A production-grade risk measurement system

It demonstrates the conceptual principle that historical VaR may
underestimate risk during elevated volatility regimes because the
historical sample may not contain analogous stress periods.

CONCEPTUAL BASIS:
=================
Historical Simulation VaR relies on the assumption that the historical
return sample is representative of the future risk environment.

During volatility regime changes (e.g., the start of a market crisis),
the current 20-day realized volatility may substantially exceed the
historical average captured in the VaR lookback window.

A simple intuition: If current market volatility is 40% annualized
but the historical VaR window reflects an average of 15% volatility,
the VaR estimate may understate the true risk.

SIMPLIFIED ADD-ON FORMULA:
===========================
Let:
  vol_current  = current 20-day rolling portfolio volatility (annualized)
  vol_pct80    = 80th percentile of historical 20-day vol series
  portfolio_V  = portfolio notional

If vol_current > vol_pct80:
  excess_vol = vol_current - vol_pct80
  addon = k × excess_vol × portfolio_V × (1/sqrt(252))  [for 1-day scaling]

where k is a configurable multiplier (default: 2.5).

The daily scaling factor 1/sqrt(252) converts the annualized excess
volatility to a daily equivalent.

This formula is entirely illustrative. The multiplier k is not derived
from any regulatory or theoretical framework.

LIMITATIONS:
============
1. The threshold (80th percentile) is arbitrary.
2. The multiplier k is not theoretically grounded.
3. This does not capture correlation changes during stress.
4. This is not a substitute for proper stress testing.
5. This should NOT be used for capital allocation decisions.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_residual_risk_addon(
    current_vol_20d: float,
    historical_vol_series: pd.Series,
    portfolio_notional: float,
    addon_multiplier: float = 2.5,
    threshold_percentile: float = 80.0,
    vol_window: int = 20,
    annualization_factor: int = 252,
) -> dict:
    """
    Compute the simplified residual-risk add-on.

    ⚠️  This is an educational approximation — NOT regulatory capital.

    Parameters
    ----------
    current_vol_20d : float
        Current 20-day annualized portfolio volatility.
    historical_vol_series : pd.Series
        Full time series of historical 20-day annualized volatility.
    portfolio_notional : float
        Total portfolio value in USD.
    addon_multiplier : float
        Scaling factor applied to excess volatility (configurable).
    threshold_percentile : float
        Percentile of historical vol above which we apply the add-on.
    vol_window : int
        Volatility window used (for reporting clarity).
    annualization_factor : int
        Trading days per year for daily scaling.

    Returns
    -------
    dict
        Add-on components with full explanation.
    """
    clean_vol_series = historical_vol_series.dropna()

    if len(clean_vol_series) == 0:
        logger.warning("Empty volatility series for residual risk add-on.")
        return _empty_addon()

    # Historical threshold
    vol_threshold = float(np.percentile(clean_vol_series.values, threshold_percentile))

    # Daily scaling: convert annualized excess vol to 1-day equivalent
    daily_scale = 1.0 / np.sqrt(annualization_factor)

    # Compute add-on
    if current_vol_20d > vol_threshold:
        excess_vol = current_vol_20d - vol_threshold
        addon = addon_multiplier * excess_vol * portfolio_notional * daily_scale
        regime = "ELEVATED"
    else:
        excess_vol = 0.0
        addon = 0.0
        regime = "NORMAL"

    result = {
        "current_vol_20d_annualized": float(current_vol_20d),
        "vol_threshold_annualized": float(vol_threshold),
        "threshold_percentile": float(threshold_percentile),
        "excess_vol": float(excess_vol),
        "addon_multiplier": float(addon_multiplier),
        "portfolio_notional": float(portfolio_notional),
        "addon_usd": float(addon),
        "volatility_regime": regime,
        "disclaimer": (
            "SIMPLIFIED RESIDUAL-RISK ADD-ON — EDUCATIONAL APPROXIMATION ONLY. "
            "This is NOT a regulatory capital calculation. "
            "Formula: addon = k × excess_vol × notional / sqrt(252) "
            f"where k={addon_multiplier}, threshold={threshold_percentile:.0f}th percentile "
            f"of {vol_window}-day rolling vol."
        ),
    }

    logger.info(
        f"Residual risk add-on: regime={regime}, current_vol={current_vol_20d:.4f}, "
        f"threshold={vol_threshold:.4f}, addon=${addon:,.0f}"
    )

    return result


def get_adjusted_var(
    var_1d: float,
    addon: float,
) -> float:
    """
    Return the VaR adjusted by the residual risk add-on.

    Adjusted estimate = Base VaR + Add-On

    This is not a regulatory-endorsed combination method.
    It is a simple additive approximation for educational purposes.

    Parameters
    ----------
    var_1d : float
        Base 1-day VaR (positive loss amount).
    addon : float
        Residual risk add-on (positive loss amount).

    Returns
    -------
    float
        Adjusted risk estimate (positive loss amount).
    """
    return var_1d + addon


def _empty_addon() -> dict:
    """Return empty add-on result for error cases."""
    return {
        "current_vol_20d_annualized": None,
        "vol_threshold_annualized": None,
        "threshold_percentile": None,
        "excess_vol": 0.0,
        "addon_multiplier": None,
        "portfolio_notional": None,
        "addon_usd": 0.0,
        "volatility_regime": "UNKNOWN",
        "disclaimer": "Add-on could not be computed.",
    }
