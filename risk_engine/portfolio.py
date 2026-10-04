"""
risk_engine/portfolio.py
------------------------
Portfolio construction and position management.

The portfolio is defined by:
  - A set of asset weights (must sum to 1.0)
  - A total notional value (e.g., $10,000,000)
  - A mapping from internal names to market data columns

Position value calculation:
  - Position_i = weight_i × portfolio_notional
  - The number of units held is implied by the current price:
      units_i = Position_i / current_price_i

Portfolio P&L under a scenario:
  For historical simulation, we apply historical log returns
  to the current portfolio positions.

  P&L_t = sum_i [ Position_i × (exp(r_{i,t}) - 1) ]

  where r_{i,t} is the log return of asset i on historical day t.

  This is equivalent to:
  - Keeping position sizes fixed in dollar terms.
  - Applying the percentage shock to each position.

  This is a "mark-to-market" P&L calculation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Represents a multi-asset portfolio for risk engine calculations.

    Attributes
    ----------
    weights : dict
        {asset_name: weight} where weights sum to 1.0.
    notional : float
        Total portfolio value in USD.
    asset_classes : dict
        {asset_name: asset_class_label} for reporting.
    asset_names : dict
        {asset_name: human_readable_name} for display.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        notional: float,
        asset_classes: Optional[Dict[str, str]] = None,
        asset_names: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize portfolio.

        Parameters
        ----------
        weights : dict
            {asset_name: weight}. Must sum to approximately 1.0.
        notional : float
            Total portfolio notional value in USD.
        asset_classes : dict, optional
            {asset_name: "Equity" | "FX" | "Fixed Income"}
        asset_names : dict, optional
            {asset_name: display_name}
        """
        self.weights = weights
        self.notional = notional
        self.asset_classes = asset_classes or {}
        self.asset_names = asset_names or {k: k for k in weights}

        self._validate()

    def _validate(self) -> None:
        """Validate portfolio configuration."""
        # Check weight sum
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Portfolio weights sum to {total_weight:.4f}, not 1.0. "
                "Consider normalizing."
            )

        # Check for negative weights (short positions)
        negatives = {k: v for k, v in self.weights.items() if v < 0}
        if negatives:
            logger.warning(f"Negative weights (short positions): {negatives}")

        logger.info(
            f"Portfolio initialized: {len(self.weights)} assets, "
            f"notional = ${self.notional:,.0f}"
        )

    @property
    def assets(self) -> List[str]:
        """Return list of asset names."""
        return list(self.weights.keys())

    def position_values(self) -> Dict[str, float]:
        """
        Calculate USD position values.

        Returns
        -------
        dict
            {asset_name: position_value_USD}
        """
        return {
            asset: self.weights[asset] * self.notional
            for asset in self.assets
        }

    def position_weights_series(self) -> pd.Series:
        """Return weights as a pandas Series."""
        return pd.Series(self.weights)

    def position_values_series(self) -> pd.Series:
        """Return position values as a pandas Series."""
        return pd.Series(self.position_values())

    def get_composition_report(self, prices: Optional[pd.DataFrame] = None) -> List[dict]:
        """
        Generate portfolio composition for the dashboard.

        Parameters
        ----------
        prices : pd.DataFrame, optional
            Current prices. Used to display per-unit holdings if provided.

        Returns
        -------
        list of dict
            One dict per asset with composition details.
        """
        report = []
        pos_values = self.position_values()

        for asset in self.assets:
            entry = {
                "asset": asset,
                "display_name": self.asset_names.get(asset, asset),
                "asset_class": self.asset_classes.get(asset, "Unknown"),
                "weight": self.weights[asset],
                "weight_pct": self.weights[asset] * 100,
                "position_value_usd": pos_values[asset],
            }

            if prices is not None and asset in prices.columns:
                latest_price = float(prices[asset].iloc[-1])
                entry["latest_price"] = latest_price
                entry["units"] = pos_values[asset] / latest_price if latest_price != 0 else 0
            else:
                entry["latest_price"] = None
                entry["units"] = None

            report.append(entry)

        return report


def from_config(config: dict) -> Portfolio:
    """
    Construct a Portfolio object from the YAML configuration dictionary.

    Parameters
    ----------
    config : dict
        Parsed config.yaml content.

    Returns
    -------
    Portfolio
    """
    portfolio_cfg = config["portfolio"]

    return Portfolio(
        weights=portfolio_cfg["weights"],
        notional=portfolio_cfg["notional"],
        asset_classes=portfolio_cfg.get("asset_classes", {}),
        asset_names=portfolio_cfg.get("asset_names", {}),
    )
