"""
backend/api.py
--------------
FastAPI route handlers for the Market Risk Engine API.

All endpoints return data computed by the Python risk engine.
No values are hard-coded. The engine is initialized once at startup
and cached in memory.

Data flow:
  Browser → fetch() → FastAPI → Risk Engine → JSON → JavaScript → Charts
"""

import logging
from typing import Any, Dict

import numpy as np

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io
import csv

logger = logging.getLogger(__name__)

router = APIRouter()

# Engine state — populated at startup by app.py
_engine_data: Dict[str, Any] = {}


def set_engine_data(data: Dict[str, Any]) -> None:
    """Called by app.py after engine computation to populate the cache."""
    global _engine_data
    _engine_data = data
    logger.info("Engine data loaded into API cache.")


def _require_engine() -> Dict[str, Any]:
    """Raise 503 if engine hasn't been initialized yet."""
    if not _engine_data:
        raise HTTPException(status_code=503, detail="Risk engine not yet initialized. Please wait.")
    return _engine_data


# ─────────────────────────────────────────────────────────────
# GET /api/summary
# ─────────────────────────────────────────────────────────────

@router.get("/summary")
def get_summary() -> Dict[str, Any]:
    """
    Top-level risk summary for the dashboard KPI cards.

    Returns current VaR, ES, violation statistics, and engine status.
    All values are computed from the actual risk engine.
    """
    d = _require_engine()
    return {
        "portfolio_value": d["portfolio_notional"],
        "var_1d_99": d["var_1d"],
        "es_1d_99": d["es_1d"],
        "var_10d_99": d["var_10d"],
        "es_10d_99": d["es_10d"],
        "var_10d_sqrt_approx": d["var_10d_sqrt_approx"],
        "violations": d["backtest"]["summary"]["n_violations"],
        "observations": d["backtest"]["summary"]["n_observations"],
        "violation_rate": d["backtest"]["summary"]["violation_rate"],
        "expected_violation_rate": d["backtest"]["summary"]["expected_violation_rate"],
        "as_of_date": d["as_of_date"],
        "data_status": "Loaded (Yahoo Finance)" if not d.get("is_synthetic") else "Loaded (Synthetic Fallback)",
        "model_status": "Active",
        "confidence_level": d["confidence_level"],
        "historical_window": d["historical_window"],
    }


# ─────────────────────────────────────────────────────────────
# GET /api/portfolio
# ─────────────────────────────────────────────────────────────

@router.get("/portfolio")
def get_portfolio() -> Dict[str, Any]:
    """Portfolio composition and current position values."""
    d = _require_engine()
    return {
        "portfolio_value": d["portfolio_notional"],
        "assets": d["portfolio_composition"],
        "as_of_date": d["as_of_date"],
        "risk_contributions": d.get("risk_contributions", []),
    }


# ─────────────────────────────────────────────────────────────
# GET /api/pnl
# ─────────────────────────────────────────────────────────────

@router.get("/pnl")
def get_pnl() -> Dict[str, Any]:
    """
    Historical P&L time series for the main backtesting chart.

    Returns realized P&L, VaR, ES, and violation flag per date.
    """
    d = _require_engine()
    records = d["backtesting_timeline"]
    return {
        "records": records,
        "total_records": len(records),
    }


# ─────────────────────────────────────────────────────────────
# GET /api/risk
# ─────────────────────────────────────────────────────────────

@router.get("/risk")
def get_risk() -> Dict[str, Any]:
    """
    Current risk metrics: VaR, ES, 10-day metrics, residual risk add-on.
    """
    d = _require_engine()
    return {
        "var_1d_99": d["var_1d"],
        "es_1d_99": d["es_1d"],
        "var_10d_99": d["var_10d"],
        "es_10d_99": d["es_10d"],
        "var_10d_sqrt_approx": d["var_10d_sqrt_approx"],
        "confidence_level": d["confidence_level"],
        "historical_window": d["historical_window"],
        "as_of_date": d["as_of_date"],
        "distribution_summary": d["distribution_summary"],
        "residual_risk": d["residual_risk"],
    }


# ─────────────────────────────────────────────────────────────
# GET /api/backtest
# ─────────────────────────────────────────────────────────────

@router.get("/backtest")
def get_backtest() -> Dict[str, Any]:
    """
    Full backtesting results including Kupiec POF and Christoffersen tests.
    """
    d = _require_engine()
    return d["backtest"]


# ─────────────────────────────────────────────────────────────
# GET /api/violations
# ─────────────────────────────────────────────────────────────

@router.get("/violations")
def get_violations() -> Dict[str, Any]:
    """VaR violation records with dates and excess losses."""
    d = _require_engine()
    violations = d["backtest"].get("violations", [])
    return {
        "violations": violations,
        "total_violations": len(violations),
    }


# ─────────────────────────────────────────────────────────────
# GET /api/volatility
# ─────────────────────────────────────────────────────────────

@router.get("/volatility")
def get_volatility() -> Dict[str, Any]:
    """Rolling volatility time series for the volatility regime chart."""
    d = _require_engine()
    return {
        "records": d["volatility_records"],
        "current_vol_20d": d["current_vol_20d"],
        "current_vol_60d": d["current_vol_60d"],
        "high_vol_threshold": d.get("high_vol_threshold"),
    }


# ─────────────────────────────────────────────────────────────
# GET /api/distribution
# ─────────────────────────────────────────────────────────────

@router.get("/distribution")
def get_distribution() -> Dict[str, Any]:
    """
    P&L distribution for the histogram chart.

    Returns binned histogram data and VaR/ES markers.
    """
    d = _require_engine()
    pnl_values = d["pnl_distribution_1d"]

    # Compute histogram
    pnl_array = np.array(pnl_values)
    n_bins = 60
    counts, bin_edges = np.histogram(pnl_array, bins=n_bins)

    return {
        "pnl_values": [float(x) for x in pnl_values],
        "var_99": d["var_1d"],
        "es_99": d["es_1d"],
        "histogram_bins": [float(x) for x in bin_edges],
        "histogram_counts": [int(x) for x in counts],
        "n_scenarios": len(pnl_values),
    }


# ─────────────────────────────────────────────────────────────
# GET /api/config
# ─────────────────────────────────────────────────────────────

@router.get("/config")
def get_config() -> Dict[str, Any]:
    """Engine configuration parameters."""
    d = _require_engine()
    return {
        "portfolio_notional": d["portfolio_notional"],
        "confidence_level": d["confidence_level"],
        "historical_window": d["historical_window"],
        "holding_period": d["holding_period"],
        "assets": d["assets"],
        "weights": d["weights"],
        "data_source": "Yahoo Finance (public data)" if not d.get("is_synthetic") else "Synthetic (fallback)",
        "data_start": d.get("data_start", "N/A"),
    }


# ─────────────────────────────────────────────────────────────
# GET /api/scenarios
# ─────────────────────────────────────────────────────────────

@router.get("/scenarios")
def get_scenarios() -> Dict[str, Any]:
    """Predefined historical scenario analysis."""
    d = _require_engine()
    return {
        "scenarios": d.get("scenarios", []),
    }


# ─────────────────────────────────────────────────────────────
# GET /api/export/pnl
# ─────────────────────────────────────────────────────────────

@router.get("/export/pnl")
def export_pnl_csv() -> StreamingResponse:
    """
    Download P&L backtesting data as CSV.

    Columns: Date, Realized_PnL, VaR_1D, ES_1D, Violation
    """
    d = _require_engine()
    records = d["backtesting_timeline"]

    output = io.StringIO()
    fieldnames = ["date", "realized_pnl", "var_1d", "es_1d", "violation"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        writer.writerow({k: rec.get(k, "") for k in fieldnames})

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pnl_backtest.csv"},
    )


@router.get("/export/risk_report")
def export_risk_report_csv() -> StreamingResponse:
    """Download current risk report as CSV."""
    d = _require_engine()

    rows = [
        {"Metric": "Portfolio Value", "Value": d["portfolio_notional"], "Unit": "USD"},
        {"Metric": "1-Day VaR (99%)", "Value": d["var_1d"], "Unit": "USD"},
        {"Metric": "1-Day ES (99%)", "Value": d["es_1d"], "Unit": "USD"},
        {"Metric": "10-Day VaR (99%)", "Value": d["var_10d"], "Unit": "USD"},
        {"Metric": "10-Day ES (99%)", "Value": d["es_10d"], "Unit": "USD"},
        {"Metric": "10-Day VaR Sqrt Approx", "Value": d["var_10d_sqrt_approx"], "Unit": "USD"},
        {"Metric": "VaR Violations", "Value": d["backtest"]["summary"]["n_violations"], "Unit": "count"},
        {"Metric": "Violation Rate", "Value": d["backtest"]["summary"]["violation_rate"], "Unit": "%"},
        {"Metric": "Kupiec p-value", "Value": d["backtest"]["kupiec_pof"].get("p_value", ""), "Unit": ""},
        {"Metric": "Christoffersen p-value", "Value": d["backtest"]["christoffersen_independence"].get("p_value_independence", ""), "Unit": ""},
        {"Metric": "Residual Risk Add-On", "Value": d["residual_risk"].get("addon_usd", 0), "Unit": "USD"},
        {"Metric": "As of Date", "Value": d["as_of_date"], "Unit": ""},
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Metric", "Value", "Unit"])
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=risk_report.csv"},
    )
