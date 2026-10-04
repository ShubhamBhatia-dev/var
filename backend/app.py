"""
backend/app.py
--------------
FastAPI application entry point.

This module:
1. Initializes the risk engine at startup.
2. Runs the full risk calculation pipeline.
3. Caches results in memory.
4. Serves the frontend HTML dashboard.
5. Registers all API routes.

Startup sequence:
  1. Load configuration from config.yaml.
  2. Download/load market data.
  3. Clean and preprocess data.
  4. Build portfolio.
  5. Run historical simulation.
  6. Calculate VaR and ES.
  7. Run backtesting.
  8. Compute volatility metrics.
  9. Compute residual risk add-on.
  10. Load all results into API cache.
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from risk_engine import (
    data_loader,
    preprocessing,
    portfolio as portfolio_module,
    historical_simulation,
    var as var_module,
    expected_shortfall,
    backtesting,
    volatility as vol_module,
    residual_risk,
)
from backend.api import router as api_router, set_engine_data

# ─────────────────────────────────────────────────────────────
# Configure Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Configuration Loader
# ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load and return config.yaml from project root."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


# ─────────────────────────────────────────────────────────────
# Main Risk Engine Pipeline
# ─────────────────────────────────────────────────────────────

def run_risk_engine(config: dict) -> dict:
    """
    Execute the complete risk engine pipeline.

    Returns a dictionary of all computed risk metrics and data
    ready for the API layer.

    Parameters
    ----------
    config : dict
        Parsed configuration from config.yaml.

    Returns
    -------
    dict
        All risk engine outputs.
    """
    logger.info("=" * 60)
    logger.info("STARTING RISK ENGINE PIPELINE")
    logger.info("=" * 60)

    # ── 1. Extract configuration ─────────────────────────────
    port_cfg = config["portfolio"]
    risk_cfg = config["risk"]
    data_cfg = config["data"]
    vol_cfg = config["volatility"]
    bt_cfg = config["backtesting"]
    rr_cfg = config["residual_risk"]

    portfolio_notional = float(port_cfg["notional"])
    confidence_level = float(risk_cfg["confidence_level"])
    historical_window = int(risk_cfg["historical_window"])
    holding_period = int(risk_cfg["holding_period"])
    min_bt_obs = int(bt_cfg["min_observations"])
    alpha = float(bt_cfg["alpha"])

    logger.info(
        f"Config: notional=${portfolio_notional:,.0f}, "
        f"confidence={confidence_level:.0%}, window={historical_window}d"
    )

    # ── 2. Load market data ──────────────────────────────────
    logger.info("Step 1/9: Loading market data...")
    tickers = data_cfg["yf_tickers"]
    start_date = data_cfg["start_date"]
    source = data_cfg.get("source", "yfinance")
    fallback_dir = data_cfg.get("fallback_dir", "data/raw")

    raw_prices = data_loader.load_market_data(
        tickers=tickers,
        start_date=start_date,
        fallback_dir=fallback_dir,
        source=source,
    )

    # Detect if data is synthetic
    is_synthetic = any(
        "SYNTHETIC" in col for col in raw_prices.columns
    )

    # ── 3. Clean and preprocess ──────────────────────────────
    logger.info("Step 2/9: Cleaning and preprocessing data...")
    clean_prices = preprocessing.clean_prices(raw_prices)
    data_quality = preprocessing.get_data_quality_report(clean_prices)

    aligned_prices, log_returns = preprocessing.align_to_common_index(
        clean_prices, window=historical_window
    )

    # Save data for offline use
    data_loader.save_raw_data(raw_prices, fallback_dir)

    as_of_date = str(aligned_prices.index[-1].date())
    logger.info(f"Data aligned: {len(aligned_prices)} observations, as of {as_of_date}")

    # ── 4. Build portfolio ───────────────────────────────────
    logger.info("Step 3/9: Building portfolio...")
    portfolio = portfolio_module.Portfolio(
        weights=port_cfg["weights"],
        notional=portfolio_notional,
        asset_classes=port_cfg.get("asset_classes", {}),
        asset_names=port_cfg.get("asset_names", {}),
    )

    portfolio_composition = portfolio.get_composition_report(aligned_prices)

    # ── 5. Historical Simulation ─────────────────────────────
    logger.info("Step 4/9: Running historical simulation (1-day)...")
    pnl_1d = historical_simulation.compute_portfolio_pnl_1d(
        log_returns=log_returns,
        portfolio=portfolio,
        window=historical_window,
    )

    logger.info("Step 5/9: Running historical simulation (10-day)...")
    pnl_10d = historical_simulation.compute_portfolio_pnl_10d(
        log_returns=log_returns,
        portfolio=portfolio,
        window=historical_window,
        holding_period=holding_period,
    )

    # ── 6. VaR and ES ────────────────────────────────────────
    logger.info("Step 6/9: Calculating VaR and Expected Shortfall...")
    var_1d = var_module.calculate_var(pnl_1d.values, confidence_level)
    es_1d = expected_shortfall.calculate_es(pnl_1d.values, confidence_level)

    var_10d = var_module.calculate_var(pnl_10d.values, confidence_level)
    es_10d = expected_shortfall.calculate_es(pnl_10d.values, confidence_level)

    var_10d_sqrt_approx = var_module.calculate_var_sqrt_of_time(var_1d, holding_period)

    es_decomp = expected_shortfall.es_decomposition(pnl_1d.values, confidence_level)

    logger.info(
        f"1D: VaR=${var_1d:,.0f}, ES=${es_1d:,.0f} | "
        f"10D: VaR=${var_10d:,.0f}, ES=${es_10d:,.0f}"
    )

    # ── 7. Rolling backtesting ───────────────────────────────
    logger.info("Step 7/9: Running rolling VaR backtesting...")
    rolling_results = historical_simulation.compute_rolling_var_es(
        log_returns=log_returns,
        portfolio=portfolio,
        window=historical_window,
        confidence=confidence_level,
        min_window=min_bt_obs,
    )

    backtest_result = backtesting.run_backtest(
        rolling_results=rolling_results,
        confidence_level=confidence_level,
        alpha=alpha,
    )

    backtesting_timeline = backtesting.get_violation_timeline(rolling_results)

    # ── 8. Volatility analysis ───────────────────────────────
    logger.info("Step 8/9: Computing volatility metrics...")
    portfolio_returns = vol_module.compute_portfolio_returns(
        log_returns=log_returns,
        weights=port_cfg["weights"],
    )

    short_win = vol_cfg["short_window"]
    long_win = vol_cfg["long_window"]
    ann_factor = vol_cfg["annualization_factor"]

    vol_df = vol_module.rolling_volatility(
        portfolio_returns=portfolio_returns,
        short_window=short_win,
        long_window=long_win,
        annualization_factor=ann_factor,
    )

    current_vol_20d = float(vol_df[f"Vol_{short_win}D"].dropna().iloc[-1])
    current_vol_60d = float(vol_df[f"Vol_{long_win}D"].dropna().iloc[-1])

    high_vol_flags = vol_module.identify_high_volatility_periods(
        vol_df[f"Vol_{short_win}D"],
        threshold_pct=rr_cfg.get("volatility_threshold_pct", 80.0),
    )

    high_vol_threshold_val = float(
        np.nanpercentile(
            vol_df[f"Vol_{short_win}D"].dropna().values,
            rr_cfg.get("volatility_threshold_pct", 80.0)
        )
    )

    # Build volatility records for API
    volatility_records = []
    for date in vol_df.index:
        v20 = vol_df.loc[date, f"Vol_{short_win}D"]
        v60 = vol_df.loc[date, f"Vol_{long_win}D"]
        is_hv = bool(high_vol_flags.get(date, False))
        volatility_records.append({
            "date": str(date.date()),
            "vol_20d": float(v20) if not pd.isna(v20) else None,
            "vol_60d": float(v60) if not pd.isna(v60) else None,
            "is_high_vol": is_hv,
        })

    # Risk contributions
    try:
        risk_contrib_df = vol_module.compute_risk_contributions(
            log_returns=log_returns.tail(historical_window),
            weights=port_cfg["weights"],
            annualization_factor=ann_factor,
        )
        risk_contributions = risk_contrib_df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"Risk contribution calculation failed: {e}")
        risk_contributions = []

    # Scenario analysis
    scenarios = vol_module.get_scenario_analysis(
        log_returns=log_returns,
        portfolio_pnl_1d=pnl_1d,
    )

    # ── 9. Residual risk add-on ──────────────────────────────
    logger.info("Step 9/9: Computing residual risk add-on...")
    residual_result = residual_risk.compute_residual_risk_addon(
        current_vol_20d=current_vol_20d,
        historical_vol_series=vol_df[f"Vol_{short_win}D"],
        portfolio_notional=portfolio_notional,
        addon_multiplier=float(rr_cfg.get("addon_multiplier", 2.5)),
        threshold_percentile=float(rr_cfg.get("volatility_threshold_pct", 80.0)),
        vol_window=short_win,
        annualization_factor=ann_factor,
    )

    # ── Compile distribution summary ─────────────────────────
    distribution_summary = {
        "n_scenarios": int(len(pnl_1d)),
        "mean_pnl": float(pnl_1d.mean()),
        "std_pnl": float(pnl_1d.std()),
        "min_pnl": float(pnl_1d.min()),
        "max_pnl": float(pnl_1d.max()),
        "skewness": float(pd.Series(pnl_1d.values).skew()),
        "kurtosis": float(pd.Series(pnl_1d.values).kurt()),
        "pct_1": float(np.percentile(pnl_1d.values, 1)),
        "pct_5": float(np.percentile(pnl_1d.values, 5)),
        "pct_95": float(np.percentile(pnl_1d.values, 95)),
        "pct_99": float(np.percentile(pnl_1d.values, 99)),
        "es_to_var_ratio": es_decomp.get("es_to_var_ratio"),
    }

    # ── Package all data for API ─────────────────────────────
    engine_data = {
        # Core metrics
        "portfolio_notional": portfolio_notional,
        "var_1d": float(var_1d),
        "es_1d": float(es_1d),
        "var_10d": float(var_10d),
        "es_10d": float(es_10d),
        "var_10d_sqrt_approx": float(var_10d_sqrt_approx),
        "confidence_level": confidence_level,
        "historical_window": historical_window,
        "holding_period": holding_period,
        "as_of_date": as_of_date,
        "is_synthetic": is_synthetic,
        # Portfolio
        "portfolio_composition": portfolio_composition,
        "assets": list(portfolio.weights.keys()),
        "weights": dict(portfolio.weights),
        "risk_contributions": risk_contributions,
        # P&L distribution
        "pnl_distribution_1d": [float(x) for x in pnl_1d.values],
        "pnl_distribution_10d": [float(x) for x in pnl_10d.values],
        # Backtesting
        "backtest": backtest_result,
        "backtesting_timeline": backtesting_timeline,
        # Volatility
        "volatility_records": volatility_records,
        "current_vol_20d": current_vol_20d,
        "current_vol_60d": current_vol_60d,
        "high_vol_threshold": float(high_vol_threshold_val),
        # Distribution summary
        "distribution_summary": distribution_summary,
        # Residual risk
        "residual_risk": residual_result,
        # Scenarios
        "scenarios": scenarios,
        # Data metadata
        "data_start": data_cfg["start_date"],
        "data_quality": data_quality,
    }

    logger.info("=" * 60)
    logger.info("RISK ENGINE PIPELINE COMPLETE")
    logger.info(
        f"VaR 1D: ${var_1d:,.0f} | ES 1D: ${es_1d:,.0f} | "
        f"Violations: {backtest_result['summary']['n_violations']}"
    )
    logger.info("=" * 60)

    return engine_data


# ─────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize risk engine on startup."""
    logger.info("Initializing risk engine...")
    try:
        config = load_config()
        engine_data = run_risk_engine(config)
        set_engine_data(engine_data)
        logger.info("Risk engine ready.")
    except Exception as e:
        logger.error(f"Risk engine initialization failed: {e}", exc_info=True)
        logger.warning("API will return 503 until engine is initialized.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Multi-Asset Market Risk & Tail-Loss Analytics Engine",
    description=(
        "Historical Simulation VaR, Expected Shortfall, Kupiec POF, "
        "Christoffersen Independence test, rolling backtesting, "
        "and volatility regime analysis for a multi-asset portfolio."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes under /api
app.include_router(api_router, prefix="/api")

# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
else:
    logger.warning("Frontend directory not found. Dashboard will not be served.")
