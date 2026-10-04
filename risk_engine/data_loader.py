"""
risk_engine/data_loader.py
--------------------------
Loads historical market data from Yahoo Finance (primary) or CSV fallback.

Data sources:
  - Yahoo Finance via yfinance library (publicly available data)
  - Local CSV files in data/raw/ (fallback when internet is unavailable)

Important: We do NOT claim data came from Bloomberg, Refinitiv, or any
proprietary source. All data is from publicly available Yahoo Finance.

FX Tickers: Yahoo Finance uses the convention EURUSD=X for FX pairs.
Fixed Income: We use Treasury ETFs (IEF, TLT) as interest-rate proxies.
  IEF tracks 7-10 year US Treasuries; TLT tracks 20+ year US Treasuries.
  These capture duration risk and credit-risk-free return dynamics.
  Note: ETF prices include management fees and do not perfectly replicate
  direct bond yields, but they serve as adequate market proxies for a
  portfolio risk engine.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_market_data(
    tickers: Dict[str, str],
    start_date: str,
    end_date: Optional[str] = None,
    fallback_dir: str = "data/raw",
    source: str = "yfinance",
) -> pd.DataFrame:
    """
    Load adjusted closing prices for all portfolio instruments.

    Parameters
    ----------
    tickers : dict
        Mapping from internal asset name -> Yahoo Finance ticker symbol.
        e.g. {"SPY": "SPY", "EURUSD": "EURUSD=X"}
    start_date : str
        ISO date string (YYYY-MM-DD) for data start.
    end_date : str, optional
        ISO date string for data end. Defaults to today.
    fallback_dir : str
        Directory containing CSV fallback files.
    source : str
        "yfinance" to download, "csv" to force CSV fallback.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and one column per internal asset name.
        Values are adjusted closing prices.
    """
    if source == "yfinance":
        try:
            logger.info("Attempting to download data from Yahoo Finance...")
            prices = _download_yfinance(tickers, start_date, end_date)
            logger.info(f"Successfully downloaded {len(prices)} rows from Yahoo Finance.")
            return prices
        except Exception as e:
            logger.warning(f"Yahoo Finance download failed: {e}. Falling back to CSV.")
            return _load_csv_fallback(tickers, fallback_dir)
    else:
        logger.info("Using CSV fallback data source.")
        return _load_csv_fallback(tickers, fallback_dir)


def _download_yfinance(
    tickers: Dict[str, str],
    start_date: str,
    end_date: Optional[str],
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Uses yfinance's download function. The 'Auto Adjust' feature applies
    corporate action adjustments (splits, dividends) automatically.

    Parameters
    ----------
    tickers : dict
        {internal_name: yf_ticker} mapping.
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str, optional
        End date (YYYY-MM-DD). None means today.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices indexed by date.
    """
    import yfinance as yf

    yf_tickers = list(tickers.values())
    internal_names = list(tickers.keys())

    # Download all tickers at once for efficiency
    # auto_adjust=True ensures prices are split/dividend adjusted
    raw = yf.download(
        tickers=yf_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns multi-level columns when multiple tickers requested
    # We extract the "Close" column for each ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker — columns are flat
        prices = raw[["Close"]].rename(columns={"Close": yf_tickers[0]})

    # Rename columns from YF ticker names to our internal names
    ticker_to_internal = {v: k for k, v in tickers.items()}
    prices = prices.rename(columns=ticker_to_internal)

    # Keep only columns we asked for (some may be missing)
    available = [name for name in internal_names if name in prices.columns]
    if len(available) < len(internal_names):
        missing = set(internal_names) - set(available)
        logger.warning(f"Missing tickers after download: {missing}")
    prices = prices[available]

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    return prices


def _load_csv_fallback(
    tickers: Dict[str, str],
    fallback_dir: str,
) -> pd.DataFrame:
    """
    Load price data from local CSV files.

    Expected format: One CSV per asset, named {internal_name}.csv
    Each CSV must have columns: Date, Close

    If no CSV files exist, generates synthetic fallback data for
    offline demonstration purposes. Generated data is clearly labeled.

    Parameters
    ----------
    tickers : dict
        {internal_name: yf_ticker} mapping.
    fallback_dir : str
        Directory containing CSV files.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices indexed by date.
    """
    fallback_path = Path(fallback_dir)
    all_prices = {}

    for internal_name in tickers.keys():
        csv_file = fallback_path / f"{internal_name}.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file, parse_dates=["Date"], index_col="Date")
                if "Close" in df.columns:
                    all_prices[internal_name] = df["Close"]
                    logger.info(f"Loaded {internal_name} from {csv_file}")
                else:
                    logger.warning(f"{csv_file} missing 'Close' column, skipping.")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        else:
            logger.warning(f"No CSV found for {internal_name} at {csv_file}")

    if len(all_prices) < len(tickers):
        # Generate synthetic fallback for any missing assets
        logger.warning(
            "Some assets unavailable from CSV. Generating synthetic fallback data. "
            "THIS DATA IS SIMULATED — for demonstration only."
        )
        missing = set(tickers.keys()) - set(all_prices.keys())
        synthetic = _generate_synthetic_prices(list(missing))
        all_prices.update(synthetic)

    if not all_prices:
        raise RuntimeError("No data available from CSV or synthetic fallback.")

    prices = pd.DataFrame(all_prices)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices.sort_index(inplace=True)

    return prices


def _generate_synthetic_prices(
    assets: list,
    n_days: int = 1500,
    seed: int = 42,
) -> Dict[str, pd.Series]:
    """
    Generate synthetic price series for offline demonstration.

    Uses geometric Brownian motion with reasonable volatility parameters.
    Data is clearly labeled as simulated in all outputs.

    NOTE: This data is NOT real market data. It is a mathematical
    simulation for testing and offline demonstration only.

    Parameters
    ----------
    assets : list
        List of asset names to generate.
    n_days : int
        Number of trading days to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {asset_name: pd.Series} with simulated prices.
    """
    rng = np.random.default_rng(seed)

    # Approximate starting prices and daily volatilities
    params = {
        "SPY":    {"start": 300.0, "mu": 0.0003, "sigma": 0.0120},
        "QQQ":    {"start": 250.0, "mu": 0.0004, "sigma": 0.0150},
        "EURUSD": {"start": 1.10,  "mu": 0.0000, "sigma": 0.0040},
        "GBPUSD": {"start": 1.25,  "mu": 0.0000, "sigma": 0.0050},
        "IEF":    {"start": 100.0, "mu": 0.0001, "sigma": 0.0030},
        "TLT":    {"start": 140.0, "mu": 0.0001, "sigma": 0.0060},
    }

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    result = {}

    for asset in assets:
        p = params.get(asset, {"start": 100.0, "mu": 0.0001, "sigma": 0.0100})
        daily_returns = rng.normal(p["mu"], p["sigma"], n_days)
        prices = p["start"] * np.exp(np.cumsum(daily_returns))
        result[asset] = pd.Series(prices, index=dates, name=asset)
        logger.warning(f"Generated SYNTHETIC prices for {asset} (not real market data)")

    return result


def save_raw_data(prices: pd.DataFrame, output_dir: str = "data/raw") -> None:
    """
    Save downloaded price data to CSV files for offline use.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame to save.
    output_dir : str
        Directory to save CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for col in prices.columns:
        file_path = output_path / f"{col}.csv"
        prices[[col]].rename(columns={col: "Close"}).to_csv(file_path)
        logger.info(f"Saved {col} to {file_path}")
