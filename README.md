# Multi-Asset Market Risk & Tail-Loss Analytics Engine

A complete, end-to-end quantitative market-risk system demonstrating Historical Simulation Value at Risk (VaR), Expected Shortfall (ES), statistical backtesting (Kupiec POF, Christoffersen Independence), rolling risk monitoring, and volatility regime analysis — built in Python with a professional dark-terminal HTML/CSS/JavaScript dashboard.

> **Important:** This is an educational portfolio project demonstrating quantitative finance methodology. It is NOT a regulatory-certified system, does NOT produce regulatory capital numbers, and does NOT use proprietary or Bloomberg/Refinitiv data. All market data is sourced from Yahoo Finance (publicly available) or synthetic fallback data.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  BROWSER DASHBOARD                  │
│           HTML / CSS / Vanilla JavaScript           │
│              Chart.js visualizations                │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP fetch()
                        ▼
┌─────────────────────────────────────────────────────┐
│               FASTAPI REST API                      │
│  /api/summary  /api/pnl  /api/risk  /api/backtest   │
│  /api/volatility  /api/distribution  /api/portfolio │
│  /api/scenarios  /api/config  /api/export/*         │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               PYTHON RISK ENGINE                    │
│                                                     │
│  data_loader.py         → Yahoo Finance / CSV       │
│  preprocessing.py       → Clean, align, returns     │
│  portfolio.py           → Position values           │
│  historical_simulation.py → 1D / 10D P&L dist.     │
│  var.py                 → 99% Historical VaR        │
│  expected_shortfall.py  → 99% ES                    │
│  backtesting.py         → Rolling VaR backtest      │
│  kupiec.py              → POF likelihood ratio test │
│  christoffersen.py      → Independence test / CC    │
│  volatility.py          → Rolling vol, regime       │
│  residual_risk.py       → Simplified add-on (educ.) │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                MARKET DATA                          │
│  Yahoo Finance (primary) / CSV fallback             │
│  SPY, QQQ (Equity) | EURUSD=X, GBPUSD=X (FX)       │
│  IEF, TLT (US Treasury ETF proxies)                 │
└─────────────────────────────────────────────────────┘
```

---

## Why Historical Simulation?

Historical Simulation (HS) is a non-parametric VaR methodology that uses the actual empirical distribution of historical returns rather than assuming a specific distribution shape.

**Advantages over parametric approaches:**
- **No distributional assumption** — captures fat tails, skewness, and non-normality automatically
- **Preserves joint dependencies** — correlations, tail dependencies, and co-movements are embedded in historical data
- **Transparent scenarios** — every scenario corresponds to an actual observed market day, making it explainable
- **Regulatory acceptance** — HS is widely used in internal models for market risk capital

**Limitations compared to Monte Carlo:**
- Limited by sample size (only as many scenarios as historical observations)
- Assumes the past is representative of the future (regime changes can invalidate this)
- Cannot easily simulate scenarios more extreme than historical experience
- Look-back period choice is subjective and consequential

---

## Mathematical Methodology

### 1. Historical P&L (Per Scenario)

For each historical date $t$ and asset $i$, the log return is:

$$r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)$$

Portfolio P&L under scenario $t$:

$$\text{P\&L}_t = \sum_i w_i \cdot N \cdot \left(e^{r_{i,t}} - 1\right)$$

where $w_i$ is the weight and $N$ is the portfolio notional.

### 2. Value at Risk (99%)

$$\text{VaR}_{99} = -Q_{0.01}\left(\text{P\&L distribution}\right)$$

VaR is expressed as a **positive loss amount**.

### 3. Expected Shortfall (99%)

$$\text{ES}_{99} = -\mathbb{E}\left[\text{P\&L} \mid \text{P\&L} \leq Q_{0.01}\right]$$

ES is the average loss in the worst 1% of scenarios.

### 4. Kupiec POF Likelihood Ratio

$$\text{LR}_{\text{POF}} = -2\left[\ln\left((1-p)^{N-x} p^x\right) - \ln\left((1-\hat{p})^{N-x} \hat{p}^x\right)\right]$$

where:
- $N$ = number of observations
- $x$ = number of VaR violations
- $p = 1 - \text{confidence}$ = theoretical violation probability
- $\hat{p} = x/N$ = observed violation rate

$\text{LR}_{\text{POF}} \sim \chi^2(1)$ under $H_0$

### 5. Christoffersen Independence Test

Define transition counts $N_{ij}$ (state $i$ → state $j$, where 1 = violation):

$$\pi_0 = \frac{N_{01}}{N_{00} + N_{01}}, \quad \pi_1 = \frac{N_{11}}{N_{10} + N_{11}}, \quad \pi = \frac{N_{01} + N_{11}}{N_{00} + N_{01} + N_{10} + N_{11}}$$

$$\text{LR}_{\text{IND}} = -2\left[\ln L(\pi) - \ln L(\pi_0, \pi_1)\right] \sim \chi^2(1)$$

### 6. Conditional Coverage

$$\text{LR}_{\text{CC}} = \text{LR}_{\text{POF}} + \text{LR}_{\text{IND}} \sim \chi^2(2)$$

---

## Project Structure

```
market-risk-engine/
│
├── backend/
│   ├── app.py             FastAPI application + engine pipeline
│   ├── api.py             REST route handlers
│   └── schemas.py         Pydantic response models
│
├── risk_engine/
│   ├── data_loader.py     Yahoo Finance / CSV data loading
│   ├── preprocessing.py   Data cleaning, log returns
│   ├── portfolio.py       Portfolio construction
│   ├── historical_simulation.py  1D / 10D P&L, rolling backtest
│   ├── var.py             Historical VaR calculation
│   ├── expected_shortfall.py     ES calculation
│   ├── backtesting.py     Backtest orchestration
│   ├── kupiec.py          Kupiec POF test
│   ├── christoffersen.py  Christoffersen Independence + CC
│   ├── volatility.py      Rolling vol, risk contributions
│   └── residual_risk.py   Simplified add-on (educational)
│
├── data/
│   ├── raw/               CSV files (auto-saved after download)
│   └── processed/         (reserved)
│
├── frontend/
│   ├── index.html         Dashboard HTML
│   ├── styles.css         Dark terminal styling
│   └── app.js             Chart.js + API integration
│
├── tests/
│   ├── test_var.py
│   ├── test_es.py
│   ├── test_kupiec.py
│   └── test_christoffersen.py
│
├── config.yaml            All configurable parameters
├── requirements.txt       Python dependencies
├── run.py                 Entry point
└── README.md
```

---

## Installation

```bash
# 1. Clone or download the project
cd market-risk-engine

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
# Start the server (data download + risk calculation + API + dashboard)
python run.py

# Or use uvicorn directly
uvicorn backend.app:app --port 8000

# For development with auto-reload
uvicorn backend.app:app --reload --port 8000
```

The dashboard opens at: **http://localhost:8000**

API documentation: **http://localhost:8000/docs**

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_var.py -v
pytest tests/test_kupiec.py -v

# Run with coverage
pytest tests/ --tb=short
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/summary` | Top-level KPI summary |
| `GET /api/portfolio` | Portfolio composition + risk contributions |
| `GET /api/pnl` | Rolling P&L time series |
| `GET /api/risk` | VaR, ES, distribution stats, residual risk |
| `GET /api/backtest` | Kupiec, Christoffersen, CC test results |
| `GET /api/violations` | VaR exception records |
| `GET /api/volatility` | Rolling volatility time series |
| `GET /api/distribution` | P&L histogram data |
| `GET /api/scenarios` | Historical scenario analysis |
| `GET /api/config` | Engine configuration |
| `GET /api/export/pnl` | Download P&L data as CSV |
| `GET /api/export/risk_report` | Download risk report as CSV |

---

## Configuration

Edit `config.yaml` to customize:

```yaml
portfolio:
  notional: 10_000_000    # Portfolio size in USD
  weights:
    SPY: 0.30             # Adjust weights (must sum to 1.0)
    QQQ: 0.20
    EURUSD: 0.15
    GBPUSD: 0.10
    IEF: 0.15
    TLT: 0.10

risk:
  confidence_level: 0.99  # VaR confidence
  historical_window: 500  # Lookback days
  holding_period: 10      # Multi-day horizon
```

---

## Limitations

This project explicitly acknowledges the following limitations:

1. **Historical dependence** — assumes the future resembles the past; regime changes invalidate this
2. **Limited sample** — 500 observations give only ~5 tail observations for 99% VaR
3. **Linear instruments only** — delta-equivalent approximation; options require full revaluation
4. **Simplified fixed income** — Treasury ETFs used as bond proxies; no term structure modeling
5. **No intraday risk** — daily mark-to-market only; no intraday gap risk
6. **Simplified residual risk** — the add-on formula is educational; NOT a regulatory RRAO
7. **No liquidity risk** — positions assumed fully liquid at mark prices
8. **No credit risk** — this is a market risk engine only
9. **No operational risk** — not included in scope

---

## Data Sources

- **Equities:** Yahoo Finance via `yfinance` (SPY, QQQ)
- **FX:** Yahoo Finance spot rates (EURUSD=X, GBPUSD=X)
- **Fixed Income:** Yahoo Finance (IEF = iShares 7-10Y Treasury ETF, TLT = iShares 20+Y Treasury ETF)

Treasury ETFs are used as fixed-income market proxies. They capture interest rate sensitivity and duration risk, but do not replicate individual bond cash flows or yield curve movements directly.

If Yahoo Finance is unavailable, the engine falls back to CSV files in `data/raw/` or generates synthetic demonstration data (clearly labeled as simulated).

---

