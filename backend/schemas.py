"""
backend/schemas.py
------------------
Pydantic-style response schemas for the FastAPI endpoints.
Used for documentation and type validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SummaryResponse(BaseModel):
    portfolio_value: float
    var_1d_99: float
    es_1d_99: float
    var_10d_99: float
    es_10d_99: float
    var_10d_sqrt_approx: float
    violations: int
    observations: int
    violation_rate: float
    expected_violation_rate: float
    as_of_date: str
    data_status: str
    model_status: str


class PortfolioAsset(BaseModel):
    asset: str
    display_name: str
    asset_class: str
    weight: float
    weight_pct: float
    position_value_usd: float
    latest_price: Optional[float]


class PortfolioResponse(BaseModel):
    portfolio_value: float
    assets: List[PortfolioAsset]
    as_of_date: str


class PnLRecord(BaseModel):
    date: str
    realized_pnl: float
    var_1d: float
    es_1d: float
    violation: int


class PnLResponse(BaseModel):
    records: List[PnLRecord]
    total_records: int


class RiskResponse(BaseModel):
    var_1d_99: float
    es_1d_99: float
    var_10d_99: float
    es_10d_99: float
    var_10d_sqrt_approx: float
    confidence_level: float
    historical_window: int
    as_of_date: str
    distribution_summary: Dict[str, Any]
    residual_risk: Dict[str, Any]


class BacktestResponse(BaseModel):
    summary: Dict[str, Any]
    kupiec_pof: Dict[str, Any]
    christoffersen_independence: Dict[str, Any]
    conditional_coverage: Dict[str, Any]


class ViolationRecord(BaseModel):
    date: str
    realized_pnl: float
    var_1d: float
    loss_in_excess_of_var: float


class ViolationsResponse(BaseModel):
    violations: List[ViolationRecord]
    total_violations: int


class VolatilityRecord(BaseModel):
    date: str
    vol_20d: Optional[float]
    vol_60d: Optional[float]
    is_high_vol: Optional[bool]


class VolatilityResponse(BaseModel):
    records: List[VolatilityRecord]
    current_vol_20d: Optional[float]
    current_vol_60d: Optional[float]


class DistributionResponse(BaseModel):
    pnl_values: List[float]
    var_99: float
    es_99: float
    histogram_bins: List[float]
    histogram_counts: List[int]


class ConfigResponse(BaseModel):
    portfolio_notional: float
    confidence_level: float
    historical_window: int
    holding_period: int
    assets: List[str]
    weights: Dict[str, float]
    data_source: str
    data_start: str
