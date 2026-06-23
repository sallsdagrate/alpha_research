"""Simple execution-cost and metric helpers for research runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    taker_fee_bps: float
    spread_bps: float
    slippage_bps: float

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * self.taker_fee_bps + self.spread_bps + self.slippage_bps


def apply_linear_cost(gross_return: float, turnover: float, cost_bps: float) -> float:
    """Apply linear transaction costs to a return."""

    return gross_return - turnover * cost_bps / 10_000.0
