"""Common feature metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    required_columns: tuple[str, ...]
    lookback_seconds: int
    availability_column: str = "available_ts"
