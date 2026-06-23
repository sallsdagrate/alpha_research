"""Funding and basis feature placeholder module."""

from __future__ import annotations

from alpha_research.features.common import FeatureSpec

BASIS_ZSCORE = FeatureSpec(
    name="basis_zscore",
    required_columns=("basis",),
    lookback_seconds=8 * 60 * 60,
)
