"""Order-flow feature placeholder module."""

from __future__ import annotations

from alpha_research.features.common import FeatureSpec

L1_IMBALANCE = FeatureSpec(
    name="l1_imbalance",
    required_columns=("bid_size", "ask_size"),
    lookback_seconds=0,
)
