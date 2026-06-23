"""Cross-venue feature placeholder module."""

from __future__ import annotations

from alpha_research.features.common import FeatureSpec

VENUE_RETURN_LEAD = FeatureSpec(
    name="venue_return_lead",
    required_columns=("leader_return",),
    lookback_seconds=1,
)
