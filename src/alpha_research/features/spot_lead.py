"""Hypothesis 1 spot-leads-perp features."""

from __future__ import annotations

from alpha_research.exceptions import MissingDependencyError
from alpha_research.features.common import FeatureSpec

SPOT_IMBALANCE_1S = FeatureSpec(
    name="spot_signed_quote_imbalance_1s",
    required_columns=("spot_signed_quote", "spot_quote_notional"),
    lookback_seconds=1,
    availability_column="spot_available_ts",
)


def add_spot_lead_features(frame):
    """Add simple spot order-flow features to a Polars frame/lazy frame."""

    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Feature generation requires polars. Install dependencies first."
        ) from exc

    return frame.with_columns(
        (
            pl.col("spot_signed_quote")
            / pl.when(pl.col("spot_quote_notional") == 0)
            .then(None)
            .otherwise(pl.col("spot_quote_notional"))
        ).alias("spot_signed_quote_imbalance_1s"),
        (pl.col("spot_vwap").pct_change()).alias("spot_return_1s"),
    )
