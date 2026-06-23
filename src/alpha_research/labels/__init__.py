"""Label builders."""

from __future__ import annotations

from alpha_research.exceptions import MissingDependencyError


def add_forward_returns(frame, *, price_column: str, horizons_seconds: tuple[int, ...]):
    """Add forward percentage-return labels for one-second rows."""

    try:
        import polars as pl
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Label generation requires polars. Install dependencies first."
        ) from exc

    expressions = []
    for horizon in horizons_seconds:
        expressions.append(
            (
                (pl.col(price_column).shift(-horizon) / pl.col(price_column)) - 1.0
            ).alias(f"fwd_return_{horizon}s")
        )
        expressions.append(
            pl.col("bucket_ts").shift(-horizon).alias(f"label_end_ts_{horizon}s")
        )
    return frame.with_columns(expressions)
