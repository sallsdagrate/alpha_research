"""Notebook-friendly loaders for downloaded Binance trade archives."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Literal

import polars as pl

from alpha_research.exceptions import MissingDependencyError
from alpha_research.paths import DataPaths

SPOT_COLUMNS = [
    "trade_id",
    "price",
    "quantity",
    "quote_notional",
    "time_ms",
    "is_buyer_maker",
    "is_best_match",
]

PERP_RENAME = {
    "id": "trade_id",
    "qty": "quantity",
    "quote_qty": "quote_notional",
    "time": "time_ms",
}


def find_trade_archives(
    *,
    market: Literal["spot", "perp"],
    symbol: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    data_root: str | Path = "data",
) -> list[Path]:
    """Find locally downloaded daily trade ZIP files in chronological order."""

    start = _as_date(start_date)
    end = _as_date(end_date)
    if start is not None and end is not None and end < start:
        raise ValueError("end_date must be on or after start_date")

    base = (
        DataPaths(Path(data_root)).raw
        / "exchange=binance"
        / f"market={market}"
        / "dataset=trades"
        / f"symbol={symbol}"
    )
    archives: list[Path] = []
    for path in sorted(base.glob("date=*/*.zip")):
        day = date.fromisoformat(path.parent.name.removeprefix("date="))
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        archives.append(path)
    return archives


def load_trades(
    *,
    market: Literal["spot", "perp"],
    symbol: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    data_root: str | Path = "data",
    engine: Literal["polars", "pandas"] = "polars",
) -> Any:
    """Load raw Binance trades using Polars or pandas."""

    if engine == "polars":
        return load_trades_polars(
            market=market,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_root=data_root,
        )
    if engine == "pandas":
        return load_trades_pandas(
            market=market,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_root=data_root,
        )
    raise ValueError(f"Unsupported engine: {engine}")


def load_trades_polars(
    *,
    market: Literal["spot", "perp"],
    symbol: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    data_root: str | Path = "data",
) -> pl.DataFrame:
    """Load downloaded trade ZIPs into one Polars DataFrame."""

    archives = _require_archives(
        market=market,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_root=data_root,
    )
    frames: list[pl.DataFrame] = []
    for path in archives:
        source = _read_zipped_csv(path)
        if market == "spot":
            frame = pl.read_csv(
                source,
                has_header=False,
                new_columns=SPOT_COLUMNS,
                infer_schema_length=1000,
            )
        else:
            frame = pl.read_csv(
                source,
                has_header=True,
                infer_schema_length=1000,
            ).rename(PERP_RENAME)
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.Boolean).alias("is_best_match")
            )
        frames.append(
            frame.with_columns(
                pl.lit("binance").alias("exchange"),
                pl.lit(market).alias("market"),
                pl.lit(symbol).alias("symbol"),
                pl.lit(_archive_date(path)).cast(pl.Date).alias("source_date"),
            )
        )
    return pl.concat(frames, how="diagonal_relaxed")


def load_trades_pandas(
    *,
    market: Literal["spot", "perp"],
    symbol: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    data_root: str | Path = "data",
):
    """Load downloaded trade ZIPs into one pandas DataFrame."""

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Install pandas to use the pandas loader."
        ) from exc

    archives = _require_archives(
        market=market,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_root=data_root,
    )
    frames: list[Any] = []
    for path in archives:
        if market == "spot":
            frame = pd.read_csv(
                path,
                compression="zip",
                header=None,
                names=SPOT_COLUMNS,
            )
        else:
            frame = pd.read_csv(path, compression="zip").rename(columns=PERP_RENAME)
            frame["is_best_match"] = None
        frame["exchange"] = "binance"
        frame["market"] = market
        frame["symbol"] = symbol
        frame["source_date"] = _archive_date(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _require_archives(**kwargs: Any) -> list[Path]:
    archives = find_trade_archives(**kwargs)
    if not archives:
        raise FileNotFoundError(
            "No downloaded trade archives matched the requested market, symbol, "
            "and date range."
        )
    return archives


def _read_zipped_csv(path: Path) -> io.BytesIO:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {len(names)}")
        return io.BytesIO(archive.read(names[0]))


def _archive_date(path: Path) -> date:
    return date.fromisoformat(path.parent.name.removeprefix("date="))


def _as_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)
