"""Filesystem paths for downloaded exchange archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Resolve raw archive paths from a configurable data root."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    def ensure(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)

    def raw_file(
        self,
        *,
        exchange: str,
        market: str,
        dataset: str,
        symbol: str,
        day: date,
        suffix: str,
    ) -> Path:
        return (
            self.raw
            / f"exchange={exchange}"
            / f"market={market}"
            / f"dataset={dataset}"
            / f"symbol={symbol}"
            / f"date={day.isoformat()}"
            / f"{symbol}-{dataset}-{day.isoformat()}{suffix}"
        )
