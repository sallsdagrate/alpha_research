"""Filesystem path helpers for the local research lake."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Resolve project data paths from a configurable data root."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def normalized(self) -> Path:
        return self.root / "normalized"

    @property
    def curated(self) -> Path:
        return self.root / "curated"

    @property
    def features(self) -> Path:
        return self.root / "features"

    def ensure(self) -> None:
        for path in [self.raw, self.normalized, self.curated, self.features]:
            path.mkdir(parents=True, exist_ok=True)

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

    def normalized_partition(
        self,
        *,
        dataset: str,
        exchange: str,
        market: str,
        symbol: str,
        day: date,
    ) -> Path:
        return (
            self.normalized
            / dataset
            / f"exchange={exchange}"
            / f"market={market}"
            / f"symbol={symbol}"
            / f"date={day.isoformat()}"
        )

    def curated_dataset(self, dataset: str, exchange: str, symbol: str) -> Path:
        return self.curated / dataset / f"exchange={exchange}" / f"symbol={symbol}"


def manifest_path(path: Path) -> Path:
    """Return the sidecar manifest path for a file or directory output."""

    if path.suffix:
        return path.with_suffix(path.suffix + ".manifest.json")
    return path / "_manifest.json"
