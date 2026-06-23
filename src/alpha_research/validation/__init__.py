"""Research validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChronologicalSplit:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def split_index(length: int, train_fraction: float) -> int:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if length < 2:
        raise ValueError("Need at least two rows for a train/test split")
    return max(1, min(length - 1, int(length * train_fraction)))
