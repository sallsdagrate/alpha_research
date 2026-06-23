from datetime import date
from pathlib import Path
import unittest

from alpha_research.paths import DataPaths, manifest_path


class PathTests(unittest.TestCase):
    def test_normalized_partition_path(self) -> None:
        paths = DataPaths(Path("data"))

        partition = paths.normalized_partition(
            dataset="trades",
            exchange="binance",
            market="perp",
            symbol="BTCUSDT",
            day=date(2024, 1, 1),
        )

        self.assertEqual(
            partition.as_posix(),
            (
                "data/normalized/trades/exchange=binance/market=perp/"
                "symbol=BTCUSDT/date=2024-01-01"
            ),
        )

    def test_manifest_path_for_file_and_directory(self) -> None:
        self.assertEqual(manifest_path(Path("x/y.zip")), Path("x/y.zip.manifest.json"))
        self.assertEqual(manifest_path(Path("x/y")), Path("x/y/_manifest.json"))
