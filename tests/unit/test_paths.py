import unittest
from datetime import date
from pathlib import Path

from alpha_research.paths import DataPaths


class PathTests(unittest.TestCase):
    def test_raw_archive_path(self) -> None:
        paths = DataPaths(Path("data"))

        archive = paths.raw_file(
            dataset="trades",
            exchange="binance",
            market="perp",
            symbol="BTCUSDT",
            day=date(2024, 1, 1),
            suffix=".zip",
        )

        self.assertEqual(
            archive.as_posix(),
            (
                "data/raw/exchange=binance/market=perp/dataset=trades/"
                "symbol=BTCUSDT/date=2024-01-01/"
                "BTCUSDT-trades-2024-01-01.zip"
            ),
        )
