from __future__ import annotations

import unittest
import zipfile
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from alpha_research.data import find_trade_archives, load_trades_polars
from alpha_research.paths import DataPaths


class TradeLoaderTests(unittest.TestCase):
    def test_loads_spot_and_perp_archives_directly(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = DataPaths(root)
            spot = paths.raw_file(
                exchange="binance",
                market="spot",
                dataset="trades",
                symbol="BTCUSDT",
                day=date(2024, 1, 1),
                suffix=".zip",
            )
            perp = paths.raw_file(
                exchange="binance",
                market="perp",
                dataset="trades",
                symbol="BTCUSDT",
                day=date(2024, 1, 1),
                suffix=".zip",
            )
            self._write_zip(
                spot,
                "1,42000,0.1,4200,1704067200000,True,True\n",
            )
            self._write_zip(
                perp,
                (
                    "id,price,qty,quote_qty,time,is_buyer_maker\n"
                    "2,42001,0.2,8400.2,1704067200001,false\n"
                ),
            )

            spot_frame = load_trades_polars(
                market="spot",
                symbol="BTCUSDT",
                data_root=root,
            )
            perp_frame = load_trades_polars(
                market="perp",
                symbol="BTCUSDT",
                data_root=root,
            )

            self.assertEqual(spot_frame.height, 1)
            self.assertEqual(perp_frame.height, 1)
            self.assertEqual(spot_frame["market"][0], "spot")
            self.assertEqual(perp_frame["market"][0], "perp")
            self.assertEqual(perp_frame["quantity"][0], 0.2)

    def test_filters_archives_by_date(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = DataPaths(root)
            for day in (date(2024, 1, 1), date(2024, 1, 2)):
                path = paths.raw_file(
                    exchange="binance",
                    market="spot",
                    dataset="trades",
                    symbol="BTCUSDT",
                    day=day,
                    suffix=".zip",
                )
                self._write_zip(path, "1,1,1,1,1,True,True\n")

            archives = find_trade_archives(
                market="spot",
                symbol="BTCUSDT",
                start_date="2024-01-02",
                end_date="2024-01-02",
                data_root=root,
            )

            self.assertEqual(len(archives), 1)
            self.assertIn("date=2024-01-02", archives[0].as_posix())

    @staticmethod
    def _write_zip(path: Path, contents: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("trades.csv", contents)
