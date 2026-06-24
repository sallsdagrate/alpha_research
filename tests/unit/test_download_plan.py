import unittest
from datetime import date
from pathlib import Path

from alpha_research.config import DataConfig
from alpha_research.data.download import binance_archive_url, plan_downloads


class DownloadPlanTests(unittest.TestCase):
    def test_binance_archive_url_for_spot_trade_day(self) -> None:
        url = binance_archive_url("spot", "trades", "BTCUSDT", date(2024, 1, 1))

        self.assertEqual(
            url,
            (
                "https://data.binance.vision/data/spot/daily/trades/BTCUSDT/"
                "BTCUSDT-trades-2024-01-01.zip"
            ),
        )

    def test_plan_downloads_has_one_request_per_market_symbol_day(self) -> None:
        config = DataConfig.from_file(Path("configs/data/binance_core.yaml"))

        requests = plan_downloads(config)
        self.assertEqual(len(requests), 12)
        self.assertTrue(requests[0].destination.as_posix().startswith("data/raw/"))
        self.assertTrue(requests[0].id.startswith("raw.binance."))
