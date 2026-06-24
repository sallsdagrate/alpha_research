import unittest
from datetime import date
from pathlib import Path

from alpha_research.config import DataConfig


class ConfigTests(unittest.TestCase):
    def test_data_config_loads_committed_yaml(self) -> None:
        config = DataConfig.from_file(Path("configs/data/binance_core.yaml"))

        self.assertEqual(config.data_root, Path("data"))
        self.assertEqual(config.symbols, ("BTCUSDT", "ETHUSDT"))
        self.assertEqual(config.markets, ("spot", "perp"))
        self.assertEqual(config.start_date, date(2024, 1, 1))
        self.assertEqual(config.end_date, date(2024, 1, 3))
