from datetime import date
from pathlib import Path
import unittest

from alpha_research.config import DataConfig, ExperimentConfig


class ConfigTests(unittest.TestCase):
    def test_data_config_loads_committed_yaml(self) -> None:
        config = DataConfig.from_file(Path("configs/data/binance_core.yaml"))

        self.assertEqual(config.data_root, Path("data"))
        self.assertEqual(config.exchange, "binance")
        self.assertEqual(config.symbols, ("BTCUSDT", "ETHUSDT"))
        self.assertEqual(config.markets, ("spot", "perp"))
        self.assertEqual(config.datasets, ("trades",))
        self.assertEqual(config.start_date, date(2024, 1, 1))
        self.assertEqual(config.end_date, date(2024, 1, 3))

    def test_experiment_config_loads_committed_yaml(self) -> None:
        config = ExperimentConfig.from_file(
            Path("configs/experiments/h1_spot_lead_baseline.yaml")
        )

        self.assertEqual(config.hypothesis_id, "h1_spot_leads_perp")
        self.assertEqual(config.symbols, ("BTCUSDT", "ETHUSDT"))
        self.assertEqual(config.horizons_seconds, (1, 5, 30, 60))
