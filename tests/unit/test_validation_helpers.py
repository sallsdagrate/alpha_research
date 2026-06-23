import unittest

from alpha_research.backtest import CostModel, apply_linear_cost
from alpha_research.validation import split_index


class ValidationHelperTests(unittest.TestCase):
    def test_split_index_keeps_both_sides_non_empty(self) -> None:
        self.assertEqual(split_index(10, 0.7), 7)
        self.assertEqual(split_index(2, 0.9), 1)

    def test_cost_model_round_trip_bps(self) -> None:
        model = CostModel(taker_fee_bps=5.0, spread_bps=1.0, slippage_bps=1.0)

        self.assertEqual(model.round_trip_bps, 12.0)
        self.assertAlmostEqual(
            apply_linear_cost(0.01, turnover=1.0, cost_bps=10.0),
            0.009,
        )
