import unittest
from unittest.mock import patch

import pandas as pd

from src.data_loader import load_market_data


class TestICTSignal(unittest.TestCase):
    def test_live_market_data_is_preferred_when_available(self):
        live_df = pd.DataFrame([
            {"Open": 2348.0, "High": 2350.0, "Low": 2347.0, "Close": 2349.0, "Volume": 1200},
        ], index=pd.to_datetime(["2026-09-25 10:00:00"]))
        live_df.index.name = "Datetime"
        live_df["Symbol"] = "XAUUSD"

        with patch('src.data_loader.get_live_mt5_data', return_value=live_df), patch('src.data_loader.generate_synthetic_ohlcv', side_effect=AssertionError('Synthetic fallback should not be used when live data is available')):
            result = load_market_data('XAUUSD', '2026-09-25', '2026-09-25', use_live_data=True)
            self.assertEqual(result.iloc[-1]['Close'], 2349.0)
            self.assertFalse(result.empty)

    def test_no_synthetic_fallback_when_live_data_missing(self):
        with patch('src.data_loader.get_live_mt5_data', return_value=None):
            with self.assertRaises(RuntimeError):
                load_market_data('XAUUSD', '2026-09-25', '2026-09-25', use_live_data=True)


if __name__ == '__main__':
    unittest.main()
