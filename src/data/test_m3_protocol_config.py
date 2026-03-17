from __future__ import annotations

import unittest
from pathlib import Path

from src.data.loader import load_yaml


class M3ProtocolConfigTests(unittest.TestCase):
    def test_official_m3_protocol_config_has_required_fields(self) -> None:
        protocol_path = Path("config/evaluation/m3_protocol.yaml")
        self.assertTrue(protocol_path.exists(), f"Missing protocol file: {protocol_path}")

        data = load_yaml(protocol_path)

        self.assertIn("protocol", data)
        self.assertEqual(data["protocol"].get("milestone"), "M3")

        for key in ["data", "universe", "benchmark", "portfolio", "execution", "m3_requirements"]:
            self.assertIn(key, data)

        self.assertTrue(data["data"].get("start_date"))
        self.assertTrue(data["data"].get("end_date"))
        self.assertTrue(data["universe"].get("symbols"))
        self.assertTrue(data["benchmark"].get("benchmark_symbol"))
        self.assertIsNotNone(data["portfolio"].get("initial_cash"))

        requirements = data["m3_requirements"]
        self.assertTrue(requirements.get("required_outputs"))
        self.assertTrue(requirements.get("required_metrics"))


if __name__ == "__main__":
    unittest.main()
