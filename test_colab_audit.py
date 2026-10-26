"""Check pairing and uncertainty units in the saved-result audit."""
import os
import unittest
from unittest.mock import patch

import pandas as pd

from audit_colab_results import paired_forecast
from model_comparison import redact_secret
from colab_experiment import collect_forecast_summary
from pathlib import Path


class AuditTests(unittest.TestCase):
    def test_seed_repetition_does_not_increase_week_count_or_change_interval(self):
        a = pd.DataFrame({"seed": [1, 1], "season": ["2022-23"]*2, "event": [9, 10],
                          "player_key": ["a", "a"], "fixture": [1, 2],
                          "total_points": [2., 4.], "prediction": [0., 1.],
                          "minutes6": [90., 90.], "rate60": [1., 1.]})
        b = a.assign(prediction=[1., 3.])
        first = paired_forecast(a, b, 200)
        repeated = paired_forecast(pd.concat([a, a.assign(seed=2)]),
                                   pd.concat([b, b.assign(seed=2)]), 200)
        for column in ["rmse_gain", "rmse_low", "rmse_high", "mae_gain", "gameweeks"]:
            self.assertEqual(first[column].tolist(), repeated[column].tolist())
        self.assertAlmostEqual(first.mae_gain.iloc[0], 1.5)

    def test_partial_forecast_pair_is_rejected(self):
        d = pd.DataFrame({"seed": [1, 1], "season": ["s", "s"], "event": [9, 10],
                          "player_key": ["p", "p"], "fixture": [1, 2]})
        with self.assertRaisesRegex(ValueError, "identical row coverage"):
            paired_forecast(d, d.head(1), 20)

    def test_traceback_token_is_redacted(self):
        with patch.dict(os.environ, {"TABPFN_TOKEN": "test-secret"}):
            self.assertEqual(redact_secret("error test-secret"), "error <redacted>")

    def test_partial_results_are_labelled_incomplete(self):
        # A single season is insufficient for the planned two-season comparison.
        run = dict(status="ok", run_id="s1:0:1:baseline:ridge:direct", season="s1",
                   context_size=0, seed=1, feature_spec="baseline", model="ridge",
                   approach="direct", all={"rmse": 2.})
        import json
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            phase = Path(folder) / "phases" / "test"
            phase.mkdir(parents=True)
            (phase / "summary.json").write_text(json.dumps({
                "dataset": {"test_seasons": ["s1", "s2"], "seeds": [1]}, "runs": [run]}))
            result = collect_forecast_summary(Path(folder), ["test"])
        self.assertFalse(result.complete_coverage.iloc[0])
        self.assertEqual(result.completed_cells.iloc[0], 1)
        self.assertEqual(result.expected_cells.iloc[0], 2)


if __name__ == "__main__":
    unittest.main()
