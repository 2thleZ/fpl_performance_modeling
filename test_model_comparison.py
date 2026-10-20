"""Unit checks for the model-comparison protocol."""
import unittest

import numpy as np
import pandas as pd

from model_comparison import (
    likely_starter_mask,
    pooled_fixture_comparisons,
    pooled_model_comparisons,
    regression_metrics,
    sample_training_rows,
)


class ModelComparisonTests(unittest.TestCase):
    def test_likely_starter_uses_only_prior_features(self):
        d = pd.DataFrame({"minutes6": [44, 45, 90], "rate60": [.8, .5, .49],
                          "minutes": [90, 0, 90]})
        self.assertEqual(likely_starter_mask(d).tolist(), [False, True, False])

    def test_negative_predictions_are_scored_without_clipping(self):
        metrics = regression_metrics(np.array([0., 1.]), np.array([-2., 1.]))
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["bias"], -1.0)

    def test_training_sample_is_reproducible_and_unmodified_when_small(self):
        d = pd.DataFrame({"x": range(20)})
        self.assertEqual(sample_training_rows(d, 5, 7).x.tolist(),
                         sample_training_rows(d, 5, 7).x.tolist())
        self.assertEqual(sample_training_rows(d, 50, 7).x.tolist(), list(range(20)))

    def test_pooled_comparisons_match_rows_across_seasons(self):
        rows = []
        for season in ["2022-23", "2023-24"]:
            for spec, model, prediction in [
                ("baseline", "ridge", 0.0),
                ("baseline", "tabicl", 0.5),
                ("elo", "ridge", 0.5),
            ]:
                rows.append({"season": season, "event": 9, "player_key": "p",
                             "fixture": 1, "total_points": 1.0, "prediction": prediction,
                             "run_id": f"{season}:{spec}:{model}", "context_size": 5_000,
                             "seed": 1, "feature_spec": spec, "model": model,
                             "approach": "direct"})
        predictions = pd.DataFrame(rows)
        model_result = pooled_model_comparisons(
            predictions.loc[predictions.feature_spec.eq("baseline")])[0]
        fixture_result = pooled_fixture_comparisons(
            predictions.loc[predictions.model.eq("ridge")])[0]
        self.assertEqual(model_result["gameweeks"], 2)
        self.assertEqual(len(model_result["baseline_runs"]), 2)
        self.assertAlmostEqual(model_result["mae_gain"], 0.5)
        self.assertEqual(fixture_result["gameweeks"], 2)
        self.assertAlmostEqual(fixture_result["rmse_gain"], 0.5)


if __name__ == "__main__":
    unittest.main()
