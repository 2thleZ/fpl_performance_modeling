"""Unit checks for the model-comparison protocol."""
import unittest

import numpy as np
import pandas as pd

from model_comparison import likely_starter_mask, regression_metrics, sample_training_rows


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


if __name__ == "__main__":
    unittest.main()
