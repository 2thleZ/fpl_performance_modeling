"""Checks for matched likely-starter selection analysis."""
import unittest

import pandas as pd

from selection_analysis import (
    collapse_player_gameweeks,
    compare_fixture_selections,
    select_top_players,
    summarize_selection_comparisons,
)


class SelectionAnalysisTests(unittest.TestCase):
    def test_double_gameweeks_are_summed_before_matched_selection(self):
        rows = []
        for spec in ("baseline", "interaction"):
            for player, price, prediction, points in (
                ("a", 6.5, 5.0 if spec == "baseline" else 1.0, 2.0),
                ("b", 6.7, 4.0, 3.0),
                ("c", 7.0, 3.0, 4.0),
                ("d", 7.5, 1.0 if spec == "baseline" else 6.0, 8.0),
            ):
                fixtures = [1, 2] if player == "a" else [1]
                for fixture in fixtures:
                    rows.append({
                        "run_id": f"2025-26:5000:1:{spec}:ridge:direct",
                        "context_size": 5000, "seed": 1, "feature_spec": spec,
                        "model": "ridge", "approach": "direct", "season": "2025-26",
                        "event": 10, "player_key": player, "fixture": fixture,
                        "position": "MID", "price_m": price, "minutes": 90,
                        "minutes6": 80, "rate60": 1.0,
                        "total_points": points, "prediction": prediction,
                        "play_probability": float("nan"),
                        "conditional_prediction": float("nan"),
                    })
        predictions = pd.DataFrame(rows)
        collapsed = collapse_player_gameweeks(predictions)
        a = collapsed.loc[(collapsed.player_key == "a") &
                          (collapsed.feature_spec == "baseline")].iloc[0]
        self.assertEqual(a.fixture_count, 2)
        self.assertEqual(a.total_points, 4.0)
        self.assertEqual(a.prediction, 10.0)

        selected = select_top_players(collapsed, top_k=3)
        comparisons = compare_fixture_selections(selected, top_k=3)
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons.changed_picks.iloc[0], 1)
        self.assertEqual(comparisons.actual_points_gain.iloc[0], 4.0)
        summary = summarize_selection_comparisons(comparisons, reps=100, seed=1)
        self.assertAlmostEqual(summary.gain_per_pick.iloc[0], 4 / 3)


if __name__ == "__main__":
    unittest.main()
