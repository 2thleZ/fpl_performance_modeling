"""Checks for blank and double gameweek candidate reconstruction."""
import unittest

import pandas as pd

from candidate_universe_audit import aggregate_scheduled_rows, expand_seen_pool


class CandidateUniverseTests(unittest.TestCase):
    def test_double_collapses_and_blank_player_remains(self):
        raw = pd.DataFrame({
            "element": [1, 1], "name": ["A", "A"], "position": ["MID", "MID"],
            "team": [1, 1], "value": [75, 75], "total_points": [3, 5],
            "fixture": [10, 11], "own_team_id": [1, 1],
        })
        scheduled = aggregate_scheduled_rows(raw)
        seen = pd.DataFrame({"element": [1, 2], "name": ["A", "B"],
                             "position": ["MID", "FWD"], "team": [1, 2],
                             "price": [75, 65]})
        result = expand_seen_pool(seen, scheduled, {1: 2}).set_index("element")
        self.assertEqual(result.loc[1, "scheduled_fixtures"], 2)
        self.assertEqual(result.loc[1, "actual_points"], 8)
        self.assertEqual(result.loc[2, "scheduled_fixtures"], 0)
        self.assertEqual(result.loc[2, "actual_points"], 0)


if __name__ == "__main__":
    unittest.main()
