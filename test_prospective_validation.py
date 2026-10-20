"""Small timing checks for the prospective fixture analysis."""
import unittest

import pandas as pd

from prospective_validation import strict_elo, deadline_features


class DeadlineTests(unittest.TestCase):
    def test_elo_is_frozen_before_first_gameweek_kickoff(self):
        fixtures = pd.DataFrame([
            (1, 1, "2024-08-01T12:00:00Z", 1, 2, 3, 0),
            (2, 1, "2024-08-03T12:00:00Z", 3, 4, 1, 0),
            (3, 2, "2024-08-10T12:00:00Z", 1, 3, 0, 0),
        ], columns=["id", "event", "kickoff_time", "team_h", "team_a",
                    "team_h_score", "team_a_score"])
        ratings, _ = strict_elo(fixtures, "2024-25")
        ratings = ratings.set_index("fixture")
        self.assertEqual(ratings.loc[1, "home_elo"], 1500.)
        self.assertEqual(ratings.loc[2, "home_elo"], 1500.)
        self.assertGreater(ratings.loc[3, "home_elo"], 1500.)
        self.assertGreater(ratings.loc[3, "away_elo"], 1500.)

    def test_double_gameweek_player_history_is_frozen(self):
        dates = pd.date_range("2024-08-01", periods=7, freq="7D", tz="UTC")
        rows = []
        for i in range(5):
            rows.append((i+1, i+1, dates[i], 2, 90))
        rows += [(6, 9, dates[5], 10, 90),
                 (7, 9, dates[5] + pd.Timedelta(days=2), 20, 90)]
        d = pd.DataFrame(rows, columns=["fixture", "event", "kickoff_time",
                                        "total_points", "minutes"])
        d["season"] = "2024-25"
        d["player_key"] = "2024-25_1"
        d["position"] = "MID"
        d["value"] = 60
        d["was_home"] = True
        d["own_elo"] = 1500.
        d["opp_elo"] = 1500.
        found = deadline_features(d).sort_values("fixture")
        self.assertEqual(len(found), 2)
        self.assertEqual(found.quality6.tolist(), [2., 2.])
        self.assertEqual(found.prior_count.tolist(), [5, 5])


if __name__ == "__main__":
    unittest.main()
