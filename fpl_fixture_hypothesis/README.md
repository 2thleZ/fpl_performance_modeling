# FPL fixture-strength hypothesis test

This project tests whether stronger opposition is associated with fewer Fantasy Premier League points **for midfielders and forwards**, and whether that association changes for historically high-performing attackers.

## Install and run

Requirements: Python 3.10+, pandas, numpy, matplotlib, and statsmodels. With the
downloaded archive in the project-level `data/` directory, run from the project root:

```bash
python fpl_fixture_hypothesis/fpl_fixture_hypothesis.py --data-root data
```

This reads local CSVs only. Alternatively, the script can download the archived CSVs:

```bash
python -m pip install pandas numpy matplotlib statsmodels
python fpl_fixture_hypothesis.py
```

`python` runs Python; `-m` runs a Python module (`pip`, the Python package installer). The script fetches archived CSVs from the community-maintained [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) repository, not the live FPL service. The downloaded files are cached under `data/`.

## Scope

- Seasons: 2022–23, 2023–24, 2024–25.
- An observation is a player **in one match**, not an entire gameweek (double gameweeks count as multiple matches).
- Primary analysis: midfielders and forwards who played at least 60 minutes, in gameweek 9 onward, and already had at least five earlier appearances of 60+ minutes in the same season.
- Outcome: actual FPL points in that match.
- Opponent strength: chronological Elo rating calculated strictly from **earlier** match results. All clubs initialize at 1500 each season; early GWs are excluded because ratings need time to become informative. Elo settings: K=24, home advantage=65 Elo points. These are modeling assumptions, not official FPL ratings.
- Player quality: average FPL points from their previous **up to eight appearances of 60+ minutes**. The current match is excluded.
- Controls: player's own team's pre-match Elo, whether playing at home, season-week trend, player's prior form, and player-season fixed effects.
- Inference: two-way-clustered standard errors by player-season and fixture.

## Hypothesis tests

**H1:** Do harder opponents reduce attackers' FPL points, controlling for player-season quality and other covariates? Check `opp_100` in `regression_coefficients.csv`. A negative coefficient implies fewer points against opponents rated 100 Elo higher, for a player at average prior form. A confidence interval including zero indicates the data do not rule out a null effect.

**H2:** Do historically stronger attackers respond differently to difficult opponents? Check `quality_x_opp`. A positive coefficient means that higher historical player form is associated with a smaller penalty (or larger benefit) from strong opposition. **It is not proof of a big-match boost.** To test for an actual boost, check whether the `top_quartile_quality` opponent slope in `results.json` is significantly **above** zero.

**H3 (descriptive only):** Is a strong-team attacker against strong opposition still scoring more than a medium-team attacker against weak opposition? Inspect `team_matchups.csv`. These groups include mixed player qualities and costs, so a simple comparison cannot establish transfer profitability.

## Outputs

- `results/opponent_strength.png`: mean points by opposition tier and player prior form.
- `results/opponent_tier_summary.csv`: averages by position, pre-match player quality and opponent tier.
- `results/team_matchups.csv`: raw means for combinations of own-team and opposing-team strength.
- `results/regression_coefficients.csv`: fixed-effects regression with confidence intervals and p-values.
- `results/results.json`: sample sizes, top/bottom quartile opposition slopes, plus sensitivity estimates using players who played at least 75 minutes.

## Important caveats

1. This is **observational**, not a causal experiment. Strong opposition may affect minutes, tactical role, and selection in ways we have not modeled. The primary dataset **conditions on playing at least 60 minutes**. That makes results useful for on-field performance but not yet directly useful for selecting an FPL squad before team news.
2. The Elo approach is a starting model. More rigorous follow-up: carry ratings across seasons with promotion/relegation adjustments, distinguish attack/defence strength, or use independent pre-match bookmaker odds as a validation source.
3. Prior-form estimates may be volatile with only five earlier qualifying appearances; check sensitivity to longer history.
4. The relationship may be nonlinear; a next iteration should use opponent-strength bins or smooth curves and test home/away and MID/FWD interactions separately.
5. FPL scoring and data collection may vary across seasons. Validate missing rows, duplicates, and scoring rules before inference beyond these seasons.
6. Points per fixture differ from points per gameweek. The eventual squad simulation must handle blank/double gameweeks and player availability.
7. The historical community archive is not a guarantee of unrestricted rights to the underlying league data. Review licensing/terms if distributing derived datasets.
8. Regenerate the outputs from the local archive before interpreting them. Do **not** interpret placeholder/synthetic tests as football evidence.
