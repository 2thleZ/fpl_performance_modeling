# FPL fixture strength and transfer strategy

This project asks whether difficult fixtures reduce Fantasy Premier League (FPL)
points, whether that information improves forecasts, and whether it helps a
budget-constrained squad earn more points after transfer costs. It contains three
steps: an observational attacker analysis, a held-out forecast comparison, and a
squad strategy backtest. The Python scripts generate the results; the Jupyter
notebook is an optional, interactive walkthrough of the original hypothesis test.

## Install and run

Requirements: Python 3.10+, pandas, numpy, matplotlib, statsmodels,
scikit-learn, and SciPy. The full backtest uses `scipy.optimize.milp` and needs
SciPy 1.9 or later. Install the dependencies with:

```bash
python -m pip install pandas numpy matplotlib statsmodels scikit-learn 'scipy>=1.9'
```

Place the downloaded [vaastav/Fantasy-Premier-League data archive](https://github.com/vaastav/Fantasy-Premier-League/tree/master/data)
in the project-level `data/` directory. Each season must contain
`gws/merged_gw.csv`, plus `fixtures.csv` when available; older seasons also
need `players_raw.csv` for position labels. From the project root, run:

```bash
python fpl_fixture_hypothesis/fpl_fixture_hypothesis.py --data-root data
python fpl_fixture_hypothesis/heldout_prediction.py --data-root data
python fpl_fixture_hypothesis/strategy_backtest.py --data-root data
```

These commands read the local archive and write outputs beside their scripts to
`results/`, `heldout_results/`, and `strategy_results/`, respectively. The first
script can also download its archive inputs into
`fpl_fixture_hypothesis/data/` if run without `--data-root`:

```bash
python fpl_fixture_hypothesis/fpl_fixture_hypothesis.py
```

This download option reads the third-party historical archive, not the live FPL
service. Its cache is `fpl_fixture_hypothesis/data/`; the local archive used by
the three commands above is the separate project-level `data/` directory.

## 1. Attacker fixture hypothesis

- Seasons: every completed season from 2016–17 through 2024–25. The archive lacks
  standalone fixture tables for 2016–17 and 2017–18, so the script reconstructs
  those fixtures from the player-match rows. Position labels missing from older
  gameweek files are joined from each season's `players_raw.csv`.
- An observation is a player **in one match**, not an entire gameweek (double gameweeks count as multiple matches).
- Primary analysis: midfielders and forwards who played at least 60 minutes, in gameweek 9 onward, and already had at least five earlier appearances of 60+ minutes in the same season.
- Outcome: actual FPL points in that match.
- Opponent strength: chronological Elo rating calculated strictly from **earlier** match results. All clubs initialize at 1500 each season; early GWs are excluded because ratings need time to become informative. Elo settings: K=24, home advantage=65 Elo points. These are modeling assumptions, not official FPL ratings. This exploratory regression freezes ratings before each kickoff; the forecast and backtest freeze them before the gameweek.
- Player quality: average FPL points from their previous **up to eight appearances of 60+ minutes**. The current match is excluded.
- Controls: player's own team's pre-match Elo, whether playing at home, season-week trend, player's prior form, and player-season fixed effects.
- Inference: two-way-clustered standard errors by player-season and fixture.

## Hypothesis tests

**H1:** Do harder opponents reduce attackers' FPL points, controlling for player-season quality and other covariates? Check `opp_100` in `regression_coefficients.csv`. A negative coefficient implies fewer points against opponents rated 100 Elo higher, for a player at average prior form. A confidence interval including zero indicates the data do not rule out a null effect.

**H2:** Do historically stronger attackers respond differently to difficult opponents? Check `quality_x_opp`. A positive coefficient means that higher historical player form is associated with a smaller penalty (or larger benefit) from strong opposition. **It is not proof of a big-match boost.** To test for an actual boost, check whether the `top_quartile_quality` opponent slope in `results.json` is significantly **above** zero.

**H3 (descriptive only):** Is a strong-team attacker against strong opposition still scoring more than a medium-team attacker against weak opposition? Inspect `team_matchups.csv`. These groups include mixed player qualities and costs, so a simple comparison cannot establish transfer profitability.

In the saved run across 2016–17 through 2024–25 (25,404 qualifying
player-match observations), an opponent rated 100 Elo points higher is
associated with **0.55 fewer FPL points** for an attacker at average prior
form (95% CI: 0.49 to 0.62 fewer points). The high-form interaction is negative:
the top prior-form quartile also has a negative opponent-strength slope. These
observational results do not establish a causal fixture effect or an elite
player “big-match boost.”

### Outputs

- `results/opponent_strength.png`: mean points by opposition tier and player prior form.
- `results/opponent_tier_summary.csv`: averages by position, pre-match player quality and opponent tier.
- `results/team_matchups.csv`: raw means for combinations of own-team and opposing-team strength.
- `results/regression_coefficients.csv`: fixed-effects regression with confidence intervals and p-values.
- `results/results.json`: sample sizes, top/bottom quartile opposition slopes, plus sensitivity estimates using players who played at least 75 minutes.

## 2. Held-out prediction test

Run the pre-deadline forecast comparison with:

```bash
python fpl_fixture_hypothesis/heldout_prediction.py --data-root data
```

This trains two otherwise identical gradient-boosting point models on
midfielders and forwards from 2016–17 through 2024–25 and evaluates them on
2025–26 gameweeks 9–38:

- The **player-history baseline** uses prior points, prior minutes, prior 60-minute
  rate, longer-run season averages, price, position, and gameweek.
- The **fixture-aware model** adds pre-match opponent Elo, own-team Elo, their
  difference, and home advantage.

All rolling features and Elo ratings are frozen at the gameweek deadline. Two
fixtures in the same gameweek therefore receive the same player-history features,
while their individual fixture features may differ. Fixture predictions are
summed before player-gameweek metrics are calculated.

This older comparison does **not** isolate opponent strength: its baseline
also omits home advantage and own-team Elo, while its fixture-aware model adds
all three. It uses the price in the archived fixture row, which is not a
verified deadline snapshot. Treat its top-20 ranking as a diagnostic, not
evidence for an opponent-Elo-specific transfer gain.

The script also trains a separate start-probability model and point models for
players who start and players who do not. Exact start labels are available for
2022–23 through 2024–25; the start model is evaluated on 2025–26. Its held-out
ROC-AUC is **0.923** and Brier score is **0.101** for attackers. Starting is
not the same as playing 60 minutes or being confirmed fit at the deadline.

The fixture-aware point model has a player-gameweek mean absolute error (MAE)
of **1.0068**, versus **1.0087** for player history alone. A paired gameweek
bootstrap interval for the improvement crosses zero, so this one season does
not establish a consistent forecast advantage.

Outputs are written under `heldout_results/`:

- `summary.json`: error, ranking correlation, paired gameweek bootstrap, and an
  unconstrained top-20 ranking diagnostic.
- `mae_by_gameweek.csv`: player-gameweek error for each held-out gameweek.
- `top20_by_gameweek.csv`: realized points from the two unconstrained rankings.
- `heldout_comparison.png`: error and cumulative ranking comparison.

The top-20 calculation is a ranking diagnostic without FPL squad rules or
transfer costs.

## Prospective fixture-value validation

Run `python fpl_fixture_hypothesis/prospective_validation.py` using the local
project-level `data/` archive. It compares three nested ridge regressions:
historical player quality, prior minutes, prior 60-minute rate, last observed
price, position, home advantage, gameweek, and own-team Elo; then opponent Elo;
then an opponent-Elo by long-term-quality interaction. The long-term quality
feature is the player's mean points across **all earlier season fixtures**,
shrunk toward 2.5 points using five pseudo-fixtures. This is a modeling prior,
not an estimated player rating. All features precede the earliest kickoff in
the gameweek. Elo uses only earlier actual kickoffs and is frozen for the whole
gameweek. Double-gameweek fixtures share their player history but receive
individual opponent ratings.

The main target is unconditional fixture points: all archived MID/FWD
player-fixture rows in GW 9+ with five earlier fixture records and a last
observed price, including zero-minute rows. Walk-forward tests train on
earlier seasons and test 2022–23, 2023–24, 2024–25, and 2025–26 in turn.
`prospective_results/summary.json` holds metrics, gameweek-clustered bootstrap
intervals, coefficient checks, Elo carryover comparisons, and matchup groups.
The script also emits a row-level out-of-sample CSV for audit.

On 50,125 out-of-sample fixture rows, baseline MAE is 1.0373 points.
Adding opponent Elo alone raises MAE to 1.0419, though RMSE falls from
2.0194 to 2.0159. Adding the interaction lowers MAE to 1.0336 and RMSE to
2.0133. Thus the three-way comparison finds a small, model-dependent gain,
not a clear standalone opponent-Elo gain. A separate 60+-minute diagnostic
has larger fixture gains but conditions on minutes that are unknown at the
deadline. No transfer strategy is optimized in this validation.

## 3. Constrained squad backtest

Run `strategy_backtest.py` with the local archive command above. Its training
period is 2016–17 through 2024–25 and its held-out scoring period is 2025–26
gameweeks 9–38. It includes goalkeepers and defenders as well as attackers.
For the fixture strategies, the point forecast combines the probability of
starting with separate expected points for starts and non-starts. The quality
hold strategy uses the player-history point forecast. The simulator selects a
15-player squad within a £100.0m budget (2 GK, 5 DEF, 5 MID, 3 FWD; at most
three per club), then picks a legal predicted XI and captain each gameweek.

| Strategy | Transfer choice | Net points | Transfers | Hit cost |
| --- | --- | ---: | ---: | ---: |
| Quality hold | Initial player-history squad; no transfers | 1,489 | 0 | 0 |
| Fixture hybrid | Greedy positive swaps using the current plus three upcoming gameweeks | 1,571 | 30 | 0 |
| Fixture churn | Re-optimize the full squad each gameweek | 929 | 200 | 680 |

The hybrid's **82-point lead** over quality hold is a result for this one
held-out season and these combined modeling choices. It does not isolate the
effect of fixture data: the initial squad, start-aware lineups, and transfer
policy differ between the strategies. The churn policy has higher points
before hits (1,609) but loses 680 points to transfers.

For hybrid and churn decisions, the current and next three gameweeks carry
weights 1.0, 0.8, 0.6, and 0.4. Player history and team Elo are frozen at the
selection gameweek; the upcoming opponents and home/away slots supply the
fixture horizon. Unused free transfers accumulate to a maximum of five;
extra transfers cost four points each. Squad ownership, budgets, starting XIs,
and captains are checked each gameweek.

`strategy_results/summary.csv` and `summary.json` contain the totals and
simulation assumptions. `weekly_results.csv` records points, hits, transfers,
free-transfer balances, and squad costs by gameweek.
`cumulative_points.png` plots cumulative net points.

## Notebook

`FPL_Fixture_Hypothesis.ipynb` is an interactive presentation of the original
attacker question. It imports functions from `fpl_fixture_hypothesis.py` and
does not drive the saved forecast or backtest results. Use the three Python
commands above to reproduce the full analysis; use the notebook to inspect
and experiment with the first stage.

## Important caveats

1. The regression is **observational** and conditions on playing at least 60
   minutes. Tactical role, selection, and minutes can still confound its
   association. It does not measure the gain from a particular transfer.
2. The backtest uses the completed archive's gameweek assignments as the
   upcoming fixture schedule. If matches were postponed or rescheduled, their
   final gameweeks may not have been known at the historical deadline. That
   schedule hindsight can favor horizon-based decisions; a timestamped fixture
   snapshot would be needed for a strict point-in-time backtest.
3. The transfer candidate pool comes from player rows recorded in that
   gameweek, not a timestamped catalogue of everyone selectable at its
   deadline. A player with no recorded fixture that week cannot be bought in
   the current implementation. There are no archived injury or suspension
   status snapshots to screen candidates.
4. The simulator does not model actual FPL selling prices, an explicit cash bank,
   substitutions, vice captains, chips, or all transfer edge cases. Retained
   players keep their acquisition price; incoming players use their recorded
   price for the new gameweek. Blank and double gameweeks are counted from
   the archive, but squad scoring has no automatic substitutes.
5. The Elo model starts each season at 1500 and does not separately estimate
   attack and defence. FPL rules and data collection vary across seasons.
   Results from one held-out season should be checked on additional seasons
   before they guide real transfers.
