# Initial nonlinear model comparison

This is the first controlled benchmark for the proposed TabPFN/TabICL
experiment. It is a validation step, not a transfer-strategy simulation.

## Protocol

- Model-ready data: 111,016 MID/FWD player-fixture rows from 2016-17 through
  2025-26. Rows require gameweek 9 or later, five earlier fixture records, and
  a previously observed price. Zero-minute rows are retained (60,577; 54.6%).
- Test data: 50,125 rows from four chronological test seasons (2022-23 through
  2025-26). Each split trains only on earlier seasons.
- Initial compute setting: one reproducible 5,000-row training sample per split,
  one seed, and one TabICL ensemble member. This is deliberately a first-pass
  Mac benchmark, not a final foundation-model estimate.
- Baseline features: recent and longer-term historical player points, prior
  minutes and 60-minute rate, historical count, last observed price, position,
  home/away, gameweek, and own-team Elo.
- Fixture specifications: baseline; baseline plus opponent Elo; baseline plus
  opponent Elo and longer-term-quality by opponent-Elo interaction. Every
  feature is frozen before the first kickoff of its gameweek.
- Targets and predictions are not clipped. TabICL is explicitly asked for its
  predictive mean. The two-stage forecast is probability of minutes greater
  than zero multiplied by expected points conditional on minutes greater than
  zero.

Four archived rows have negative points with zero recorded minutes. They are
retained in scoring, but the two-stage identity treats the non-playing outcome
as zero and therefore cannot represent those four cases exactly.

## Out-of-sample results

All values pool the same 50,125 test rows. Lower MAE and RMSE are better.

| Model | Form | Fixture specification | MAE | RMSE | Calibration slope |
| --- | --- | --- | ---: | ---: | ---: |
| Ridge | Direct | Baseline | 1.0571 | 2.0224 | 1.037 |
| Ridge | Direct | + opponent Elo | 1.0694 | 2.0199 | 1.033 |
| Ridge | Direct | + Elo interaction | 1.0535 | 2.0169 | 1.032 |
| CatBoost | Direct | Baseline | 1.0532 | 2.0694 | 0.930 |
| CatBoost | Direct | + opponent Elo | 1.0472 | 2.0632 | 0.932 |
| CatBoost | Direct | + Elo interaction | 1.0501 | 2.0662 | 0.918 |
| TabICL | Direct | Baseline | 1.0133 | 2.0203 | 1.002 |
| TabICL | Direct | + opponent Elo | 1.0091 | 2.0171 | 1.001 |
| TabICL | Direct | + Elo interaction | **1.0073** | 2.0169 | 1.009 |
| Ridge | Two-stage | Baseline | 1.0696 | 2.0250 | 1.010 |
| CatBoost | Two-stage | Baseline | 1.0458 | 2.0701 | 0.912 |
| TabICL | Two-stage | Baseline | 1.0221 | 2.0215 | 0.978 |
| TabICL | Two-stage | + Elo interaction | 1.0176 | **2.0167** | 0.980 |

Positive gains below mean that the second specification has lower error. The
95% intervals use a paired bootstrap over the 120 held-out season-gameweeks.

| Model and form | Comparison with baseline | MAE gain (95% CI) | RMSE gain (95% CI) |
| --- | --- | ---: | ---: |
| Ridge, direct | + opponent Elo | -0.0123 (-0.0146, -0.0099) | 0.0025 (0.0012, 0.0039) |
| Ridge, direct | + Elo interaction | 0.0036 (0.0016, 0.0057) | 0.0054 (0.0025, 0.0086) |
| CatBoost, direct | + opponent Elo | 0.0059 (0.0006, 0.0111) | 0.0062 (-0.0023, 0.0141) |
| CatBoost, direct | + Elo interaction | 0.0031 (-0.0017, 0.0075) | 0.0032 (-0.0042, 0.0102) |
| TabICL, direct | + opponent Elo | 0.0042 (0.0027, 0.0057) | 0.0032 (0.0010, 0.0056) |
| TabICL, direct | + Elo interaction | 0.0060 (0.0041, 0.0079) | 0.0033 (0.0002, 0.0066) |

The main signal is consistency, not magnitude: fixture features improve both
metrics for direct TabICL, but by less than 0.01 FPL points per observation.
Ridge shows why both metrics matter: opponent Elo alone improves RMSE while
making MAE materially worse. CatBoost has worse RMSE than Ridge in every direct
feature specification in this initial configuration.

Direct TabICL improves baseline MAE over direct Ridge by 0.0438 points (95% CI
0.0408 to 0.0470), but its RMSE gain is only 0.0021 (-0.0005 to 0.0047). This
is not yet evidence of a large practical player-selection advantage.

## Likely-starter and ranking checks

The pre-deadline likely-starter subset is defined only from earlier fixtures:
mean minutes over the previous six fixtures at least 45 and a previous-six
60-minute rate at least 0.5. It contains 13,529 held-out rows.

| Direct model | Fixture specification | Likely-starter MAE | Likely-starter RMSE |
| --- | --- | ---: | ---: |
| Ridge | Baseline | 2.1845 | 3.1334 |
| Ridge | + Elo interaction | 2.1728 | **3.1221** |
| TabICL | Baseline | 2.1847 | 3.1411 |
| TabICL | + opponent Elo | 2.1751 | 3.1340 |
| TabICL | + Elo interaction | **2.1698** | 3.1340 |

The likely-starter subset is much harder in absolute error because it removes
many easy zero-point predictions. Fixture features still make small MAE gains,
but Ridge has the lowest RMSE here.

Ranking changes are mixed. Across position and price-band gameweek groups,
direct Ridge's mean Spearman correlation changes from 0.5673 (baseline) to
0.5689 (interaction), while direct TabICL changes from 0.5691 to 0.5625. The
realized mean of each group's predicted top three rises from 4.37 points to
4.48 for Ridge and 4.44 for TabICL. These unconstrained rankings are diagnostics,
not legal FPL squad selections, and they do not establish a useful transfer gain.

## Direct versus two-stage

The two-stage decomposition is not consistently better. TabICL's participation
model has the best Brier score (about 0.107 versus 0.113 for CatBoost and 0.123
for Ridge), but direct TabICL still has lower MAE than its two-stage counterpart.
This supports keeping both formulations in the experiment rather than assuming
that modeling participation separately must improve total-point forecasts.

## Candidate-universe audit

The separate audit reconstructs 212,812 player-gameweek rows for 2018-19 through
2025-26. A blank-gameweek player remains one candidate with zero scheduled
fixtures; a double-gameweek player remains one candidate with multiple fixture
forecasts and summed realized points. The older two seasons have no standalone
fixture tables and are not reconstructed.

This is only a restricted universe. Membership is inferred from whether a
player has appeared in the archive by that gameweek; blank-week prices are
carried forward; historical status/eligibility is unavailable; and the final
season fixture files may contain rescheduling hindsight. It is therefore not
yet a faithful historical FPL selection market.

## Current conclusion and next gate

TabICL runs successfully on the Mac and is competitive, particularly on MAE.
The fixture-strength gain is statistically detectable in several paired checks
but practically tiny at this stage, and the likely-starter/ranking evidence is
mixed. TabPFN has no result yet because its checkpoint requires the user to
accept the Prior Labs license and provide a `TABPFN_TOKEN`.

Before any strategy optimization, the next defensible step is to unlock TabPFN,
repeat the experiment across multiple seeds and ensemble members, then increase
the context size. Only a stable improvement in candidate ranking or constrained
selection—not merely a few thousandths of RMSE—would justify progressing to a
transfer-policy simulation.
