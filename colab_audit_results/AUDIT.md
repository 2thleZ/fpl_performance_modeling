# Colab result audit

This audit uses saved predictions only. Positive gains favor the comparison.
All comparisons are exploratory: configurations were chosen after viewing the original report.
Intervals resample season-gameweeks after averaging seeds. They are pointwise,
not adjusted for multiple comparisons; serial dependence across weeks may narrow them too much.
Partial TabICL configurations are excluded from comparative conclusions.

## Matched forecast errors

| reference | comparison | population | mae_a | mae_b | mae_gain | mae_low | mae_high | rmse_a | rmse_b | rmse_gain | rmse_low | rmse_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tabpfn20_base | tabpfn20_elo | all | 1.0012 | 0.9949 | 0.0063 | 0.0044 | 0.0081 | 2.0175 | 2.0115 | 0.0060 | 0.0032 | 0.0088 |
| tabpfn20_base | tabpfn20_elo | likely_starter | 2.1680 | 2.1526 | 0.0154 | 0.0095 | 0.0210 | 3.1380 | 3.1250 | 0.0130 | 0.0069 | 0.0191 |
| ridge_full_interaction | tabpfn20_elo | all | 1.0386 | 0.9949 | 0.0437 | 0.0416 | 0.0458 | 2.0134 | 2.0115 | 0.0020 | -0.0003 | 0.0042 |
| ridge_full_interaction | tabpfn20_elo | likely_starter | 2.1674 | 2.1526 | 0.0148 | 0.0112 | 0.0183 | 3.1222 | 3.1250 | -0.0029 | -0.0071 | 0.0015 |
| ridge20_interaction | tabpfn20_elo | all | 1.0431 | 0.9949 | 0.0482 | 0.0459 | 0.0504 | 2.0144 | 2.0115 | 0.0030 | 0.0008 | 0.0052 |
| ridge20_interaction | tabpfn20_elo | likely_starter | 2.1724 | 2.1526 | 0.0198 | 0.0162 | 0.0233 | 3.1227 | 3.1250 | -0.0023 | -0.0066 | 0.0020 |

## Matched top-three selections

Each selection is within season, gameweek, position and price band. These are
independent ranking tasks, not one legal FPL squad or a sequence of transfers.
Price bands do not enforce identical spending; mean selected prices are shown.

| reference | comparison | scope | baseline_points_per_pick | comparison_points_per_pick | gain_per_pick | ci_low | ci_high | mean_price_a | mean_price_b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tabpfn20_base | tabpfn20_elo | pooled | 4.4989 | 4.6095 | 0.1106 | 0.0291 | 0.1931 | 7.2224 | 7.1913 |
| tabpfn20_base | tabpfn20_elo | 2022-23 | 4.5238 | 4.5745 | 0.0507 | -0.0599 | 0.1679 | 7.6542 | 7.6349 |
| tabpfn20_base | tabpfn20_elo | 2023-24 | 4.8322 | 4.9632 | 0.1310 | -0.0899 | 0.3516 | 6.7935 | 6.7674 |
| tabpfn20_base | tabpfn20_elo | 2024-25 | 4.5869 | 4.7784 | 0.1915 | 0.0088 | 0.3820 | 7.2444 | 7.2028 |
| tabpfn20_base | tabpfn20_elo | 2025-26 | 4.0930 | 4.1693 | 0.0762 | -0.0640 | 0.2048 | 7.1203 | 7.0829 |
| tabpfn20_base | tabpfn20_elo | FWD | 4.0263 | 4.0745 | 0.0482 | -0.0598 | 0.1619 | 6.6645 | 6.6455 |
| tabpfn20_base | tabpfn20_elo | MID | 4.8319 | 4.9865 | 0.1546 | 0.0420 | 0.2678 | 7.6155 | 7.5759 |
| ridge_full_interaction | tabpfn20_elo | pooled | 4.5506 | 4.6095 | 0.0589 | -0.0143 | 0.1306 | 7.1923 | 7.1913 |
| ridge_full_interaction | tabpfn20_elo | 2022-23 | 4.5696 | 4.5745 | 0.0049 | -0.1803 | 0.1757 | 7.6178 | 7.6349 |
| ridge_full_interaction | tabpfn20_elo | 2023-24 | 4.7991 | 4.9632 | 0.1641 | -0.0115 | 0.3602 | 6.7594 | 6.7674 |
| ridge_full_interaction | tabpfn20_elo | 2024-25 | 4.6745 | 4.7784 | 0.1039 | 0.0065 | 0.1998 | 7.2178 | 7.2028 |
| ridge_full_interaction | tabpfn20_elo | 2025-26 | 4.1899 | 4.1693 | -0.0207 | -0.1246 | 0.0784 | 7.0971 | 7.0829 |
| ridge_full_interaction | tabpfn20_elo | FWD | 4.0538 | 4.0745 | 0.0207 | -0.0840 | 0.1208 | 6.6616 | 6.6455 |
| ridge_full_interaction | tabpfn20_elo | MID | 4.9007 | 4.9865 | 0.0859 | -0.0031 | 0.1795 | 7.5662 | 7.5759 |
| ridge20_interaction | tabpfn20_elo | pooled | 4.5898 | 4.6095 | 0.0198 | -0.0463 | 0.0827 | 7.1876 | 7.1913 |
| ridge20_interaction | tabpfn20_elo | 2022-23 | 4.6691 | 4.5745 | -0.0946 | -0.2554 | 0.0407 | 7.6217 | 7.6349 |
| ridge20_interaction | tabpfn20_elo | 2023-24 | 4.8595 | 4.9632 | 0.1038 | -0.0501 | 0.2826 | 6.7577 | 6.7674 |
| ridge20_interaction | tabpfn20_elo | 2024-25 | 4.7098 | 4.7784 | 0.0686 | -0.0315 | 0.1667 | 7.2090 | 7.2028 |
| ridge20_interaction | tabpfn20_elo | 2025-26 | 4.1505 | 4.1693 | 0.0187 | -0.0793 | 0.1111 | 7.0847 | 7.0829 |
| ridge20_interaction | tabpfn20_elo | FWD | 4.0681 | 4.0745 | 0.0064 | -0.0911 | 0.0989 | 6.6540 | 6.6455 |
| ridge20_interaction | tabpfn20_elo | MID | 4.9574 | 4.9865 | 0.0292 | -0.0481 | 0.1077 | 7.5636 | 7.5759 |

## TabPFN 20k Elo versus its baseline: subgroup checks

| season | position | price_band | events | seeds | gain_per_pick | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | all | all | 120.0000 | 3.0000 | 0.1106 | 0.0291 | 0.1931 |
| 2022-23 | all | all | 30.0000 | 3.0000 | 0.0507 | -0.0599 | 0.1679 |
| 2023-24 | all | all | 30.0000 | 3.0000 | 0.1310 | -0.0899 | 0.3516 |
| 2024-25 | all | all | 30.0000 | 3.0000 | 0.1915 | 0.0088 | 0.3820 |
| 2025-26 | all | all | 30.0000 | 3.0000 | 0.0762 | -0.0640 | 0.2048 |
| all | FWD | all | 120.0000 | 3.0000 | 0.0482 | -0.0598 | 0.1619 |
| all | MID | all | 120.0000 | 3.0000 | 0.1546 | 0.0420 | 0.2678 |
| all | all | 10_plus | 44.0000 | 3.0000 | 0.0000 | 0.0000 | 0.0000 |
| all | all | 6_to_8 | 120.0000 | 3.0000 | 0.0879 | -0.0801 | 0.2556 |
| all | all | 8_to_10 | 115.0000 | 3.0000 | 0.0530 | -0.0556 | 0.1592 |
| all | all | under_6 | 120.0000 | 3.0000 | 0.1903 | 0.0491 | 0.3370 |
| 2022-23 | FWD | all | 30.0000 | 3.0000 | -0.0458 | -0.2290 | 0.1231 |
| 2022-23 | MID | all | 30.0000 | 3.0000 | 0.1082 | -0.0412 | 0.2688 |
| 2023-24 | FWD | all | 30.0000 | 3.0000 | -0.0510 | -0.3278 | 0.2354 |
| 2023-24 | MID | all | 30.0000 | 3.0000 | 0.2543 | 0.0133 | 0.5031 |
| 2024-25 | FWD | all | 30.0000 | 3.0000 | 0.3117 | 0.0778 | 0.5648 |
| 2024-25 | MID | all | 30.0000 | 3.0000 | 0.1032 | -0.1748 | 0.3727 |
| 2025-26 | FWD | all | 30.0000 | 3.0000 | -0.0356 | -0.1825 | 0.1042 |
| 2025-26 | MID | all | 30.0000 | 3.0000 | 0.1690 | -0.0526 | 0.3918 |
| all | FWD | 10_plus | 1.0000 | 3.0000 | 0.0000 | 0.0000 | 0.0000 |
| all | FWD | 6_to_8 | 119.0000 | 3.0000 | -0.0840 | -0.2698 | 0.1036 |
| all | FWD | 8_to_10 | 39.0000 | 3.0000 | 0.0000 | 0.0000 | 0.0000 |
| all | FWD | under_6 | 120.0000 | 3.0000 | 0.1954 | 0.0259 | 0.3648 |
| all | MID | 10_plus | 44.0000 | 3.0000 | 0.0000 | 0.0000 | 0.0000 |
| all | MID | 6_to_8 | 120.0000 | 3.0000 | 0.2583 | 0.0046 | 0.5130 |
| all | MID | 8_to_10 | 112.0000 | 3.0000 | 0.0714 | -0.0724 | 0.2192 |
| all | MID | under_6 | 120.0000 | 3.0000 | 0.1852 | -0.0509 | 0.4297 |

## Candidate counts and forced selections

A group with exactly three candidates forces every model to select all three.
Its zero gain is not evidence of no fixture effect. Groups with fewer than three
are excluded altogether. This particularly limits premium-player conclusions.

| position | price_band | observed_groups | scored_groups | forced_top_three_groups | groups_with_choice |
| --- | --- | --- | --- | --- | --- |
| FWD | under_6 | 120.0000 | 120.0000 | 1.0000 | 119.0000 |
| FWD | 6_to_8 | 120.0000 | 119.0000 | 1.0000 | 118.0000 |
| FWD | 8_to_10 | 118.0000 | 39.0000 | 39.0000 | 0.0000 |
| FWD | 10_plus | 106.0000 | 1.0000 | 1.0000 | 0.0000 |
| MID | under_6 | 120.0000 | 120.0000 | 0.0000 | 120.0000 |
| MID | 6_to_8 | 120.0000 | 120.0000 | 0.0000 | 120.0000 |
| MID | 8_to_10 | 120.0000 | 112.0000 | 16.0000 | 96.0000 |
| MID | 10_plus | 118.0000 | 44.0000 | 41.0000 | 3.0000 |

## Interpretation and next experiment

- All historical tests cover MID/FWD, GW9 onward, with five prior fixture rows;
  they do not validate goalkeepers, defenders, new arrivals, or early season picks.
- Histories/Elo are frozen before the first gameweek kickoff, a proxy for the
  true deadline. Recorded kickoff/schedule data are final archive values. Match
  completion times and historical deadline/schedule snapshots are not verified.
- Prices are the last observed archived prices, not verified deadline or manager
  selling prices. Injury, suspension and historical eligibility are unavailable.
- The long quality measure uses earlier fixtures within the same season, with
  shrinkage to a fixed 2.5-point prior; it is not a multi-season player estimate.
- Four model-ready rows have zero minutes but nonzero points. The two-stage
  decomposition assumes zero points for nonappearance and misses these exceptions.
- Target scoring rules changed over the historical period. Raw total_points
  have not been harmonized to current rules.
- Pooled gains must not be multiplied directly into transfer profits. A squad
  decision needs budget, selling prices, free transfers, bench cover, captaincy,
  chips and the opportunity to wait. A four-point hit is charged immediately.
- Next validation should fix a small model/feature shortlist, evaluate added
  minutes/availability and attacking-process features, and reserve genuinely
  unseen deadlines for evaluation. Reusing these seasons is exploratory.
