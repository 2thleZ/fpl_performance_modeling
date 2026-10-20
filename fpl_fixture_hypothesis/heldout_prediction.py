#!/usr/bin/env python3
"""Compare player-quality and fixture-aware FPL forecasts on a held-out season."""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from fpl_fixture_hypothesis import SEASONS, get_fixture_data


HOLDOUT_SEASON = "2025-26"
BASELINE_FEATURES = [
    "prior_points_6", "prior_minutes_6", "prior_60_rate_6",
    "prior_onfield_points_8", "season_points_per_fixture",
    "season_minutes_per_fixture", "value", "is_fwd", "event_scaled",
]
FIXTURE_FEATURES = BASELINE_FEATURES + ["opp_100", "own_100", "elo_gap_100", "home"]


def pre_deadline_features(appearances: pd.DataFrame) -> pd.DataFrame:
    """Create features using only gameweeks completed before the current deadline.

    Features are frozen for all fixtures in the same gameweek. This prevents the
    first match of a double gameweek from informing the forecast for the second.
    """
    d = (appearances.sort_values(["player_key", "event", "kickoff_time", "fixture"])
         .reset_index(drop=True).copy())
    player = d.groupby("player_key", sort=False)
    # Calculate row-level histories, then copy the value available before the
    # first fixture of a gameweek to every row in that player-gameweek.
    d["_fixture_count"] = player.cumcount().astype(float)
    d["_points_6"] = player["total_points"].transform(
        lambda s: s.shift().rolling(6, min_periods=1).mean())
    d["_minutes_6"] = player["minutes"].transform(
        lambda s: s.shift().rolling(6, min_periods=1).mean())
    d["_rate_60_6"] = (d["minutes"].ge(60).astype(float).groupby(d["player_key"], sort=False)
                         .transform(lambda s: s.shift().rolling(6, min_periods=1).mean()))
    d["_season_points"] = player["total_points"].transform(
        lambda s: s.shift().expanding(min_periods=1).mean())
    d["_season_minutes"] = player["minutes"].transform(
        lambda s: s.shift().expanding(min_periods=1).mean())
    frozen = {
        "prior_fixture_count": "_fixture_count",
        "prior_points_6": "_points_6",
        "prior_minutes_6": "_minutes_6",
        "prior_60_rate_6": "_rate_60_6",
        "prior_onfield_points_8": "prior_8_points",
        "season_points_per_fixture": "_season_points",
        "season_minutes_per_fixture": "_season_minutes",
    }
    keys = ["player_key", "event"]
    first_rows = d.drop_duplicates(keys, keep="first")[keys + list(frozen.values())].copy()
    first_rows = first_rows.rename(columns={source: output for output, source in frozen.items()})
    d = d.drop(columns=["_fixture_count", "_points_6", "_minutes_6", "_rate_60_6",
                        "_season_points", "_season_minutes", "prior_8_points"])
    d = d.merge(first_rows, on=keys, how="left", validate="many_to_one")

    for col in ("value", "event", "total_points", "minutes"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["is_fwd"] = d["position"].eq("FWD").astype(float)
    d["event_scaled"] = d["event"] / 38.0
    d["opp_100"] = (d["opp_elo"] - 1500.0) / 100.0
    d["own_100"] = (d["own_elo"] - 1500.0) / 100.0
    d["elo_gap_100"] = (d["own_elo"] - d["opp_elo"]) / 100.0
    d["home"] = d["was_home"].astype(float)
    return d.loc[(d["event"] >= 9) & (d["prior_fixture_count"] >= 5)].copy()


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=2.0, random_state=2026,
    )


def metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(actual)
    return {
        "observations": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_actual": float(np.mean(actual)),
        "mean_prediction": float(np.mean(prediction)),
        "spearman": float(pd.Series(actual).corr(pd.Series(prediction), method="spearman")),
    }


def aggregate_gameweeks(test: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "event", "player_key", "name", "position"]
    return (test.groupby(keys, as_index=False)
            .agg(actual_points=("total_points", "sum"),
                 baseline_prediction=("baseline_prediction", "sum"),
                 fixture_prediction=("fixture_prediction", "sum"),
                 prior_60_rate_6=("prior_60_rate_6", "first"),
                 prior_minutes_6=("prior_minutes_6", "first"),
                 value=("value", "first"), fixture_count=("fixture", "size")))


def top_n_points(gameweeks: pd.DataFrame, prediction: str, n: int = 20) -> pd.Series:
    likely = gameweeks.loc[(gameweeks["prior_60_rate_6"] >= 0.5) &
                           (gameweeks["prior_minutes_6"] >= 45)].copy()
    return likely.groupby("event").apply(
        lambda g: g.nlargest(n, prediction)["actual_points"].sum(),
        include_groups=False,
    )


def bootstrap_mae_difference(gameweeks: pd.DataFrame, iterations: int = 5000) -> dict[str, float]:
    by_gw = (gameweeks.assign(
        baseline_abs=lambda x: (x["baseline_prediction"] - x["actual_points"]).abs(),
        fixture_abs=lambda x: (x["fixture_prediction"] - x["actual_points"]).abs())
        .groupby("event")[["baseline_abs", "fixture_abs"]].mean())
    differences = (by_gw["baseline_abs"] - by_gw["fixture_abs"]).to_numpy()
    rng = np.random.default_rng(2026)
    sampled = rng.choice(differences, size=(iterations, len(differences)), replace=True).mean(axis=1)
    return {
        "mae_improvement": float(differences.mean()),
        "low_95": float(np.quantile(sampled, 0.025)),
        "high_95": float(np.quantile(sampled, 0.975)),
        "gameweeks": int(len(differences)),
    }


def save_outputs(test: pd.DataFrame, gameweeks: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    likely = gameweeks.loc[(gameweeks["prior_60_rate_6"] >= 0.5) &
                           (gameweeks["prior_minutes_6"] >= 45)]
    summary = {
        "training_seasons": list(SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "features_frozen_by_gameweek": True,
        "elo_frozen_by_gameweek": True,
        "fixture_level": {
            "baseline": metrics(test["total_points"], test["baseline_prediction"]),
            "fixture_aware": metrics(test["total_points"], test["fixture_prediction"]),
        },
        "player_gameweek": {
            "all_candidates": {
                "baseline": metrics(gameweeks["actual_points"], gameweeks["baseline_prediction"]),
                "fixture_aware": metrics(gameweeks["actual_points"], gameweeks["fixture_prediction"]),
                "paired_gameweek_bootstrap": bootstrap_mae_difference(gameweeks),
            },
            "likely_starters": {
                "definition": "prior 6-fixture 60+ rate >= 0.5 and mean minutes >= 45",
                "baseline": metrics(likely["actual_points"], likely["baseline_prediction"]),
                "fixture_aware": metrics(likely["actual_points"], likely["fixture_prediction"]),
                "paired_gameweek_bootstrap": bootstrap_mae_difference(likely),
            },
        },
    }
    baseline_top = top_n_points(gameweeks, "baseline_prediction")
    fixture_top = top_n_points(gameweeks, "fixture_prediction")
    top = pd.concat([baseline_top.rename("baseline_top20_points"),
                     fixture_top.rename("fixture_top20_points")], axis=1).fillna(0)
    top["fixture_minus_baseline"] = top["fixture_top20_points"] - top["baseline_top20_points"]
    summary["unconstrained_top20_likely_starters"] = {
        "baseline_realized_points": float(top["baseline_top20_points"].sum()),
        "fixture_aware_realized_points": float(top["fixture_top20_points"].sum()),
        "difference": float(top["fixture_minus_baseline"].sum()),
        "note": "Diagnostic ranking comparison; no budget, squad, captaincy, or transfer constraints.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    top.to_csv(out / "top20_by_gameweek.csv")

    per_gw = (gameweeks.assign(
        baseline_abs=lambda x: (x["baseline_prediction"] - x["actual_points"]).abs(),
        fixture_abs=lambda x: (x["fixture_prediction"] - x["actual_points"]).abs())
        .groupby("event")[["baseline_abs", "fixture_abs"]].mean())
    per_gw.to_csv(out / "mae_by_gameweek.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(per_gw.index, per_gw["baseline_abs"], label="Player quality")
    axes[0].plot(per_gw.index, per_gw["fixture_abs"], label="Fixture aware")
    axes[0].set(title="Held-out player-GW mean absolute error", xlabel="Gameweek", ylabel="MAE")
    axes[0].legend()
    axes[1].plot(top.index, top["fixture_minus_baseline"].cumsum(), color="tab:green")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Cumulative top-20 ranking difference", xlabel="Gameweek",
                ylabel="Fixture-aware minus baseline realized points")
    fig.tight_layout()
    fig.savefig(out / "heldout_comparison.png", dpi=160)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "heldout_results")
    args = parser.parse_args()

    seasons = SEASONS + (HOLDOUT_SEASON,)
    appearances, _ = get_fixture_data(
        seasons, Path(__file__).parent / "data", args.data_root,
        freeze_elo_by_event=True,
    )
    frame = pre_deadline_features(appearances)
    train = frame.loc[frame["season"].isin(SEASONS)].copy()
    test = frame.loc[frame["season"].eq(HOLDOUT_SEASON)].copy()
    print(f"Training rows: {len(train):,}; held-out rows: {len(test):,}")

    baseline = make_model().fit(train[BASELINE_FEATURES], train["total_points"])
    fixture = make_model().fit(train[FIXTURE_FEATURES], train["total_points"])
    test["baseline_prediction"] = np.maximum(0.0, baseline.predict(test[BASELINE_FEATURES]))
    test["fixture_prediction"] = np.maximum(0.0, fixture.predict(test[FIXTURE_FEATURES]))
    gameweeks = aggregate_gameweeks(test)
    summary = save_outputs(test, gameweeks, args.out)
    print(json.dumps(summary, indent=2))
    print(f"Files saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
