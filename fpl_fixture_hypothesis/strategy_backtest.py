#!/usr/bin/env python3
"""Backtest constrained FPL squad strategies on the held-out 2025-26 season."""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from fpl_fixture_hypothesis import SEASONS, get_fixture_data
from heldout_prediction import (BASELINE_FEATURES, FIXTURE_FEATURES, HOLDOUT_SEASON,
                                make_model, pre_deadline_features)

SQUAD_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 1000  # FPL's £100m starting budget in £0.1m units.


def player_gameweeks(test: pd.DataFrame) -> pd.DataFrame:
    keys = ["event", "player_key", "name", "position", "own_team_id"]
    return (test.groupby(keys, as_index=False)
            .agg(actual_points=("total_points", "sum"), minutes=("minutes", "sum"),
                 baseline=("baseline_prediction", "sum"), fixture=("fixture_prediction", "sum"),
                 value=("value", "first"), prior_minutes=("prior_minutes_6", "first"),
                 prior_60_rate=("prior_60_rate_6", "first"), fixtures=("fixture", "size")))


def solve_squad(candidates: pd.DataFrame, score: str) -> pd.DataFrame:
    """Choose a 15-player squad subject to FPL position, budget, and club rules."""
    d = candidates.dropna(subset=["value", score]).drop_duplicates("player_key").copy().reset_index(drop=True)
    n = len(d)
    rows, lower, upper = [], [], []
    rows.append(np.ones(n)); lower.append(15); upper.append(15)
    rows.append(d["value"].to_numpy(float)); lower.append(-np.inf); upper.append(BUDGET)
    for pos, count in SQUAD_COUNTS.items():
        rows.append(d["position"].eq(pos).to_numpy(float)); lower.append(count); upper.append(count)
    for team in d["own_team_id"].unique():
        rows.append(d["own_team_id"].eq(team).to_numpy(float)); lower.append(-np.inf); upper.append(3)
    result = milp(c=-d[score].to_numpy(float), integrality=np.ones(n),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(np.vstack(rows), lower, upper),
                  options={"time_limit": 10})
    if not result.success:
        raise RuntimeError(f"Squad optimisation failed: {result.message}")
    return d.loc[result.x > 0.5].copy()


def valid_squad(squad: pd.DataFrame) -> bool:
    if len(squad) != 15 or squad["value"].sum() > BUDGET + 1e-6:
        return False
    if squad.groupby("position").size().to_dict() != SQUAD_COUNTS:
        return False
    return bool((squad.groupby("own_team_id").size() <= 3).all())


def refresh_forecast(squad: pd.DataFrame, candidates: pd.DataFrame, score: str) -> pd.DataFrame:
    """Update retained players to the forecast available at this deadline.

    Prices intentionally remain the acquisition prices held in ``squad``.  This
    keeps the simple budget model independent of unobserved FPL selling prices.
    """
    fresh = candidates.drop_duplicates("player_key").set_index("player_key")[score]
    d = squad.copy()
    d[score] = d["player_key"].map(fresh).fillna(-np.inf)
    return d


def best_free_transfer(squad: pd.DataFrame, candidates: pd.DataFrame, score: str) -> pd.DataFrame:
    """Make the best positive single fixture-based swap, if legal and affordable."""
    current = squad.copy()
    pool = candidates.drop_duplicates("player_key").set_index("player_key", drop=False)
    best_gain, best = 0.0, current
    for out in current.itertuples():
        for inn in pool.loc[pool["position"].eq(out.position)].itertuples():
            if inn.player_key in set(current["player_key"]):
                continue
            candidate = pd.concat([current.loc[current.player_key.ne(out.player_key)],
                                   pd.DataFrame([inn._asdict()])[current.columns]], ignore_index=True)
            if not valid_squad(candidate):
                continue
            gain = float(getattr(inn, score) - getattr(out, score))
            if gain > best_gain:
                best_gain, best = gain, candidate
    return best


def valid_xi(xi: pd.DataFrame) -> bool:
    counts = xi.groupby("position").size()
    return (len(xi) == 11 and counts.get("GK", 0) == 1 and
            3 <= counts.get("DEF", 0) <= 5 and 2 <= counts.get("MID", 0) <= 5 and
            1 <= counts.get("FWD", 0) <= 3)


def pick_xi(squad: pd.DataFrame, available: pd.DataFrame, score: str) -> tuple[pd.DataFrame, str]:
    d = squad[["player_key", "position"]].merge(
        available[["player_key", "actual_points", "minutes", score]], on="player_key", how="left")
    d[["actual_points", "minutes", score]] = d[["actual_points", "minutes", score]].fillna(0)
    best_value, best_lineup = -np.inf, None
    for defenders in range(3, 6):
        for midfielders in range(2, 6):
            forwards = 10 - defenders - midfielders
            if not 1 <= forwards <= 3:
                continue
            chosen = [d.loc[d.position.eq("GK")].nlargest(1, score)]
            for pos, count in (("DEF", defenders), ("MID", midfielders), ("FWD", forwards)):
                chosen.append(d.loc[d.position.eq(pos)].nlargest(count, score))
            xi = pd.concat(chosen)
            value = xi[score].sum()
            if value > best_value:
                best_value, best_lineup = value, xi
    captain = best_lineup.nlargest(1, score).iloc[0].player_key
    if not valid_xi(best_lineup):
        raise RuntimeError("Line-up selection violated FPL formation rules")
    return best_lineup, captain


def score_week(squad: pd.DataFrame, available: pd.DataFrame, score: str) -> tuple[float, str]:
    xi, captain = pick_xi(squad, available, score)
    actual = float(xi.actual_points.sum() + xi.loc[xi.player_key.eq(captain), "actual_points"].sum())
    return actual, captain


def run_strategy(gameweeks: pd.DataFrame, name: str, score: str, mode: str) -> pd.DataFrame:
    records, squad = [], None
    for event, available in gameweeks.groupby("event", sort=True):
        if squad is None:
            squad = solve_squad(available, score)
            transfers = 0
        else:
            squad = refresh_forecast(squad, available, score)
            if mode == "hybrid":
                before = set(squad.player_key)
                squad = best_free_transfer(squad, available, score)
                transfers = len(before - set(squad.player_key))
            elif mode == "churn":
                target = solve_squad(available, score)
                transfers = len(set(squad.player_key) - set(target.player_key))
                squad = target
            else:
                transfers = 0
        if not valid_squad(squad):
            raise RuntimeError("Squad selection violated FPL rules")
        points, captain = score_week(squad, available, score)
        hit = max(0, transfers - 1) * 4
        records.append({"strategy": name, "event": int(event), "squad_points": points,
                        "transfer_hit": hit, "net_points": points - hit,
                        "transfers": transfers, "captain": captain,
                        "squad_cost": float(squad.value.sum())})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "strategy_results")
    args = parser.parse_args()
    appearances, _ = get_fixture_data(SEASONS + (HOLDOUT_SEASON,), Path(__file__).parent / "data",
                                      args.data_root, freeze_elo_by_event=True, positions=None)
    frame = pre_deadline_features(appearances)
    train, test = frame.loc[frame.season.isin(SEASONS)], frame.loc[frame.season.eq(HOLDOUT_SEASON)].copy()
    baseline = make_model().fit(train[BASELINE_FEATURES], train.total_points)
    fixture = make_model().fit(train[FIXTURE_FEATURES], train.total_points)
    test["baseline_prediction"] = np.maximum(0, baseline.predict(test[BASELINE_FEATURES]))
    test["fixture_prediction"] = np.maximum(0, fixture.predict(test[FIXTURE_FEATURES]))
    weeks = player_gameweeks(test).rename(columns={"baseline": "baseline_score", "fixture": "fixture_score"})
    results = pd.concat([
        run_strategy(weeks, "Quality hold", "baseline_score", "hold"),
        run_strategy(weeks, "Fixture hybrid", "fixture_score", "hybrid"),
        run_strategy(weeks, "Fixture churn", "fixture_score", "churn"),
    ], ignore_index=True)
    summary = (results.groupby("strategy", as_index=False)
               .agg(net_points=("net_points", "sum"), squad_points=("squad_points", "sum"),
                    transfer_hits=("transfer_hit", "sum"), transfers=("transfers", "sum")))
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "weekly_results.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, d in results.groupby("strategy"):
        ax.plot(d.event, d.net_points.cumsum(), label=name)
    ax.set(title="Constrained 2025-26 strategy backtest", xlabel="Gameweek", ylabel="Cumulative net points")
    ax.legend(); fig.tight_layout(); fig.savefig(args.out / "cumulative_points.png", dpi=160); plt.close(fig)
    payload = {"holdout_season": HOLDOUT_SEASON, "analysis_window": {
        "first_gameweek": int(weeks.event.min()), "last_gameweek": int(weeks.event.max()),
        "reason": "Forecast features require eight prior player-gameweeks.",
    }, "budget": "£100.0m", "rules": {
        "squad": "2 GK, 5 DEF, 5 MID, 3 FWD; maximum 3 players per club",
        "lineup": "predicted legal XI and predicted captain; no automatic substitutions or chips",
        "transfers": "one free transfer per gameweek; additional transfers cost 4 points; free-transfer rollovers are not modeled",
        "pricing": "price at acquisition; incoming players use their current price; FPL selling-price effects are not modeled",
    }, "summary": summary.to_dict(orient="records")}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(f"Files saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
