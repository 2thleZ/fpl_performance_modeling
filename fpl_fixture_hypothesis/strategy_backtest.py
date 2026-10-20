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
                                add_availability_predictions, fit_availability_models,
                                make_model, pre_deadline_features)

SQUAD_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 1000  # FPL's £100m starting budget in £0.1m units.
MAX_FREE_TRANSFERS = 5
HORIZON_WEIGHTS = (1.0, 0.8, 0.6, 0.4)


def player_gameweeks(test: pd.DataFrame) -> pd.DataFrame:
    keys = ["event", "player_key", "name", "position", "own_team_id"]
    return (test.groupby(keys, as_index=False)
            .agg(actual_points=("total_points", "sum"), minutes=("minutes", "sum"),
                 baseline=("baseline_prediction", "sum"), fixture=("fixture_prediction", "sum"),
                 availability=("availability_prediction", "sum"),
                 start_probability=("start_probability", lambda p: 1.0 - np.prod(1.0 - p)),
                 value=("value", "first"), prior_minutes=("prior_minutes_6", "first"),
                 prior_60_rate=("prior_60_rate_6", "first"), fixtures=("fixture", "size")))


def horizon_scores(
    test: pd.DataFrame,
    availability_models: tuple,
    horizon_weights: tuple[float, ...] = HORIZON_WEIGHTS,
) -> pd.DataFrame:
    """Project a player's current state over known upcoming fixtures.

    Player form, price and team Elo are frozen at the selection deadline.  Only
    the known home/away fixture schedule changes over the horizon, so subsequent
    outcomes cannot leak into a current transfer decision.
    """
    events = range(int(test.event.min()), int(test.event.max()) + 1)
    schedule = (test[["event", "fixture", "own_team_id", "opp_team_id", "was_home"]]
                .drop_duplicates()
                .rename(columns={"event": "target_event", "fixture": "target_fixture",
                                 "opp_team_id": "target_opp_team_id", "was_home": "target_home"}))
    ratings = (test.groupby(["event", "own_team_id"], as_index=False)["own_elo"].first()
               .pivot(index="event", columns="own_team_id", values="own_elo")
               .reindex(events).ffill())
    records = []
    for event in events:
        current = test.loc[test.event.eq(event)].drop_duplicates("player_key").copy()
        current["selection_event"] = event
        if current.empty:
            continue
        team_ratings = ratings.loc[event]
        for offset, weight in enumerate(horizon_weights):
            target_event = event + offset
            fixtures = schedule.loc[schedule.target_event.eq(target_event)]
            if fixtures.empty:
                continue
            projected = current.merge(fixtures, on="own_team_id", how="inner", validate="many_to_many")
            projected["event"] = target_event
            projected["fixture"] = projected["target_fixture"]
            projected["home"] = projected["target_home"].astype(float)
            projected["event_scaled"] = target_event / 38.0
            projected["own_elo"] = projected["own_team_id"].map(team_ratings)
            projected["opp_elo"] = projected["target_opp_team_id"].map(team_ratings)
            if projected[["own_elo", "opp_elo"]].isna().any().any():
                raise RuntimeError(f"Missing deadline Elo while projecting GW {event}")
            projected["opp_100"] = (projected["opp_elo"] - 1500.0) / 100.0
            projected["own_100"] = (projected["own_elo"] - 1500.0) / 100.0
            projected["elo_gap_100"] = (projected["own_elo"] - projected["opp_elo"]) / 100.0
            projected = add_availability_predictions(projected, availability_models)
            records.append(projected[["selection_event", "player_key", "availability_prediction"]]
                           .assign(weight=weight))
    projected_scores = pd.concat(records, ignore_index=True)
    return (projected_scores.assign(weighted=lambda d: d["availability_prediction"] * d["weight"])
            .groupby(["selection_event", "player_key"], as_index=False)["weighted"].sum()
            .rename(columns={"selection_event": "event", "weighted": "horizon_score"}))


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


def refresh_forecasts(squad: pd.DataFrame, candidates: pd.DataFrame, scores: tuple[str, ...]) -> pd.DataFrame:
    """Update retained players to the forecasts available at this deadline.

    Prices intentionally remain the acquisition prices held in ``squad``.  This
    keeps the simple budget model independent of unobserved FPL selling prices.
    """
    fresh = candidates.drop_duplicates("player_key").set_index("player_key")
    d = squad.copy()
    for score in scores:
        d[score] = d["player_key"].map(fresh[score]).fillna(-np.inf)
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
        available[["player_key", "actual_points", "minutes", "start_probability", score]],
        on="player_key", how="left")
    d[["actual_points", "minutes", "start_probability", score]] = (
        d[["actual_points", "minutes", "start_probability", score]].fillna(0))
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


def score_week(squad: pd.DataFrame, available: pd.DataFrame, score: str) -> tuple[float, str, float]:
    xi, captain = pick_xi(squad, available, score)
    actual = float(xi.actual_points.sum() + xi.loc[xi.player_key.eq(captain), "actual_points"].sum())
    return actual, captain, float(xi.start_probability.mean())


def run_strategy(
    gameweeks: pd.DataFrame, name: str, transfer_score: str, lineup_score: str, mode: str,
) -> pd.DataFrame:
    """Simulate a strategy with banking of unused free transfers up to five."""
    records, squad, free_transfers = [], None, 1
    for event, available in gameweeks.groupby("event", sort=True):
        if squad is None:
            squad = solve_squad(available, transfer_score)
            transfers = 0
        else:
            free_transfers = min(MAX_FREE_TRANSFERS, free_transfers + 1)
            squad = refresh_forecasts(squad, available, (transfer_score, lineup_score))
            if mode == "hybrid":
                transfers = 0
                for _ in range(free_transfers):
                    before = set(squad.player_key)
                    squad = best_free_transfer(squad, available, transfer_score)
                    added = len(before - set(squad.player_key))
                    transfers += added
                    if not added:
                        break
            elif mode == "churn":
                target = solve_squad(available, transfer_score)
                transfers = len(set(squad.player_key) - set(target.player_key))
                squad = target
            else:
                transfers = 0
        if not valid_squad(squad):
            raise RuntimeError("Squad selection violated FPL rules")
        points, captain, xi_start_probability = score_week(squad, available, lineup_score)
        hit = max(0, transfers - free_transfers) * 4
        free_transfers = max(0, free_transfers - transfers)
        records.append({"strategy": name, "event": int(event), "squad_points": points,
                        "transfer_hit": hit, "net_points": points - hit,
                        "transfers": transfers, "captain": captain,
                        "free_transfers_after": free_transfers,
                        "xi_start_probability": xi_start_probability,
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
    availability_models = fit_availability_models(train)
    test = add_availability_predictions(test, availability_models)
    weeks = player_gameweeks(test).rename(columns={
        "baseline": "baseline_score", "fixture": "fixture_score",
        "availability": "availability_score",
    })
    weeks = weeks.merge(horizon_scores(test, availability_models), on=["event", "player_key"],
                        how="left", validate="one_to_one")
    if weeks["horizon_score"].isna().any():
        raise RuntimeError("Every current candidate needs a fixture-horizon forecast")
    results = pd.concat([
        run_strategy(weeks, "Quality hold", "baseline_score", "baseline_score", "hold"),
        run_strategy(weeks, "Fixture hybrid", "horizon_score", "availability_score", "hybrid"),
        run_strategy(weeks, "Fixture churn", "horizon_score", "availability_score", "churn"),
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
        "lineup": "predicted legal XI and predicted captain; score uses an explicit start-probability and conditional-points model; no automatic substitutions or chips",
        "transfers": "one free transfer per gameweek; unused transfers bank up to five; additional transfers cost 4 points",
        "pricing": "price at acquisition; incoming players use their current price; FPL selling-price effects are not modeled",
        "horizon": f"Fixture-hybrid transfer decisions use the current plus next {len(HORIZON_WEIGHTS) - 1} gameweeks, with weights {list(HORIZON_WEIGHTS)} and deadline-frozen player/team inputs.",
    }, "summary": summary.to_dict(orient="records")}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(f"Files saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
