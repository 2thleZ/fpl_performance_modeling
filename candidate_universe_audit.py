#!/usr/bin/env python3
"""Audit and reconstruct the archive's restricted player-gameweek universe."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prospective_validation import DATA, ROOT, SEASON_LIST

OUT = ROOT / "candidate_universe_results"
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def aggregate_scheduled_rows(current: pd.DataFrame) -> pd.DataFrame:
    """Collapse fixture rows to one selectable player row for a gameweek."""
    current = current.copy()
    if current.empty:
        return pd.DataFrame(columns=["element", "name", "position", "team", "price",
                                     "scheduled_fixtures", "actual_points"])
    for column in ("element", "value", "total_points", "fixture", "own_team_id"):
        if column in current:
            current[column] = pd.to_numeric(current[column], errors="coerce")
    current["team_value"] = current.get("team", pd.Series(np.nan, index=current.index))
    current["team_value"] = pd.to_numeric(current.team_value, errors="coerce").fillna(
        current.get("own_team_id", pd.Series(np.nan, index=current.index)))
    if "position" not in current:
        current["position"] = np.nan
    if "name" not in current:
        current["name"] = np.nan
    rows = []
    for element, group in current.groupby("element", sort=False):
        rows.append({
            "element": int(element),
            "name": group["name"].dropna().iloc[-1] if group["name"].notna().any() else None,
            "position": group.position.dropna().iloc[-1] if group.position.notna().any() else None,
            "team": float(group.team_value.dropna().iloc[-1]) if group.team_value.notna().any() else np.nan,
            "price": float(group.value.dropna().iloc[-1]) if group.value.notna().any() else np.nan,
            "scheduled_fixtures": int(group.fixture.nunique()),
            "actual_points": float(group.total_points.sum()),
        })
    return pd.DataFrame(rows)


def expand_seen_pool(seen: pd.DataFrame, scheduled: pd.DataFrame,
                     fixture_counts: dict[int, int]) -> pd.DataFrame:
    """Add previously observed blank-GW players with zero scheduled points."""
    state = seen.drop_duplicates("element", keep="last").set_index("element")
    current = scheduled.set_index("element")
    state.update(current[[c for c in current if c in state.columns]])
    missing = current.index.difference(state.index)
    for element in missing:
        for column, value in current.loc[element].items():
            state.loc[element, column] = value
    state["scheduled_fixtures"] = [fixture_counts.get(int(t), 0) if pd.notna(t) else 0
                                    for t in state.team]
    state["actual_points"] = 0.0
    overlap = state.index.intersection(current.index)
    state.loc[overlap, "actual_points"] = current.loc[overlap, "actual_points"]
    state["price_source"] = np.where(state.index.isin(current.index), "gameweek_row", "carried_forward")
    state["eligibility_source"] = "unavailable"
    return state.reset_index()


def fixture_team_counts(fixtures: pd.DataFrame, event: int) -> dict[int, int]:
    subset = fixtures.loc[pd.to_numeric(fixtures.event, errors="coerce").eq(event)]
    teams = pd.concat([subset.team_h, subset.team_a]).astype(int)
    return teams.value_counts().to_dict()


def load_positions(season_dir: Path) -> dict[int, str]:
    players = pd.read_csv(season_dir / "players_raw.csv", encoding_errors="replace")
    ids = pd.to_numeric(players.id, errors="coerce").astype("Int64")
    positions = pd.to_numeric(players.element_type, errors="coerce").map(POSITION_MAP)
    return dict(zip(ids, positions))


def build_season(season: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    season_dir = DATA / season
    fixtures_path = season_dir / "fixtures.csv"
    if not fixtures_path.is_file():
        return pd.DataFrame(), {"season": season, "status": "no_fixture_table"}
    fixtures = pd.read_csv(fixtures_path)
    fixture_lookup = fixtures.set_index("id")[["team_h", "team_a"]]
    positions = load_positions(season_dir)
    files = sorted((season_dir / "gws").glob("gw*.csv"),
                   key=lambda p: int(re.search(r"gw(\d+)", p.stem).group(1)))
    state = pd.DataFrame(columns=["element", "name", "position", "team", "price"])
    outputs, event_audit = [], []
    for path in files:
        event = int(re.search(r"gw(\d+)", path.stem).group(1))
        raw = pd.read_csv(path, encoding_errors="replace")
        raw["element"] = pd.to_numeric(raw.element, errors="coerce")
        raw["fixture"] = pd.to_numeric(raw.fixture, errors="coerce")
        merged = raw.merge(fixture_lookup, left_on="fixture", right_index=True, how="left")
        was_home = merged.was_home.astype(str).str.lower().eq("true")
        merged["own_team_id"] = np.where(was_home, merged.team_h, merged.team_a)
        if "position" not in merged:
            merged["position"] = merged.element.astype("Int64").map(positions)
        else:
            merged["position"] = merged.position.fillna(merged.element.astype("Int64").map(positions))
        scheduled = aggregate_scheduled_rows(merged)
        counts = fixture_team_counts(fixtures, event)
        candidates = expand_seen_pool(state, scheduled, counts)
        candidates.insert(0, "event", event)
        candidates.insert(0, "season", season)
        candidates["is_blank"] = candidates.scheduled_fixtures.eq(0)
        candidates["is_double"] = candidates.scheduled_fixtures.gt(1)
        outputs.append(candidates)
        state = candidates[["element", "name", "position", "team", "price"]].copy()
        event_audit.append({"event": event, "candidate_rows": int(len(candidates)),
                            "scheduled_players": int(len(scheduled)),
                            "blank_players_reconstructed": int(candidates.is_blank.sum()),
                            "double_players": int(candidates.is_double.sum()),
                            "missing_price": int(candidates.price.isna().sum()),
                            "missing_team": int(candidates.team.isna().sum()),
                            "missing_position": int(candidates.position.isna().sum())})
    universe = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    summary = {
        "season": season, "status": "restricted_reconstruction", "gameweeks": len(event_audit),
        "rows": int(len(universe)), "unique_players": int(universe.element.nunique()) if len(universe) else 0,
        "gameweek_audit": event_audit,
    }
    return universe, summary


def main() -> None:
    frames, seasons = [], []
    for season in SEASON_LIST:
        frame, audit = build_season(season)
        seasons.append(audit)
        if not frame.empty:
            frames.append(frame)
    universe = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = {
        "classification": "restricted candidate-universe reconstruction, not verified deadline snapshots",
        "sources": {
            "membership": "players observed in the current or an earlier archived gameweek row",
            "scheduled_price": "value in the archived gameweek fixture row; exact capture time is not documented",
            "blank_price": "last observed archived value carried forward",
            "position": "gameweek row where present, otherwise season-end players_raw.csv",
            "eligibility": "unavailable historically",
            "schedule": "season-final fixtures.csv; may contain postponement/rescheduling hindsight",
        },
        "limitations": [
            "Players never observed before a blank gameweek cannot be reconstructed for that deadline.",
            "Transfers or price changes during a blank gameweek may be carried forward incorrectly.",
            "The archive does not verify status, can_select, or the fixture schedule as known at each deadline.",
        ],
        "seasons": seasons,
        "total_rows": int(len(universe)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    universe.to_csv(OUT / "candidate_universe.csv", index=False)
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved {len(universe):,} restricted player-gameweek rows and audit summary")


if __name__ == "__main__":
    main()
