#!/usr/bin/env python3
"""Exploratory FPL fixture-strength hypothesis test.

Reads historical, third-party archived FPL CSVs on the user's computer.
Does not call the official FPL service. Run from the project root with:
python fpl_fixture_hypothesis/fpl_fixture_hypothesis.py --data-root data
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import argparse
import json
import math
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

SEASONS = ("2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
           "2021-22", "2022-23", "2023-24", "2024-25")
SUPPORTED_SEASONS = SEASONS + ("2025-26",)
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
COLS = ("season", "element", "position", "fixture", "minutes", "total_points", "was_home")
POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def read_csv(path: Path) -> pd.DataFrame:
    """Read archive files, including older seasons stored with Latin-1 names."""
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def reconstruct_fixtures(gws: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the fixture table for 2016-17 and 2017-18 from player-match rows."""
    p = gws.copy()
    home_flag = (p["was_home"] if p["was_home"].dtype == bool else
                 p["was_home"].astype(str).str.lower().eq("true"))
    p["was_home"] = home_flag
    event_col = "GW" if "GW" in p else "round"
    common = (p.groupby("fixture", as_index=False)
              .agg(event=(event_col, "first"), kickoff_time=("kickoff_time", "first"),
                   team_h_score=("team_h_score", "first"), team_a_score=("team_a_score", "first")))
    home = (p.loc[p["was_home"]].groupby("fixture", as_index=False)
            .agg(team_a=("opponent_team", "first")))
    away = (p.loc[~p["was_home"]].groupby("fixture", as_index=False)
            .agg(team_h=("opponent_team", "first")))
    fixtures = common.merge(home, on="fixture", validate="one_to_one").merge(
        away, on="fixture", validate="one_to_one")
    fixtures = fixtures.rename(columns={"fixture": "id"})
    if len(fixtures) != p["fixture"].nunique():
        raise ValueError("Could not reconstruct every fixture from gameweek rows")
    return fixtures


def add_positions(gws: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    if "position" in gws:
        return gws
    lookup = players.set_index("id")["element_type"].map(POSITION_NAMES)
    out = gws.copy()
    out["position"] = pd.to_numeric(out["element"], errors="coerce").map(lookup)
    if out["position"].isna().any():
        missing = out.loc[out["position"].isna(), "element"].nunique()
        raise ValueError(f"Could not map positions for {missing} player IDs")
    return out


def download_season(season: str, cache: Path, data_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data_root is not None:
        gws_path = data_root / season / "gws" / "merged_gw.csv"
        players_path = data_root / season / "players_raw.csv"
        fixtures_path = data_root / season / "fixtures.csv"
        if not gws_path.is_file():
            raise FileNotFoundError(f"Missing {season} gameweeks: {gws_path}")
        p = read_csv(gws_path)
        if "position" not in p:
            if not players_path.is_file():
                raise FileNotFoundError(f"Missing {season} player metadata: {players_path}")
            p = add_positions(p, read_csv(players_path))
        f = read_csv(fixtures_path) if fixtures_path.is_file() else reconstruct_fixtures(p)
        return p, f
    paths = {
        "gws": (f"{BASE}/{season}/gws/merged_gw.csv", cache / f"{season}_merged_gw.csv"),
        "players": (f"{BASE}/{season}/players_raw.csv", cache / f"{season}_players_raw.csv"),
    }
    out: dict[str, pd.DataFrame] = {}
    for k, (url, path) in paths.items():
        if k == "players" and "gws" in out and "position" in out["gws"]:
            continue
        if not path.exists() or path.stat().st_size == 0:
            print(f"Downloading {season} {k} from third-party historical repository...")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception:
                path.unlink(missing_ok=True)
                raise
        out[k] = read_csv(path)
    p = add_positions(out["gws"], out["players"]) if "position" not in out["gws"] else out["gws"]
    fixture_path = cache / f"{season}_fixtures.csv"
    if season in ("2016-17", "2017-18"):
        f = reconstruct_fixtures(p)
    else:
        if not fixture_path.exists() or fixture_path.stat().st_size == 0:
            urllib.request.urlretrieve(f"{BASE}/{season}/fixtures.csv", fixture_path)
        f = read_csv(fixture_path)
    return p, f


def pre_match_elo(fixtures: pd.DataFrame, k: float = 24.0, home_bonus: float = 65.0,
                  freeze_by_event: bool = False) -> pd.DataFrame:
    """Calculate chronological Elo ratings before a fixture or gameweek.

    Ratings are always frozen for concurrent kickoffs. With ``freeze_by_event``,
    every fixture in a gameweek receives the rating available at that deadline;
    this is required for pre-deadline forecasts and prevents double-gameweek
    results from informing a later fixture's forecast.
    """
    f = fixtures.copy()
    for c in ("id", "event", "team_h", "team_a", "team_h_score", "team_a_score"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f["kickoff_time"] = pd.to_datetime(f["kickoff_time"], utc=True, errors="coerce")
    f = f.dropna(subset=["id", "event", "team_h", "team_a", "team_h_score", "team_a_score", "kickoff_time"])
    f = f.sort_values(["kickoff_time", "id"]).reset_index(drop=True)
    ratings: dict[int, float] = {}
    records = []
    batch = "event" if freeze_by_event else "kickoff_time"
    for _, matches in f.groupby(batch, sort=True):
        delta: dict[int, float] = {}
        # Do not use outcomes from simultaneous fixtures to rate their opponents.
        for m in matches.itertuples(index=False):
            h, a = int(m.team_h), int(m.team_a)
            hr, ar = ratings.get(h, 1500.), ratings.get(a, 1500.)
            expected = 1. / (1. + 10. ** ((ar - hr - home_bonus) / 400.))
            actual = 1. if m.team_h_score > m.team_a_score else (0.5 if m.team_h_score == m.team_a_score else 0.)
            shift = k * (actual - expected)
            delta[h] = delta.get(h, 0.) + shift
            delta[a] = delta.get(a, 0.) - shift
            records.append((int(m.id), hr, ar))
        for t, change in delta.items():
            ratings[t] = ratings.get(t, 1500.) + change
    elo = pd.DataFrame(records, columns=["id", "pre_home_elo", "pre_away_elo"])
    return f.merge(elo, on="id", validate="one_to_one")


def get_fixture_data(seasons: tuple[str, ...], cache: Path,
                     data_root: Path | None = None,
                     freeze_elo_by_event: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_players, all_fixtures = [], []
    for season in seasons:
        p, f = download_season(season, cache, data_root)
        p["season"] = season
        p = p.loc[p["position"].isin(("MID", "FWD"))].copy()
        if "GW" not in p and "round" in p:
            p["GW"] = p["round"]
        missing = set(COLS).difference(p.columns)
        if missing:
            raise ValueError(f"Missing columns in {season}: {missing}")
        p["fixture"] = pd.to_numeric(p["fixture"], errors="coerce")
        p["element"] = pd.to_numeric(p["element"], errors="coerce")
        p["minutes"] = pd.to_numeric(p["minutes"], errors="coerce")
        p["total_points"] = pd.to_numeric(p["total_points"], errors="coerce")
        if p["was_home"].dtype != bool:
            p["was_home"] = p["was_home"].astype(str).str.lower().eq("true")
        p = p.dropna(subset=["fixture", "element", "minutes", "total_points"])
        p["fixture"] = p["fixture"].astype(int)
        p["element"] = p["element"].astype(int)
        duplicate_count = p.duplicated(["element", "fixture"]).sum()
        if duplicate_count:
            print(f"Warning: {season}: {duplicate_count} duplicate player-fixture records dropped")
            p = p.drop_duplicates(["element", "fixture"])
        elo = pre_match_elo(f, freeze_by_event=freeze_elo_by_event)
        elo["season"] = season
        # Attach season-specific Elo levels for descriptive tier comparisons.
        eligible = elo.loc[elo["event"] >= 9]
        pool = np.r_[eligible["pre_home_elo"].values, eligible["pre_away_elo"].values]
        low, high = np.quantile(pool, [.3333333, .6666667]) if len(pool) else (1490., 1510.)
        elo["elo_low_cut"] = low
        elo["elo_high_cut"] = high
        use = ["id", "season", "event", "kickoff_time", "team_h", "team_a",
               "pre_home_elo", "pre_away_elo", "elo_low_cut", "elo_high_cut"]
        # Use the fixture table's kickoff timestamp as the chronological source.
        m = p.drop(columns=["kickoff_time"], errors="ignore").merge(
                    elo[use], left_on=["season", "fixture"], right_on=["season", "id"],
                    how="inner", validate="many_to_one")
        m["own_elo"] = np.where(m["was_home"], m["pre_home_elo"], m["pre_away_elo"])
        m["opp_elo"] = np.where(m["was_home"], m["pre_away_elo"], m["pre_home_elo"])
        m["own_team_id"] = np.where(m["was_home"], m["team_h"], m["team_a"]).astype(int)
        m["opp_team_id"] = np.where(m["was_home"], m["team_a"], m["team_h"]).astype(int)
        if "opponent_team" in m.columns:
            check = pd.to_numeric(m["opponent_team"], errors="coerce")
            mismatch = (check.notna() & (check != m["opp_team_id"])).sum()
            if mismatch:
                print(f"Warning: {season}: {mismatch} opponent ID mismatches (excluded)")
                m = m.loc[check.isna() | (check == m["opp_team_id"])].copy()
        m["player_key"] = m["season"] + "_" + m["element"].astype(str)
        m["fixture_key"] = m["season"] + "_" + m["fixture"].astype(str)
        all_players.append(m)
        all_fixtures.append(elo)
        print(f"{season}: {len(m):,} attacker-fixture rows, {len(elo)} completed fixtures")
    all_p = pd.concat(all_players, ignore_index=True)
    all_p = all_p.sort_values(["player_key", "kickoff_time", "fixture"]).reset_index(drop=True)
    # Every historical 'recent' stat below is calculated before the current match.
    quality = np.full(len(all_p), np.nan)
    for indices in all_p.groupby("player_key", sort=False).indices.values():
        recent: deque[float] = deque(maxlen=8)
        for i in sorted(indices):
            if len(recent) >= 5:
                quality[i] = float(np.mean(recent))
            if all_p.at[i, "minutes"] >= 60:
                recent.append(float(all_p.at[i, "total_points"]))
    all_p["prior_8_points"] = quality
    return all_p, pd.concat(all_fixtures, ignore_index=True)


def tier(x: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.Categorical:
    vals = np.select([x < lo, x > hi], ["Weak", "Strong"], default="Medium")
    return pd.Categorical(vals, categories=["Weak", "Medium", "Strong"], ordered=True)


def model_df(appearances: pd.DataFrame, min_minutes: int = 60) -> pd.DataFrame:
    d = appearances.loc[(appearances["event"] >= 9) &
                        (appearances["minutes"] >= min_minutes) &
                        appearances["prior_8_points"].notna()].copy()
    if len(d) < 100:
        raise ValueError(f"Insufficient appearances: {len(d)}")
    d["opp_100"] = (d["opp_elo"] - 1500.) / 100.
    d["own_100"] = (d["own_elo"] - 1500.) / 100.
    d["home"] = d["was_home"].astype(float)
    d["prior_quality_mean"] = float(d["prior_8_points"].mean())
    d["quality_ctr"] = d["prior_8_points"] - d["prior_quality_mean"]
    d["quality_x_opp"] = d["quality_ctr"] * d["opp_100"]
    d["week"] = (d["event"] - 20.) / 10.
    d["opp_tier"] = tier(d["opp_elo"], d["elo_low_cut"], d["elo_high_cut"])
    d["own_tier"] = tier(d["own_elo"], d["elo_low_cut"], d["elo_high_cut"])
    q25, q75 = d["prior_8_points"].quantile([.25, .75])
    d["quality_class"] = np.where(d["prior_8_points"] >= q75, "Top quartile",
                                  np.where(d["prior_8_points"] <= q25, "Bottom quartile", "Middle"))
    return d


def fixed_effects_fit(d: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    features = ["opp_100", "own_100", "home", "quality_ctr", "quality_x_opp", "week"]
    cols = ["total_points"] + features
    # Demeaning separately for each player-season removes stable individual quality.
    averages = d.groupby("player_key")[cols].transform("mean")
    transformed = d[cols] - averages
    # Groups with one appearance contribute no within-player variation.
    keep = (d.groupby("player_key")["player_key"].transform("size") >= 2)
    transformed = transformed.loc[keep]
    d = d.loc[keep].copy()
    player_groups = pd.factorize(d["player_key"])[0]
    fixture_groups = pd.factorize(d["fixture_key"])[0]
    fit = sm.OLS(transformed["total_points"], transformed[features]).fit(
        cov_type="cluster", cov_kwds={"groups": np.column_stack([player_groups, fixture_groups])})
    return fit, d


def marginal_slope(fit: object, quality: float, reference_quality: float) -> dict:
    delta = quality - reference_quality
    b = fit.params
    v = fit.cov_params()
    estimate = b["opp_100"] + delta * b["quality_x_opp"]
    variance = v.loc["opp_100", "opp_100"] + 2 * delta * v.loc["opp_100", "quality_x_opp"] + delta ** 2 * v.loc["quality_x_opp", "quality_x_opp"]
    se = math.sqrt(max(float(variance), 0.))
    return {"slope_per_100_elo": float(estimate), "low_95": float(estimate - 1.96 * se),
            "high_95": float(estimate + 1.96 * se)}


def save_analysis(d: pd.DataFrame, fit: object, out: Path, sensitivity: object | None = None):
    out.mkdir(parents=True, exist_ok=True)
    coefs = pd.DataFrame({"estimate": fit.params, "se": fit.bse, "p_value": fit.pvalues,
                          "ci_low": fit.conf_int()[0], "ci_high": fit.conf_int()[1]})
    coefs.to_csv(out / "regression_coefficients.csv")
    group = (d.groupby(["position", "quality_class", "opp_tier"], observed=True)
             .agg(appearances=("total_points", "size"), avg_points=("total_points", "mean"),
                  median_points=("total_points", "median")).reset_index())
    group.to_csv(out / "opponent_tier_summary.csv", index=False)
    matchup = (d.groupby(["own_tier", "opp_tier", "position"], observed=True)
               .agg(appearances=("total_points", "size"), avg_points=("total_points", "mean"))
               .reset_index())
    matchup.to_csv(out / "team_matchups.csv", index=False)
    q25, q75 = d["prior_8_points"].quantile([.25, .75])
    center = float(d["prior_quality_mean"].iloc[0])
    slopes = {
        "bottom_quartile_quality": marginal_slope(fit, q25, center),
        "top_quartile_quality": marginal_slope(fit, q75, center),
    }
    metrics = {
        "seasons": sorted(d["season"].unique().tolist()),
        "appearances": len(d), "unique_players_per_season": d["player_key"].nunique(),
        "unique_fixtures": d["fixture_key"].nunique(),
        "analysis_condition": "MID/FWD; at least 60 minutes played; GW 9+; >=5 prior appearances of 60+ minutes",
        "quality_mean": center, "quality_q25": float(q25), "quality_q75": float(q75),
        "per_100_opponent_elo_slopes": slopes,
        "interaction": {
            "estimate": float(fit.params["quality_x_opp"]),
            "p_value": float(fit.pvalues["quality_x_opp"]),
            "ci_low": float(fit.conf_int().loc["quality_x_opp", 0]),
            "ci_high": float(fit.conf_int().loc["quality_x_opp", 1]),
        },
        "min_75_regression": None if sensitivity is None else {
            "opponent_coef": float(sensitivity.params["opp_100"]),
            "interaction_coef": float(sensitivity.params["quality_x_opp"]),
            "opponent_p": float(sensitivity.pvalues["opp_100"]),
            "interaction_p": float(sensitivity.pvalues["quality_x_opp"]),
        },
    }
    (out / "results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5))
    for cl in ("Bottom quartile", "Middle", "Top quartile"):
        sub = d[d["quality_class"] == cl]
        means = sub.groupby("opp_tier", observed=True)["total_points"].mean().reindex(["Weak", "Medium", "Strong"])
        ax.plot(means.index.astype(str), means.values, marker="o", label=cl)
    ax.set_title("Attacker FPL points vs pre-match opponent strength")
    ax.set_xlabel("Opponent strength tier (pre-match Elo)")
    ax.set_ylabel("Average FPL points per 60+ minute appearance")
    ax.legend(title="Prior 8-appearance form")
    fig.tight_layout()
    fig.savefig(out / "opponent_strength.png", dpi=160)
    plt.close(fig)
    print("\nRegression with player-season fixed effects and two-way-clustered standard errors:")
    print(coefs.round(4).to_string())
    print("\nSlope per +100 Elo against opponent for prior-form quartiles:")
    print(json.dumps(slopes, indent=2))
    print(f"\nFiles saved to {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--data-root", type=Path, help="Local archive data directory with season subfolders")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seasons", nargs="+", choices=SUPPORTED_SEASONS, default=list(SEASONS))
    args = parser.parse_args()
    players, fixtures = get_fixture_data(tuple(args.seasons), args.cache, args.data_root)
    print(f"Loaded {len(players):,} attacker-fixture rows and {len(fixtures):,} fixtures")
    d = model_df(players, min_minutes=60)
    fit, d = fixed_effects_fit(d)
    d75 = model_df(players, min_minutes=75)
    fit75, _ = fixed_effects_fit(d75)
    save_analysis(d, fit, args.out, sensitivity=fit75)


if __name__ == "__main__":
    main()
