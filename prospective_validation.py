#!/usr/bin/env python3
"""Chronological, pre-deadline checks of fixture-strength forecasting value."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fpl_fixture_hypothesis import SEASONS, download_season, get_fixture_data

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "prospective_results"
SEASON_LIST = SEASONS + ("2025-26",)
TEST_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
BASE = ["quality6", "quality_long", "minutes6", "rate60", "prior_count",
        "price_prior", "home", "fwd", "event_scaled", "own_100"]


def strict_elo(fixtures: pd.DataFrame, season: str, previous: dict[str, float] | None = None,
               names: dict[int, str] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Freeze at the earliest kickoff of a GW; update only from earlier kickoffs."""
    f = fixtures.copy()
    f["kickoff_time"] = pd.to_datetime(f["kickoff_time"], utc=True, errors="coerce")
    for c in ("id", "event", "team_h", "team_a", "team_h_score", "team_a_score"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.dropna(subset=["id", "event", "team_h", "team_a", "team_h_score",
                         "team_a_score", "kickoff_time"]).copy()
    f = f.sort_values(["kickoff_time", "id"]).reset_index(drop=True)
    names = names or {int(t): f"{season}:{int(t)}" for t in pd.concat([f.team_h, f.team_a]).unique()}
    ratings = {int(t): 1500. + .5 * ((previous or {}).get(name, 1500.) - 1500.)
               for t, name in names.items()}
    records = []
    match_idx = 0
    for event, cutoff in f.groupby("event")["kickoff_time"].min().sort_values().items():
        while match_idx < len(f) and f.loc[match_idx, "kickoff_time"] < cutoff:
            m = f.iloc[match_idx]
            h, a = int(m.team_h), int(m.team_a)
            hr, ar = ratings.get(h, 1500.), ratings.get(a, 1500.)
            expected = 1. / (1. + 10. ** ((ar - hr - 65.) / 400.))
            actual = 1. if m.team_h_score > m.team_a_score else (.5 if m.team_h_score == m.team_a_score else 0.)
            delta = 24. * (actual - expected)
            ratings[h], ratings[a] = hr + delta, ar - delta
            match_idx += 1
        for m in f.loc[f.event.eq(event)].itertuples():
            records.append((season, int(m.id), ratings.get(int(m.team_h), 1500.),
                            ratings.get(int(m.team_a), 1500.)))
    while match_idx < len(f):
        m = f.iloc[match_idx]
        h, a = int(m.team_h), int(m.team_a)
        hr, ar = ratings.get(h, 1500.), ratings.get(a, 1500.)
        expected = 1. / (1. + 10. ** ((ar - hr - 65.) / 400.))
        actual = 1. if m.team_h_score > m.team_a_score else (.5 if m.team_h_score == m.team_a_score else 0.)
        delta = 24. * (actual - expected)
        ratings[h], ratings[a] = hr + delta, ar - delta
        match_idx += 1
    return pd.DataFrame(records, columns=["season", "fixture", "home_elo", "away_elo"]), {
        name: ratings.get(t, 1500.) for t, name in names.items()}


def add_elo(d: pd.DataFrame, carried: bool, data_root: Path = DATA,
            cache_root: Path | None = None) -> pd.DataFrame:
    all_ratings = []
    previous = None
    cache_root = cache_root or ROOT / "cache"
    for season in SEASON_LIST:
        _, f = download_season(season, cache_root, data_root)
        names = None
        teams = data_root / season / "teams.csv"
        if carried and teams.is_file():
            t = pd.read_csv(teams)
            names = dict(zip(t.id.astype(int), t.name.astype(str).str.lower().str.strip()))
        rating, previous = strict_elo(f, season, previous if carried and names else None, names)
        all_ratings.append(rating)
    e = pd.concat(all_ratings, ignore_index=True)
    frame = d.drop(columns=[c for c in ("own_elo", "opp_elo") if c in d])
    frame = frame.merge(e, on=["season", "fixture"], validate="many_to_one")
    frame["own_elo"] = np.where(frame.was_home, frame.home_elo, frame.away_elo)
    frame["opp_elo"] = np.where(frame.was_home, frame.away_elo, frame.home_elo)
    return frame.drop(columns=["home_elo", "away_elo"])


def deadline_features(d: pd.DataFrame) -> pd.DataFrame:
    """Histories only from fixtures kicked off before the first match of the GW."""
    d = d.copy()
    d["kickoff_time"] = pd.to_datetime(d.kickoff_time, utc=True)
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    d["event"] = pd.to_numeric(d.event, errors="coerce")
    cutoff = d.groupby(["season", "event"])["kickoff_time"].transform("min")
    d["deadline_cutoff"] = cutoff
    history = []
    for key, g in d.groupby("player_key", sort=False):
        g = g.sort_values("kickoff_time")
        dates = g.kickoff_time.dt.tz_localize(None).to_numpy()
        points = g.total_points.to_numpy(dtype=float)
        minutes = g.minutes.to_numpy(dtype=float)
        prices = g.value.to_numpy(dtype=float)
        for event, sub in g.groupby("event", sort=False):
            n = int(np.searchsorted(dates, sub.deadline_cutoff.iloc[0].tz_localize(None).to_datetime64(), side="left"))
            if n < 5:
                continue
            q6 = points[max(0, n-6):n]
            m6 = minutes[max(0, n-6):n]
            # All previous season fixtures, with five virtual position-average
            # fixtures to reduce volatility for short histories.
            long_q = (points[:n].sum() + 5. * 2.5) / (n + 5.)
            history.append((key, event, n, q6.mean(), long_q, m6.mean(),
                            np.mean(m6 >= 60), prices[n-1]))
    h = pd.DataFrame(history, columns=["player_key", "event", "prior_count", "quality6",
                                       "quality_long", "minutes6", "rate60", "price_prior"])
    d = d.merge(h, on=["player_key", "event"], validate="many_to_one")
    d = d.loc[d.event.ge(9) & d.price_prior.notna()].copy()
    d["home"] = d.was_home.astype(float)
    d["fwd"] = d.position.eq("FWD").astype(float)
    d["event_scaled"] = d.event / 38.
    d["own_100"] = (d.own_elo - 1500.) / 100.
    d["opp_100"] = (d.opp_elo - 1500.) / 100.
    d["quality_x_opp"] = (d.quality_long - 2.5) * d.opp_100
    d["price_m"] = d.price_prior / 10.
    return d


def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.))


def score(y: np.ndarray, p: np.ndarray) -> dict:
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p)))}


def bootstrap_improvement(d: pd.DataFrame, first: str, second: str, reps: int = 2000) -> dict:
    clusters = [g for _, g in d.groupby(["season", "event"])]
    err_a = np.array([np.abs(g.total_points - g[first]).sum() for g in clusters])
    err_b = np.array([np.abs(g.total_points - g[second]).sum() for g in clusters])
    sq_a = np.array([np.square(g.total_points - g[first]).sum() for g in clusters])
    sq_b = np.array([np.square(g.total_points - g[second]).sum() for g in clusters])
    counts = np.array([len(g) for g in clusters])
    rng = np.random.default_rng(20261020)
    draws = rng.integers(0, len(clusters), size=(reps, len(clusters)))
    n = counts[draws].sum(axis=1)
    mae = (err_a[draws].sum(axis=1) - err_b[draws].sum(axis=1)) / n
    rmse = np.sqrt(sq_a[draws].sum(axis=1) / n) - np.sqrt(sq_b[draws].sum(axis=1) / n)
    return {"mae_gain": float((err_a.sum()-err_b.sum())/counts.sum()),
            "mae_95": np.quantile(mae, [.025, .975]).tolist(),
            "rmse_gain": float(np.sqrt(sq_a.sum()/counts.sum())-np.sqrt(sq_b.sum()/counts.sum())),
            "rmse_95": np.quantile(rmse, [.025, .975]).tolist(),
            "gameweeks": len(clusters)}


def validation(d: pd.DataFrame, population: str) -> tuple[dict, pd.DataFrame]:
    if population == "played_60":
        d = d.loc[d.minutes.ge(60)].copy()
    elif population != "all":
        raise ValueError(population)
    features = {"baseline": BASE, "elo": BASE + ["opp_100"],
                "interaction": BASE + ["opp_100", "quality_x_opp"]}
    parts = []
    split = {}
    for season in TEST_SEASONS:
        train = d.loc[d.season.lt(season)]
        test = d.loc[d.season.eq(season)].copy()
        if train.empty or test.empty:
            continue
        for name, cols in features.items():
            fitted = model().fit(train[cols], train.total_points)
            test[name] = fitted.predict(test[cols])
        split[season] = {name: score(test.total_points, test[name]) for name in features}
        parts.append(test)
    oos = pd.concat(parts, ignore_index=True)
    overall = {name: score(oos.total_points, oos[name]) for name in features}
    gain = {"elo_vs_baseline": bootstrap_improvement(oos, "baseline", "elo"),
            "interaction_vs_baseline": bootstrap_improvement(oos, "baseline", "interaction"),
            "interaction_vs_elo": bootstrap_improvement(oos, "elo", "interaction")}
    gameweeks = oos.groupby(["season", "event", "player_key"], as_index=False)[
        ["total_points", "baseline", "elo", "interaction"]].sum()
    gw_metrics = {name: score(gameweeks.total_points, gameweeks[name]) for name in features}
    gw_gain = {"elo_vs_baseline": bootstrap_improvement(gameweeks, "baseline", "elo"),
               "interaction_vs_baseline": bootstrap_improvement(gameweeks, "baseline", "interaction")}
    return {"split": split, "overall": overall, "gain": gain,
            "player_gameweek": {"overall": gw_metrics, "gain": gw_gain}}, oos


def quality_ablation(d: pd.DataFrame) -> dict:
    variants = {"recent_only": [c for c in BASE if c != "quality_long"],
                "long_only": [c for c in BASE if c != "quality6"],
                "both": BASE}
    scored = defaultdict(list)
    for season in TEST_SEASONS:
        train = d.loc[d.season.lt(season)]
        test = d.loc[d.season.eq(season)]
        for quality_name, base_cols in variants.items():
            specs = [("baseline", []), ("elo", ["opp_100"])]
            if quality_name == "both":
                specs.append(("interaction", ["opp_100", "quality_x_opp"]))
            for fixture_name, extra in specs:
                cols = base_cols + extra
                fitted = model().fit(train[cols], train.total_points)
                scored[(quality_name, fixture_name)].append((
                    test.total_points.to_numpy(), fitted.predict(test[cols])))
    return {quality_name: {
        fixture_name: score(np.concatenate([x[0] for x in scored[(quality_name, fixture_name)]]),
                            np.concatenate([x[1] for x in scored[(quality_name, fixture_name)]]))
        for fixture_name in (("baseline", "elo", "interaction") if quality_name == "both"
                             else ("baseline", "elo"))}
        for quality_name in variants}


def coefficient_checks(d: pd.DataFrame) -> dict:
    """Association diagnostics; not interpreted causally or as OOS effects."""
    out = {}
    for population, frame in (("all", d), ("played_60", d.loc[d.minutes.ge(60)])):
        for name, cols in {
            "recent_only": ["quality6", "home", "fwd", "event_scaled", "own_100", "opp_100"],
            "long_only": ["quality_long", "minutes6", "rate60", "price_prior", "home", "fwd", "event_scaled", "own_100", "opp_100"],
            "both": BASE + ["opp_100"],
            "both_interaction": BASE + ["opp_100", "quality_x_opp"],
            "without_own_elo": [c for c in BASE if c != "own_100"] + ["opp_100"],
        }.items():
            x = sm.add_constant(frame[cols].astype(float))
            fit = sm.OLS(frame.total_points, x, missing="drop").fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(frame.fixture_key)[0]})
            out[f"{population}:{name}"] = {
                k: {"coefficient": float(fit.params[k]), "se": float(fit.bse[k]),
                    "p": float(fit.pvalues[k])} for k in ("quality6", "quality_long", "own_100", "opp_100", "quality_x_opp") if k in cols}
        for name, cols in {
            "within_recent": ["quality6", "home", "own_100", "opp_100", "event_scaled"],
            "within_long": ["quality_long", "home", "own_100", "opp_100", "event_scaled"],
            "within_both": ["quality6", "quality_long", "home", "own_100", "opp_100", "event_scaled"],
        }.items():
            centered = frame[["player_key", "fixture_key", "total_points"] + cols].copy()
            centered[["total_points"] + cols] -= centered.groupby("player_key")[["total_points"] + cols].transform("mean")
            fit = sm.OLS(centered.total_points, centered[cols]).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(centered.fixture_key)[0]})
            out[f"{population}:{name}"] = {
                k: {"coefficient": float(fit.params[k]), "se": float(fit.bse[k]),
                    "p": float(fit.pvalues[k])} for k in ("quality6", "quality_long", "own_100", "opp_100") if k in cols}
    return out


def matchup_table(oos: pd.DataFrame, thresholds: dict[str, float],
                  likely_only: bool = False) -> list[dict]:
    d = oos.copy()
    if likely_only:
        d = d.loc[d.rate60.ge(.5) & d.minutes6.ge(45)].copy()
    q75, q25 = thresholds["quality_q75"], thresholds["quality_q25"]
    elo_lo, elo_hi = thresholds["opp_q33"], thresholds["opp_q67"]
    d["class"] = np.select([d.quality_long.ge(q75) & d.opp_100.ge(elo_hi),
                             d.quality_long.between(q25, q75) & d.opp_100.le(elo_lo)],
                            ["elite_strong", "mid_weak"], default="other")
    rows = []
    for pos in ("MID", "FWD"):
        for cls in ("elite_strong", "mid_weak"):
            for price in ("all", "6_to_8m", "8_to_10m"):
                g = d.loc[d.position.eq(pos) & d["class"].eq(cls)]
                if price == "6_to_8m":
                    g = g.loc[g.price_m.between(6, 8, inclusive="left")]
                elif price == "8_to_10m":
                    g = g.loc[g.price_m.between(8, 10, inclusive="left")]
                rows.append({"position": pos, "class": cls, "price_band": price,
                             "n": len(g), "mean_price_m": float(g.price_m.mean()),
                             "predicted_baseline": float(g.baseline.mean()),
                             "predicted_elo": float(g.elo.mean()),
                             "predicted_interaction": float(g.interaction.mean()),
                             "realized": float(g.total_points.mean())})
    return rows


def price_band_contrasts(oos: pd.DataFrame, thresholds: dict[str, float]) -> dict:
    """Descriptive cluster intervals for the £6–8m likely-starter groups."""
    d = oos.loc[oos.rate60.ge(.5) & oos.minutes6.ge(45) &
                oos.price_m.ge(6) & oos.price_m.lt(8)].copy()
    q25, q75 = thresholds["quality_q25"], thresholds["quality_q75"]
    low, high = thresholds["opp_q33"], thresholds["opp_q67"]
    d["group"] = np.select([d.quality_long.ge(q75) & d.opp_100.ge(high),
                            d.quality_long.between(q25, q75) & d.opp_100.le(low)],
                           ["elite_strong", "mid_weak"], default="other")
    out = {}
    for position in ("MID", "FWD"):
        sub = d.loc[d.position.eq(position) & d.group.ne("other")]
        cells = sub.groupby(["season", "event", "group"]).total_points.agg(["sum", "count"]).unstack(fill_value=0)
        a = cells["sum"].get("elite_strong", pd.Series(0, index=cells.index)).to_numpy()
        b = cells["sum"].get("mid_weak", pd.Series(0, index=cells.index)).to_numpy()
        na = cells["count"].get("elite_strong", pd.Series(0, index=cells.index)).to_numpy()
        nb = cells["count"].get("mid_weak", pd.Series(0, index=cells.index)).to_numpy()
        rng = np.random.default_rng(2026)
        draws = rng.integers(0, len(a), size=(5000, len(a)))
        differences = (a[draws].sum(axis=1) / na[draws].sum(axis=1) -
                       b[draws].sum(axis=1) / nb[draws].sum(axis=1))
        out[position] = {"realized_elite_minus_mid": float(a.sum()/na.sum()-b.sum()/nb.sum()),
                         "cluster_95": np.quantile(differences, [.025, .975]).tolist()}
    return out


def main() -> None:
    players, _ = get_fixture_data(SEASON_LIST, ROOT / "cache", DATA, positions=("MID", "FWD"))
    reset = deadline_features(add_elo(players, carried=False))
    results = {"definition": "MID/FWD fixture rows, GW 9+, at least five player fixtures strictly before first kickoff of gameweek, last observed price available; zero-minute rows retained; each DGW fixture is a separate target with same deadline features",
               "seasons": list(SEASON_LIST), "test_seasons": list(TEST_SEASONS),
               "counts": {"all": len(reset), "zero_minutes": int(reset.minutes.eq(0).sum()),
                          "played_60": int(reset.minutes.ge(60).sum())}}
    results["all"], oos_all = validation(reset, "all")
    results["played_60"], oos_60 = validation(reset, "played_60")
    results["quality_ablation_all"] = quality_ablation(reset)
    results["coefficient_checks"] = coefficient_checks(reset.loc[reset.season.le("2024-25")])
    train_reference = reset.loc[reset.season.lt(TEST_SEASONS[0])]
    thresholds = {"quality_q25": float(train_reference.quality_long.quantile(.25)),
                  "quality_q75": float(train_reference.quality_long.quantile(.75)),
                  "opp_q33": float(train_reference.opp_100.quantile(1/3)),
                  "opp_q67": float(train_reference.opp_100.quantile(2/3))}
    results["matchup_thresholds_from_2016_to_2021"] = thresholds
    results["matchups"] = matchup_table(oos_all, thresholds)
    likely_reference = train_reference.loc[train_reference.rate60.ge(.5) &
                                           train_reference.minutes6.ge(45)]
    likely_thresholds = {"quality_q25": float(likely_reference.quality_long.quantile(.25)),
                         "quality_q75": float(likely_reference.quality_long.quantile(.75)),
                         "opp_q33": float(likely_reference.opp_100.quantile(1/3)),
                         "opp_q67": float(likely_reference.opp_100.quantile(2/3))}
    results["likely_matchup_thresholds_from_2016_to_2021"] = likely_thresholds
    results["likely_starter_matchups"] = matchup_table(oos_all, likely_thresholds, likely_only=True)
    results["likely_price_6_to_8m_contrasts"] = price_band_contrasts(oos_all, likely_thresholds)
    # Like-for-like reset vs carryover, using years with available club-name maps.
    carry = deadline_features(add_elo(players, carried=True))
    both = reset.loc[reset.season.ge("2019-20")].copy()
    both = both.merge(carry[["season", "element", "fixture", "own_100", "opp_100"]],
                      on=["season", "element", "fixture"], suffixes=("_reset", "_carry"), validate="one_to_one")
    carry_result = {}
    carry_oos = {}
    for name in ("reset", "carry"):
        f = both.copy()
        f["own_100"] = f[f"own_100_{name}"]
        f["opp_100"] = f[f"opp_100_{name}"]
        f["quality_x_opp"] = (f.quality_long - 2.5) * f.opp_100
        carry_result[name], carry_oos[name] = validation(f, "all")
    joined = carry_oos["reset"][["season", "event", "element", "fixture", "total_points", "elo", "interaction"]].merge(
        carry_oos["carry"][["season", "event", "element", "fixture", "elo", "interaction"]],
        on=["season", "event", "element", "fixture"], suffixes=("_reset", "_carry"),
        validate="one_to_one")
    carry_result["direct_paired_gain"] = {
        variant: bootstrap_improvement(joined, f"{variant}_reset", f"{variant}_carry")
        for variant in ("elo", "interaction")}
    results["elo_comparison_2019_onward"] = carry_result
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    oos_all[["season", "event", "player_key", "position", "price_m", "total_points",
             "quality_long", "opp_100", "rate60", "minutes6", "baseline", "elo", "interaction"]].to_csv(
                 OUT / "oos_predictions.csv", index=False)
    print(f"Saved {OUT / 'summary.json'}; {results['all']['overall']['baseline']['n']:,} "
          "out-of-sample fixture rows")


if __name__ == "__main__":
    main()
