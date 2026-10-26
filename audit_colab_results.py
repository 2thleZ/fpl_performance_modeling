#!/usr/bin/env python3
"""Audit saved forecasts and selections; never fit models or alter source results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from colab_experiment import markdown_table
from selection_analysis import PRICE_BINS, PRICE_LABELS, _bootstrap_selection

CONFIG = ["context_size", "model", "approach", "comparison_feature_spec"]
KEYS = ["seed", "season", "event", "player_key", "fixture"]


def selection_breakdown(comparisons: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows = []
    for config, part in comparisons.groupby(CONFIG, sort=True):
        for dimensions in [[], ["season"], ["position"], ["price_band"],
                           ["season", "position"], ["position", "price_band"]]:
            groups = [((), part)] if not dimensions else part.groupby(dimensions, observed=True)
            for key, group in groups:
                key = key if isinstance(key, tuple) else (key,)
                row = dict(zip(CONFIG, config))
                row.update({name: "all" for name in ["season", "position", "price_band"]})
                row.update(dict(zip(dimensions, key)))
                row.update(_bootstrap_selection(group, reps, 20261026))
                row["groups"] = len(group)
                row["seasons"] = group.season.nunique()
                row["ci_low"], row["ci_high"] = row.pop("gain_per_pick_95")
                rows.append(row)
    return pd.DataFrame(rows)


def paired_forecast(a: pd.DataFrame, b: pd.DataFrame, reps: int) -> pd.DataFrame:
    d = a.merge(b, on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    if len(d) != len(a) or len(d) != len(b):
        raise ValueError("Forecast comparisons must have identical row coverage")
    if not np.allclose(d.total_points_a, d.total_points_b):
        raise ValueError("Paired targets disagree")
    rows = []
    for population, part in [("all", d), ("likely_starter", d.loc[
            d.minutes6_a.ge(45) & d.rate60_a.ge(.5)])]:
        e = part.assign(ae_a=(part.prediction_a-part.total_points_a).abs(),
                        ae_b=(part.prediction_b-part.total_points_a).abs(),
                        se_a=(part.prediction_a-part.total_points_a)**2,
                        se_b=(part.prediction_b-part.total_points_a)**2)
        event = e.groupby(["season", "event", "seed"]).agg(
            ae_a=("ae_a", "sum"), ae_b=("ae_b", "sum"),
            se_a=("se_a", "sum"), se_b=("se_b", "sum"), n=("ae_a", "size"))
        # Average loss sums across seeds, not predictions; seeds are not extra weeks.
        event = event.groupby(["season", "event"]).mean()
        draws = np.random.default_rng(20261026).integers(0, len(event), (reps, len(event)))
        n = event.n.to_numpy()
        stats = {c: event[c].to_numpy() for c in ["ae_a", "ae_b", "se_a", "se_b"]}
        row = {"population": population, "gameweeks": len(event),
               "observations_per_seed": len(part) // part.seed.nunique(),
               "seeds": part.seed.nunique()}
        for metric, loss in [("mae", "ae"), ("rmse", "se")]:
            transform = np.sqrt if metric == "rmse" else lambda x: x
            av, bv = stats[loss+"_a"], stats[loss+"_b"]
            row[metric+"_a"] = float(transform(av.sum()/n.sum()))
            row[metric+"_b"] = float(transform(bv.sum()/n.sum()))
            row[metric+"_gain"] = row[metric+"_a"] - row[metric+"_b"]
            gains = transform(av[draws].sum(1)/n[draws].sum(1)) - transform(bv[draws].sum(1)/n[draws].sum(1))
            row[metric+"_low"], row[metric+"_high"] = np.quantile(gains, [.025, .975])
        rows.append(row)
    return pd.DataFrame(rows)


def audit(source: Path, out: Path, reps: int = 5000) -> None:
    if source.resolve() == out.resolve():
        raise ValueError("Audit output must be separate from source results")
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([r for p in sorted((source / "phases").glob("*/summary.json"))
                         for r in json.loads(p.read_text())["runs"]])
    if runs.run_id.duplicated().any():
        raise ValueError("Duplicate run IDs in source summaries")
    manifest = json.loads((source / "manifest.json").read_text())
    (out / "audit_metadata.json").write_text(json.dumps({
        "source_git_commit": manifest["git_commit"],
        "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "successful_runs": int(runs.status.eq("ok").sum()),
        "failed_runs": int(runs.status.ne("ok").sum()),
        "bootstrap_reps": reps, "bootstrap_seed": 20261026,
        "interval_unit": "season-gameweek after averaging losses or selection totals across seeds",
        "status": "exploratory; comparisons chosen after viewing the original report",
    }, indent=2))
    coverage = runs.groupby(["model", "approach", "context_size", "status"]).size().rename("runs").reset_index()
    coverage.to_csv(out / "coverage.csv", index=False)
    expected = json.loads((source / "config_effective.json").read_text())["test_seasons"]
    complete = runs.groupby(["model", "context_size", "approach", "seed", "feature_spec"]).agg(
        successes=("status", lambda x: int(x.eq("ok").sum())))
    complete = complete.loc[complete.successes.eq(len(expected))].reset_index()
    comparisons = pd.read_csv(source / "selection/selection_comparisons.csv")
    # Exclude partial model/configuration coverage from pooled conclusions.
    good_configs = complete.groupby(["model", "context_size", "approach"]).size()
    # The source experiment prespecified 3 seeds x 3 feature specifications.
    cfg = json.loads((source / "config_effective.json").read_text())
    valid = []
    for phase in cfg["phases"].values():
        needed = len(phase["seeds"]) * len(phase["feature_specs"])
        for model in phase["models"]:
            for size in phase["context_sizes"]:
                for approach in phase["approaches"]:
                    key = (model, size, approach)
                    if good_configs.get(key, 0) == needed:
                        valid.append(key)
    valid_frame = pd.DataFrame(valid, columns=["model", "context_size", "approach"])
    comparisons = comparisons.merge(valid_frame, validate="many_to_one")
    breakdown = selection_breakdown(comparisons, reps)
    breakdown.to_csv(out / "fixture_selection_breakdown.csv", index=False)

    # Prespecified in this audit after inspecting the original report: exploratory comparisons.
    configs = {
        "tabpfn20_elo": ("tabpfn", 20000, "elo"),
        "tabpfn20_base": ("tabpfn", 20000, "baseline"),
        "ridge_full_interaction": ("ridge", 0, "interaction"),
        "ridge20_interaction": ("ridge", 20000, "interaction"),
    }
    parts = {k: [] for k in configs}
    columns = KEYS + ["model", "context_size", "feature_spec", "approach",
                      "total_points", "prediction", "minutes6", "rate60", "position", "price_m"]
    for chunk in pd.read_csv(source / "combined_predictions.csv.gz", usecols=columns, chunksize=200000):
        for name, (model, size, spec) in configs.items():
            mask = chunk.model.eq(model) & chunk.context_size.eq(size) & chunk.feature_spec.eq(spec) & chunk.approach.eq("direct")
            if mask.any():
                parts[name].append(chunk.loc[mask, KEYS + ["total_points", "prediction", "minutes6", "rate60", "position", "price_m"]])
    frames = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    pool = frames["tabpfn20_base"]
    pool = pool.loc[pool.seed.eq(pool.seed.min()) & pool.minutes6.ge(45) & pool.rate60.ge(.5)].copy()
    pool = pool.drop_duplicates(["season", "event", "player_key"])
    pool["price_band"] = pd.cut(pool.price_m, PRICE_BINS, labels=PRICE_LABELS, right=False)
    candidates = pool.groupby(["season", "event", "position", "price_band"], observed=True).size().rename("candidates").reset_index()
    candidate_coverage = candidates.groupby(["position", "price_band"], observed=True).agg(
        observed_groups=("candidates", "size"), scored_groups=("candidates", lambda x: int(x.ge(3).sum())),
        forced_top_three_groups=("candidates", lambda x: int(x.eq(3).sum())),
        groups_with_choice=("candidates", lambda x: int(x.gt(3).sum()))).reset_index()
    candidate_coverage.to_csv(out / "candidate_group_coverage.csv", index=False)
    pairs = [("tabpfn20_base", "tabpfn20_elo"),
             ("ridge_full_interaction", "tabpfn20_elo"),
             ("ridge20_interaction", "tabpfn20_elo")]
    forecasts = []
    for a, b in pairs:
        result = paired_forecast(frames[a], frames[b], reps)
        result["reference"], result["comparison"] = a, b
        forecasts.append(result)
    forecast = pd.concat(forecasts, ignore_index=True)
    forecast.to_csv(out / "paired_forecast.csv", index=False)

    picks = pd.read_csv(source / "selection/top_picks.csv.gz")
    group = ["seed", "season", "event", "position", "price_band"]
    selected = {}
    for name, (model, size, spec) in configs.items():
        p = picks.loc[picks.model.eq(model) & picks.context_size.eq(size) &
                      picks.feature_spec.eq(spec) & picks.approach.eq("direct")]
        selected[name] = p.groupby(group, as_index=False).agg(
            points=("total_points", "sum"), picks=("player_key", "size"),
            price=("price_m", "sum"), players=("player_key", frozenset))
    selection_rows = []
    for a, b in pairs:
        d = selected[a].merge(selected[b], on=group, suffixes=("_a", "_b"), validate="one_to_one")
        if len(d) != len(selected[a]) or len(d) != len(selected[b]) or not d.picks_a.eq(d.picks_b).all():
            raise ValueError("Selection coverage is not identical")
        d = d.assign(baseline_points=d.points_a, comparison_points=d.points_b,
                     actual_points_gain=d.points_b-d.points_a, picks=d.picks_a,
                     changed_picks=[len(b-a) for a, b in zip(d.players_a, d.players_b)])
        scopes = [("pooled", d)] + list(d.groupby("season")) + list(d.groupby("position"))
        for scope, part in scopes:
            result = _bootstrap_selection(part, reps, 20261026)
            result.update(reference=a, comparison=b, scope=scope,
                          mean_price_a=float(part.price_a.sum()/part.picks.sum()),
                          mean_price_b=float(part.price_b.sum()/part.picks.sum()))
            result["ci_low"], result["ci_high"] = result.pop("gain_per_pick_95")
            selection_rows.append(result)
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(out / "paired_model_selection.csv", index=False)
    focus = breakdown.loc[breakdown.model.eq("tabpfn") & breakdown.context_size.eq(20000) &
                          breakdown.approach.eq("direct") & breakdown.comparison_feature_spec.eq("elo")]
    (out / "AUDIT.md").write_text("""# Colab result audit

This audit uses saved predictions only. Positive gains favor the comparison.
All comparisons are exploratory: configurations were chosen after viewing the original report.
Intervals resample season-gameweeks after averaging seeds. They are pointwise,
not adjusted for multiple comparisons; serial dependence across weeks may narrow them too much.
Partial TabICL configurations are excluded from comparative conclusions.

## Matched forecast errors

""" + markdown_table(forecast, ["reference", "comparison", "population", "mae_a", "mae_b", "mae_gain", "mae_low", "mae_high", "rmse_a", "rmse_b", "rmse_gain", "rmse_low", "rmse_high"]) + """

## Matched top-three selections

Each selection is within season, gameweek, position and price band. These are
independent ranking tasks, not one legal FPL squad or a sequence of transfers.
Price bands do not enforce identical spending; mean selected prices are shown.

""" + markdown_table(selection, ["reference", "comparison", "scope", "baseline_points_per_pick", "comparison_points_per_pick", "gain_per_pick", "ci_low", "ci_high", "mean_price_a", "mean_price_b"]) + """

## TabPFN 20k Elo versus its baseline: subgroup checks

""" + markdown_table(focus, ["season", "position", "price_band", "events", "seeds", "gain_per_pick", "ci_low", "ci_high"]) + """

## Candidate counts and forced selections

A group with exactly three candidates forces every model to select all three.
Its zero gain is not evidence of no fixture effect. Groups with fewer than three
are excluded altogether. This particularly limits premium-player conclusions.

""" + markdown_table(candidate_coverage, list(candidate_coverage.columns)) + """

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
""")
    print(f"Audit written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("fpl_colab_results"))
    parser.add_argument("--out", type=Path, default=Path("colab_audit_results"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    audit(args.source, args.out, args.bootstrap_reps)
