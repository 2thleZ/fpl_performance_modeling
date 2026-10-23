#!/usr/bin/env python3
"""Evaluate likely-starter selection quality from saved OOS model predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from model_comparison import likely_starter_mask

RUN_COLS = ["context_size", "seed", "feature_spec", "model", "approach"]
PRICE_BINS = [-np.inf, 6, 8, 10, np.inf]
PRICE_LABELS = ["under_6", "6_to_8", "8_to_10", "10_plus"]


def collapse_player_gameweeks(predictions: pd.DataFrame) -> pd.DataFrame:
    """Sum double-gameweek fixtures after applying the pre-deadline starter screen."""
    d = predictions.loc[likely_starter_mask(predictions)].copy()
    group_cols = RUN_COLS + ["season", "event", "player_key"]
    collapsed = d.groupby(group_cols, as_index=False, observed=True).agg(
        position=("position", "first"),
        price_m=("price_m", "first"),
        fixture_count=("fixture", "nunique"),
        total_points=("total_points", "sum"),
        prediction=("prediction", "sum"),
    )
    collapsed["price_band"] = pd.cut(
        collapsed.price_m, PRICE_BINS, labels=PRICE_LABELS, right=False)
    return collapsed.dropna(subset=["price_band"])


def select_top_players(player_gameweeks: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Select exactly top_k candidates within every matched position-price group."""
    group_cols = RUN_COLS + ["season", "event", "position", "price_band"]
    parts = []
    for _, group in player_gameweeks.groupby(group_cols, observed=True, sort=False):
        if len(group) < top_k:
            continue
        chosen = group.sort_values(
            ["prediction", "player_key"], ascending=[False, True], kind="mergesort"
        ).head(top_k).copy()
        chosen["selected_rank"] = np.arange(1, top_k + 1)
        parts.append(chosen)
    if parts:
        return pd.concat(parts, ignore_index=True)
    empty = player_gameweeks.iloc[0:0].copy()
    empty["selected_rank"] = pd.Series(dtype=int)
    return empty


def selection_scores(selected: pd.DataFrame) -> pd.DataFrame:
    cols = RUN_COLS + ["season"]
    return selected.groupby(cols, as_index=False, observed=True).agg(
        groups=("event", "size"),
        gameweeks=("event", "nunique"),
        actual_points=("total_points", "sum"),
        predicted_points=("prediction", "sum"),
        actual_points_per_pick=("total_points", "mean"),
        predicted_points_per_pick=("prediction", "mean"),
    ).rename(columns={"groups": "selected_player_rows"})


def compare_fixture_selections(selected: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Pair each fixture-aware selection with its exact baseline selection group."""
    keys = ["context_size", "seed", "model", "approach", "season", "event",
            "position", "price_band"]
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(keys, observed=True, sort=False):
        by_spec = {name: part for name, part in group.groupby("feature_spec", sort=False)}
        baseline = by_spec.get("baseline")
        if baseline is None or len(baseline) != top_k:
            continue
        baseline_players = set(baseline.player_key)
        base = {"baseline_points": float(baseline.total_points.sum()),
                "baseline_prediction": float(baseline.prediction.sum()),
                "baseline_players": "|".join(sorted(baseline_players))}
        for spec in ("elo", "interaction"):
            comparison = by_spec.get(spec)
            if comparison is None or len(comparison) != top_k:
                continue
            comparison_players = set(comparison.player_key)
            row = dict(zip(keys, key))
            row.update(base)
            row.update({
                "comparison_feature_spec": spec,
                "comparison_points": float(comparison.total_points.sum()),
                "comparison_prediction": float(comparison.prediction.sum()),
                "comparison_players": "|".join(sorted(comparison_players)),
                "actual_points_gain": float(comparison.total_points.sum() -
                                             baseline.total_points.sum()),
                "overlap": int(len(baseline_players & comparison_players)),
                "changed_picks": int(top_k - len(baseline_players & comparison_players)),
                "picks": int(top_k),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_selection(frame: pd.DataFrame, reps: int, seed: int) -> dict[str, Any]:
    """Average seeds within gameweek, then resample independent gameweek clusters."""
    per_seed_event = frame.groupby(["season", "event", "seed"], as_index=False).agg(
        baseline_points=("baseline_points", "sum"),
        comparison_points=("comparison_points", "sum"),
        actual_points_gain=("actual_points_gain", "sum"),
        picks=("picks", "sum"),
        changed_picks=("changed_picks", "sum"),
    )
    events = per_seed_event.groupby(["season", "event"], as_index=False).mean(numeric_only=True)
    if events.empty:
        return {}
    base = events.baseline_points.to_numpy()
    comp = events.comparison_points.to_numpy()
    picks = events.picks.to_numpy()
    gain = comp - base
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(events), size=(reps, len(events)))
    gain_per_pick = gain[draws].sum(axis=1) / picks[draws].sum(axis=1)
    mean_gameweek_gain = gain[draws].mean(axis=1)
    return {
        "events": int(len(events)),
        "seeds": int(per_seed_event.seed.nunique()),
        "baseline_points_per_pick": float(base.sum() / picks.sum()),
        "comparison_points_per_pick": float(comp.sum() / picks.sum()),
        "gain_per_pick": float(gain.sum() / picks.sum()),
        "gain_per_pick_95": np.quantile(gain_per_pick, [0.025, 0.975]).tolist(),
        "mean_gameweek_gain": float(gain.mean()),
        "mean_gameweek_gain_95": np.quantile(mean_gameweek_gain, [0.025, 0.975]).tolist(),
        "changed_pick_rate": float(events.changed_picks.sum() / picks.sum()),
    }


def summarize_selection_comparisons(comparisons: pd.DataFrame, reps: int = 5000,
                                    seed: int = 20261023) -> pd.DataFrame:
    if comparisons.empty:
        return pd.DataFrame()
    config = ["context_size", "model", "approach", "comparison_feature_spec"]
    rows = []
    for key, group in comparisons.groupby(config, sort=False):
        scopes = [("pooled", group)] + [
            (str(season), part) for season, part in group.groupby("season", sort=False)]
        for scope, part in scopes:
            result = _bootstrap_selection(part, reps, seed)
            if result:
                result.update(dict(zip(config, key)))
                result["scope"] = scope
                rows.append(result)
    return pd.DataFrame(rows)


def summarize_seed_seasons(comparisons: pd.DataFrame) -> pd.DataFrame:
    if comparisons.empty:
        return pd.DataFrame()
    config = ["context_size", "seed", "model", "approach",
              "comparison_feature_spec", "season"]
    grouped = comparisons.groupby(config, as_index=False, sort=False).agg(
        groups=("picks", "size"),
        baseline_points=("baseline_points", "sum"),
        comparison_points=("comparison_points", "sum"),
        actual_points_gain=("actual_points_gain", "sum"),
        picks=("picks", "sum"),
        changed_picks=("changed_picks", "sum"),
        gameweeks=("event", "nunique"),
    )
    grouped["baseline_points_per_pick"] = grouped.baseline_points / grouped.picks
    grouped["comparison_points_per_pick"] = grouped.comparison_points / grouped.picks
    grouped["gain_per_pick"] = grouped.actual_points_gain / grouped.picks
    grouped["changed_pick_rate"] = grouped.changed_picks / grouped.picks
    return grouped


def _calibration(actual: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(probability)), probability])
    intercept, slope = np.linalg.lstsq(design, actual, rcond=None)[0]
    return float(intercept), float(slope)


def participation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score participation and points conditional on participation separately."""
    if "conditional_prediction" not in predictions:
        predictions = predictions.assign(conditional_prediction=np.nan)
    d = predictions.loc[predictions.approach.eq("two_stage") &
                        predictions.play_probability.notna()].copy()
    if d.empty:
        return pd.DataFrame()
    rows = []
    config = ["context_size", "seed", "feature_spec", "model"]
    for key, group in d.groupby(config, sort=False):
        scopes = [("pooled", group)] + [
            (str(season), part) for season, part in group.groupby("season", sort=False)]
        for scope, scoped in scopes:
            for population, part in (
                ("all", scoped),
                ("likely_starter", scoped.loc[likely_starter_mask(scoped)]),
            ):
                if part.empty:
                    continue
                actual_play = part.minutes.gt(0).astype(int).to_numpy()
                probability = part.play_probability.to_numpy(dtype=float)
                intercept, slope = _calibration(actual_play, probability)
                played = part.loc[part.minutes.gt(0) & part.conditional_prediction.notna()]
                row = dict(zip(config, key))
                row.update({
                    "scope": scope,
                    "population": population,
                    "n": int(len(part)),
                    "participation_rate": float(actual_play.mean()),
                    "brier": float(brier_score_loss(actual_play, probability)),
                    "roc_auc": (float(roc_auc_score(actual_play, probability))
                                if np.unique(actual_play).size == 2 else np.nan),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "conditional_n": int(len(played)),
                    "conditional_mae": (float(mean_absolute_error(
                        played.total_points, played.conditional_prediction)) if len(played) else np.nan),
                    "conditional_rmse": (float(np.sqrt(mean_squared_error(
                        played.total_points, played.conditional_prediction))) if len(played) else np.nan),
                })
                rows.append(row)
    return pd.DataFrame(rows)


def analyze(predictions: pd.DataFrame, top_k: int = 3, bootstrap_reps: int = 5000
            ) -> dict[str, pd.DataFrame]:
    player_gameweeks = collapse_player_gameweeks(predictions)
    selected = select_top_players(player_gameweeks, top_k)
    comparisons = compare_fixture_selections(selected, top_k)
    return {
        "selection_scores": selection_scores(selected),
        "selection_comparisons": comparisons,
        "selection_season_seed": summarize_seed_seasons(comparisons),
        "selection_summary": summarize_selection_comparisons(
            comparisons, reps=bootstrap_reps),
        "participation_metrics": participation_metrics(predictions),
        "top_picks": selected,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No completed comparisons were available."
    d = frame[columns].copy()
    for column in d.select_dtypes(include=["number"]):
        d[column] = d[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join([header, rule] + body)


def write_findings(outputs: dict[str, pd.DataFrame], path: Path, top_k: int) -> None:
    summary = outputs["selection_summary"]
    pooled = (summary.loc[summary.scope.eq("pooled")].copy()
              if "scope" in summary else pd.DataFrame())
    if not pooled.empty:
        pooled["gain_95"] = pooled.gain_per_pick_95.map(
            lambda x: f"{x[0]:.4f} to {x[1]:.4f}")
    seasons = (summary.loc[summary.scope.ne("pooled")].copy()
               if "scope" in summary else pd.DataFrame())
    if not seasons.empty:
        seasons["gain_95"] = seasons.gain_per_pick_95.map(
            lambda x: f"{x[0]:.4f} to {x[1]:.4f}")
    columns = ["context_size", "model", "approach", "comparison_feature_spec",
               "baseline_points_per_pick", "comparison_points_per_pick", "gain_per_pick",
               "gain_95", "changed_pick_rate"]
    text = f"""# Colab model and selection findings

This report evaluates the actual points earned by the top {top_k} pre-deadline
likely starters within each position and price band. Double gameweeks are summed
to one player-gameweek before ranking. Each fixture-aware selection is paired
with the baseline selection from the same season, gameweek, position, price
band, model, context size, and random seed.

## Pooled paired selection results

{_markdown_table(pooled, columns) if not pooled.empty else 'No completed comparisons were available.'}

## Season-by-season paired selection results

{_markdown_table(seasons, ['scope'] + columns) if not seasons.empty else 'No completed season comparisons were available.'}

The confidence intervals resample season-gameweeks after averaging results over
random seeds. Seeds measure training-sample sensitivity and are not treated as
independent seasons. `changed_pick_rate` reports the share of selected slots
that differ from the baseline ranking.

## Interpretation limits

The likely-starter screen uses only earlier minutes, but the archived candidate
pool is incomplete. Prices are not verified deadline snapshots, eligibility and
injury status are unavailable, blank-gameweek players are absent from the
fixture-row prediction data, and completed fixture files may contain schedule
hindsight. These are matched ranking diagnostics rather than a transfer-policy
simulation.
"""
    path.write_text(text)


def save_outputs(outputs: dict[str, pd.DataFrame], out: Path, top_k: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        suffix = ".csv.gz" if name == "top_picks" else ".csv"
        frame.to_csv(out / f"{name}{suffix}", index=False,
                     compression="gzip" if suffix.endswith(".gz") else None)
    payload = {
        "definition": {
            "likely_starter": "minutes6 >= 45 and rate60 >= 0.5 using earlier fixtures only",
            "price_bands": dict(zip(PRICE_LABELS, ["<6", "6-<8", "8-<10", "10+"])),
            "top_k": top_k,
            "double_gameweeks": "fixture forecasts and realized points summed before ranking",
            "uncertainty": "gameweek-cluster bootstrap after averaging random seeds",
        },
        "selection_summary": json.loads(outputs["selection_summary"].to_json(orient="records")),
        "participation_metrics": json.loads(
            outputs["participation_metrics"].to_json(orient="records")),
    }
    (out / "selection_analysis_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False))
    write_findings(outputs, out / "FINDINGS.md", top_k)


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["run_id", "season", "event", "player_key", "fixture"], keep="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("selection_results"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    outputs = analyze(predictions, args.top_k, args.bootstrap_reps)
    save_outputs(outputs, args.out, args.top_k)
    print(f"Saved selection analysis for {len(predictions):,} fixture predictions to {args.out}")


if __name__ == "__main__":
    main()
