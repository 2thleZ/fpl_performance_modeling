#!/usr/bin/env python3
"""Leakage-aware direct and two-stage FPL point-model comparison."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fpl_fixture_hypothesis import get_fixture_data
from prospective_validation import BASE, DATA, ROOT, SEASON_LIST, TEST_SEASONS, add_elo, deadline_features

DEFAULT_OUT = ROOT / "model_comparison_results"
FEATURE_SPECS = {
    "baseline": BASE,
    "elo": BASE + ["opp_100"],
    "interaction": BASE + ["opp_100", "quality_x_opp"],
}
KEYS = ["season", "event", "player_key", "fixture"]


def likely_starter_mask(frame: pd.DataFrame) -> pd.Series:
    """Pre-deadline playing-time screen; includes earlier zero-minute fixtures."""
    return frame["minutes6"].ge(45.0) & frame["rate60"].ge(0.5)


def sample_training_rows(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Uniformly sample prior rows, preserving the observed participation mix."""
    if size <= 0 or len(frame) <= size:
        return frame.copy()
    return frame.sample(n=size, replace=False, random_state=seed).sort_index()


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    design = np.column_stack([np.ones(len(prediction)), prediction])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "bias": float(np.mean(prediction - y)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
    }


def calibration_bins(y: np.ndarray, prediction: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    d = pd.DataFrame({"actual": y, "prediction": prediction})
    ranks = d.prediction.rank(method="first")
    d["bin"] = pd.qcut(ranks, q=min(bins, len(d)), labels=False, duplicates="drop")
    return [
        {"bin": int(i), "n": int(len(g)), "mean_prediction": float(g.prediction.mean()),
         "mean_actual": float(g.actual.mean())}
        for i, g in d.groupby("bin", observed=True)
    ]


def ranking_metrics(frame: pd.DataFrame, prediction_col: str = "prediction") -> dict[str, Any]:
    d = frame.copy()
    d["price_band"] = pd.cut(d.price_m, [-np.inf, 6, 8, 10, np.inf], right=False,
                             labels=["under_6", "6_to_8", "8_to_10", "10_plus"])
    gw = d.groupby(["season", "event", "player_key", "position", "price_band"],
                   observed=True, as_index=False)[["total_points", prediction_col]].sum()
    correlations, selected, population = [], [], []
    for _, group in gw.groupby(["season", "event", "position", "price_band"], observed=True):
        if len(group) < 5 or group[prediction_col].nunique() < 2 or group.total_points.nunique() < 2:
            continue
        correlation = spearmanr(group[prediction_col], group.total_points).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        chosen = group.nlargest(min(3, len(group)), prediction_col)
        selected.extend(chosen.total_points.tolist())
        population.extend(group.total_points.tolist())
    return {
        "ranking_groups": int(len(correlations)),
        "mean_spearman": float(np.mean(correlations)) if correlations else None,
        "top3_realized_mean": float(np.mean(selected)) if selected else None,
        "candidate_realized_mean": float(np.mean(population)) if population else None,
    }


def paired_improvement(frame: pd.DataFrame, first: str, second: str,
                       reps: int = 2000, seed: int = 20261020) -> dict[str, Any]:
    clusters = list(frame.groupby(["season", "event"], sort=False))
    mae_a = np.array([np.abs(g.total_points - g[first]).sum() for _, g in clusters])
    mae_b = np.array([np.abs(g.total_points - g[second]).sum() for _, g in clusters])
    sq_a = np.array([np.square(g.total_points - g[first]).sum() for _, g in clusters])
    sq_b = np.array([np.square(g.total_points - g[second]).sum() for _, g in clusters])
    counts = np.array([len(g) for _, g in clusters])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(reps, len(clusters)))
    n = counts[draws].sum(axis=1)
    mae_gain = (mae_a[draws].sum(axis=1) - mae_b[draws].sum(axis=1)) / n
    rmse_gain = np.sqrt(sq_a[draws].sum(axis=1) / n) - np.sqrt(sq_b[draws].sum(axis=1) / n)
    return {
        "mae_gain": float((mae_a.sum() - mae_b.sum()) / counts.sum()),
        "mae_gain_95": np.quantile(mae_gain, [0.025, 0.975]).tolist(),
        "rmse_gain": float(np.sqrt(sq_a.sum() / counts.sum()) - np.sqrt(sq_b.sum() / counts.sum())),
        "rmse_gain_95": np.quantile(rmse_gain, [0.025, 0.975]).tolist(),
        "gameweeks": int(len(clusters)),
    }


def _ridge_regressor() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))


def _ridge_classifier() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, random_state=0))


def build_estimator(model_name: str, task: str, seed: int, foundation_estimators: int,
                    device: str) -> Any:
    if model_name == "ridge":
        return _ridge_regressor() if task == "regression" else _ridge_classifier()
    if model_name == "catboost":
        if task == "regression":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                                     loss_function="RMSE", random_seed=seed,
                                     verbose=False, allow_writing_files=False)
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                  loss_function="Logloss", random_seed=seed,
                                  verbose=False, allow_writing_files=False)
    if model_name == "tabicl":
        from tabicl import TabICLClassifier, TabICLRegressor
        cls = TabICLRegressor if task == "regression" else TabICLClassifier
        return cls(n_estimators=foundation_estimators, device=None if device == "auto" else device,
                   random_state=seed, kv_cache=True, verbose=False)
    if model_name == "tabpfn":
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        cls = TabPFNRegressor if task == "regression" else TabPFNClassifier
        return cls.create_default_for_version(ModelVersion.V3_5_FAST,
                                              n_estimators=foundation_estimators,
                                              device=device, random_state=seed,
                                              fit_mode="fit_with_cache")
    raise ValueError(f"Unknown model: {model_name}")


def predict_mean(estimator: Any, model_name: str, x: pd.DataFrame) -> np.ndarray:
    if model_name in {"tabicl", "tabpfn"}:
        return np.asarray(estimator.predict(x, output_type="mean"), dtype=float)
    return np.asarray(estimator.predict(x), dtype=float)


def predict_play_probability(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(x), dtype=float)
    classes = np.asarray(estimator.classes_)
    return probabilities[:, int(np.flatnonzero(classes == 1)[0])]


def run_prediction(model_name: str, approach: str, train: pd.DataFrame,
                   test: pd.DataFrame, features: list[str], seed: int,
                   foundation_estimators: int, device: str) -> tuple[np.ndarray, np.ndarray | None, int]:
    x_train, x_test = train[features], test[features]
    if approach == "direct":
        model = build_estimator(model_name, "regression", seed, foundation_estimators, device)
        model.fit(x_train, train.total_points)
        return predict_mean(model, model_name, x_test), None, len(train)
    classifier = build_estimator(model_name, "classification", seed, foundation_estimators, device)
    played = train.minutes.gt(0).astype(int)
    classifier.fit(x_train, played)
    play_probability = predict_play_probability(classifier, x_test)
    conditional = train.loc[played.eq(1)]
    regressor = build_estimator(model_name, "regression", seed, foundation_estimators, device)
    regressor.fit(conditional[features], conditional.total_points)
    conditional_mean = predict_mean(regressor, model_name, x_test)
    return play_probability * conditional_mean, play_probability, len(conditional)


def prepare_dataset() -> pd.DataFrame:
    players, _ = get_fixture_data(SEASON_LIST, ROOT / "cache", DATA, positions=("MID", "FWD"))
    return deadline_features(add_elo(players, carried=False)).reset_index(drop=True)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    os.environ.setdefault("SKB_DATA_DIRECTORY", str((ROOT / ".cache" / "skrub").resolve()))
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str((ROOT / ".model_cache" / "tabpfn").resolve()))
    data = prepare_dataset()
    if args.test_limit:
        data = pd.concat([
            g if season not in args.test_seasons else g.sample(min(len(g), args.test_limit), random_state=args.seeds[0])
            for season, g in data.groupby("season", sort=False)
        ], ignore_index=True)
    metadata = {
        "definition": "MID/FWD fixture rows, GW 9+, >=5 earlier fixture records, previous observed price available",
        "seasons": list(SEASON_LIST), "test_seasons": args.test_seasons,
        "rows": int(len(data)), "zero_minutes": int(data.minutes.eq(0).sum()),
        "zero_minutes_nonzero_points": int((data.minutes.eq(0) & data.total_points.ne(0)).sum()),
        "likely_starter_definition": "minutes6 >= 45 and rate60 >= 0.5, using earlier fixtures only",
        "prediction_postprocessing": "none; negative predictive means retained",
    }
    predictions, runs = [], []
    for context_size in args.context_sizes:
        for seed in args.seeds:
            for season in args.test_seasons:
                pool = data.loc[data.season.lt(season)]
                test = data.loc[data.season.eq(season)].copy()
                if pool.empty or test.empty:
                    continue
                train = sample_training_rows(pool, context_size, seed)
                for spec in args.feature_specs:
                    features = FEATURE_SPECS[spec]
                    for model_name in args.models:
                        for approach in args.approaches:
                            run_id = f"{season}:{context_size}:{seed}:{spec}:{model_name}:{approach}"
                            started = time.perf_counter()
                            try:
                                prediction, play_probability, conditional_n = run_prediction(
                                    model_name, approach, train, test, features, seed,
                                    args.foundation_estimators, args.device)
                            except Exception as exc:  # preserve other completed experiment cells
                                elapsed = time.perf_counter() - started
                                runs.append({"run_id": run_id, "season": season,
                                             "context_size": int(context_size), "seed": int(seed),
                                             "feature_spec": spec, "model": model_name,
                                             "approach": approach, "train_n": int(len(train)),
                                             "seconds": float(elapsed), "status": "error",
                                             "error_type": type(exc).__name__, "error": str(exc)})
                                print(run_id, "ERROR", type(exc).__name__, str(exc).splitlines()[0])
                                continue
                            elapsed = time.perf_counter() - started
                            out = test[KEYS + ["position", "price_m", "minutes", "total_points",
                                               "minutes6", "rate60"]].copy()
                            out["run_id"] = run_id
                            out["context_size"] = int(context_size)
                            out["seed"] = int(seed)
                            out["feature_spec"] = spec
                            out["model"] = model_name
                            out["approach"] = approach
                            out["prediction"] = prediction
                            out["play_probability"] = play_probability if play_probability is not None else np.nan
                            predictions.append(out)
                            entry = {"run_id": run_id, "season": season, "context_size": int(context_size),
                                     "seed": int(seed), "feature_spec": spec, "model": model_name,
                                     "approach": approach, "train_n": int(len(train)),
                                     "conditional_train_n": int(conditional_n), "seconds": float(elapsed),
                                     "status": "ok"}
                            for population, mask in {"all": np.ones(len(out), dtype=bool),
                                                     "likely_starter": likely_starter_mask(out)}.items():
                                subset = out.loc[mask]
                                entry[population] = regression_metrics(subset.total_points, subset.prediction)
                                entry[population]["calibration_bins"] = calibration_bins(
                                    subset.total_points.to_numpy(), subset.prediction.to_numpy())
                                entry[population]["ranking"] = ranking_metrics(subset)
                            if approach == "two_stage":
                                entry["participation_brier"] = float(brier_score_loss(
                                    test.minutes.gt(0).astype(int), play_probability))
                            runs.append(entry)
                            print(run_id, f"{elapsed:.1f}s", entry["all"]["rmse"])
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    summary: dict[str, Any] = {"dataset": metadata, "runs": runs,
                               "model_vs_ridge": [], "fixture_spec_improvements": []}
    config = ["context_size", "seed", "feature_spec", "approach"]
    run_lookup = pd.DataFrame(runs).set_index("run_id")
    if prediction_frame.empty:
        return summary, prediction_frame
    for _, group in prediction_frame.groupby(config, sort=False):
        wide = group.pivot(index=KEYS, columns="run_id", values="prediction").reset_index()
        actual = group.drop_duplicates(KEYS)[KEYS + ["total_points"]]
        wide = actual.merge(wide, on=KEYS, validate="one_to_one")
        ids = [c for c in wide.columns if c not in KEYS + ["total_points"]]
        ridge_ids = [i for i in ids if run_lookup.loc[i, "model"] == "ridge"]
        if not ridge_ids:
            continue
        ridge_id = ridge_ids[0]
        for other_id in ids:
            if other_id == ridge_id:
                continue
            gain = paired_improvement(wide.rename(columns={ridge_id: "first", other_id: "second"}),
                                      "first", "second")
            gain.update({"baseline_run": ridge_id, "comparison_run": other_id})
            summary["model_vs_ridge"].append(gain)
    fixture_config = ["context_size", "seed", "model", "approach"]
    for _, group in prediction_frame.groupby(fixture_config, sort=False):
        wide = group.pivot(index=KEYS, columns="run_id", values="prediction").reset_index()
        actual = group.drop_duplicates(KEYS)[KEYS + ["total_points"]]
        wide = actual.merge(wide, on=KEYS, validate="one_to_one")
        ids = [c for c in wide.columns if c not in KEYS + ["total_points"]]
        baseline_ids = [i for i in ids if run_lookup.loc[i, "feature_spec"] == "baseline"]
        if not baseline_ids:
            continue
        baseline_id = baseline_ids[0]
        for fixture_id in ids:
            if fixture_id == baseline_id:
                continue
            gain = paired_improvement(wide.rename(columns={baseline_id: "first", fixture_id: "second"}),
                                      "first", "second")
            gain.update({"baseline_run": baseline_id, "comparison_run": fixture_id})
            summary["fixture_spec_improvements"].append(gain)
    return summary, prediction_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["ridge", "catboost", "tabpfn", "tabicl"],
                        default=["ridge", "catboost", "tabpfn", "tabicl"])
    parser.add_argument("--approaches", nargs="+", choices=["direct", "two_stage"],
                        default=["direct", "two_stage"])
    parser.add_argument("--feature-specs", nargs="+", choices=list(FEATURE_SPECS),
                        default=["baseline", "elo", "interaction"])
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[5000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261020])
    parser.add_argument("--test-seasons", nargs="+", default=list(TEST_SEASONS))
    parser.add_argument("--foundation-estimators", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--test-limit", type=int, help="Optional rows per test season for a smoke run")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, predictions = run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    predictions.to_csv(args.out / "oos_predictions.csv", index=False)
    flat_runs = pd.json_normalize(summary["runs"])
    flat_runs.drop(columns=[c for c in flat_runs.columns if "calibration_bins" in c]).to_csv(
        args.out / "metrics.csv", index=False)
    pd.DataFrame(summary["model_vs_ridge"]).to_csv(args.out / "model_vs_ridge.csv", index=False)
    pd.DataFrame(summary["fixture_spec_improvements"]).to_csv(
        args.out / "fixture_spec_improvements.csv", index=False)
    print(f"Saved {args.out / 'summary.json'} and {len(predictions):,} prediction rows")


if __name__ == "__main__":
    main()
