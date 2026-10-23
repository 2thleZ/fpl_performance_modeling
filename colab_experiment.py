#!/usr/bin/env python3
"""Run, resume, analyze, and bundle the larger-context Colab experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

import pandas as pd

from model_comparison import DATA, run, save_results
from selection_analysis import analyze, save_outputs

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "colab_experiment_config.json"
PACKAGE_NAMES = ["numpy", "pandas", "scikit-learn", "scipy", "catboost", "tabpfn", "tabicl"]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if not config.get("phases"):
        raise ValueError("Experiment config must define at least one phase")
    return config


def phase_args(name: str, phase: dict[str, Any], config: dict[str, Any],
               data_root: Path, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        models=phase["models"],
        approaches=phase["approaches"],
        feature_specs=phase["feature_specs"],
        context_sizes=[int(value) for value in phase["context_sizes"]],
        seeds=[int(value) for value in phase["seeds"]],
        test_seasons=phase.get("test_seasons", config["test_seasons"]),
        foundation_estimators=int(phase.get(
            "foundation_estimators", config["foundation_estimators"])),
        device=phase.get("device", config["device"]),
        test_limit=phase.get("test_limit"),
        data_root=data_root,
        out=out / "phases" / name,
        resume=True,
        compress_predictions=True,
    )


def checkpoint_predictions(phase_dir: Path) -> list[Path]:
    final = phase_dir / "oos_predictions.csv.gz"
    if final.is_file():
        return [final]
    return sorted((phase_dir / "checkpoints").glob("*.csv.gz"))


def checkpoint_runs(phase_dir: Path) -> list[dict[str, Any]]:
    summary = phase_dir / "summary.json"
    if summary.is_file():
        return json.loads(summary.read_text()).get("runs", [])
    rows = []
    for path in sorted((phase_dir / "checkpoints").glob("*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_combined_predictions(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No completed prediction checkpoints were found")
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(
        ["run_id", "season", "event", "player_key", "fixture"], keep="last")


def collect_forecast_summary(out: Path, selected_phases: list[str]) -> pd.DataFrame:
    frames = []
    for name in selected_phases:
        phase_dir = out / "phases" / name
        path = phase_dir / "overall_metrics.csv"
        if path.is_file() and path.stat().st_size:
            frame = pd.read_csv(path)
        else:
            completed = [row for row in checkpoint_runs(phase_dir)
                         if row.get("status") == "ok"]
            frame = pd.json_normalize(completed) if completed else pd.DataFrame()
        if not frame.empty:
            frame["phase"] = name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    group = ["phase", "context_size", "feature_spec", "model", "approach"]
    values = [column for column in (
        "all.mae", "all.rmse", "all.calibration_slope",
        "likely_starter.mae", "likely_starter.rmse", "participation_brier"
    ) if column in metrics]
    summary = metrics.groupby(group, as_index=False)[values].mean()
    seeds = metrics.groupby(group, as_index=False).seed.nunique().rename(columns={"seed": "seeds"})
    return summary.merge(seeds, on=group, validate="one_to_one")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No completed rows were available."
    d = frame[columns].copy()
    for column in d.select_dtypes(include="number"):
        d[column] = d[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |"
            for row in d.itertuples(index=False, name=None)]
    return "\n".join([header, rule] + body)


def write_bundle_report(out: Path, forecast: pd.DataFrame,
                        outputs: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> None:
    selection = outputs["selection_summary"]
    pooled = (selection.loc[selection.scope.eq("pooled")].copy()
              if "scope" in selection else pd.DataFrame())
    if not pooled.empty:
        pooled["gain_95"] = pooled.gain_per_pick_95.map(
            lambda value: f"{value[0]:.4f} to {value[1]:.4f}")
    participation = outputs["participation_metrics"]
    if not participation.empty:
        participation = participation.loc[participation.scope.eq("pooled")]
        pgroup = ["context_size", "feature_spec", "model", "population"]
        participation = participation.groupby(pgroup, as_index=False).agg(
            seeds=("seed", "nunique"), brier=("brier", "mean"),
            roc_auc=("roc_auc", "mean"), conditional_mae=("conditional_mae", "mean"),
            conditional_rmse=("conditional_rmse", "mean"))
    failed = manifest["failed_runs"]
    failure_text = ("None." if not failed else "\n".join(
        f"- `{row['run_id']}`: {row['error_type']} — {str(row['error']).splitlines()[0]}"
        for row in failed[:20]))
    report = f"""# FPL Colab experiment report

- Completed model cells: {manifest['completed_runs']}
- Failed model cells: {len(failed)}
- Row-level predictions: {manifest['prediction_rows']:,}
- Git commit: `{manifest['git_commit']}`

## Forecast performance

Metrics below are chronological out-of-sample results averaged across the
completed random seeds. Lower MAE and RMSE are better.

{markdown_table(forecast, ['phase', 'context_size', 'model', 'approach', 'feature_spec', 'seeds', 'all.mae', 'all.rmse', 'likely_starter.mae', 'likely_starter.rmse']) if not forecast.empty else 'No completed forecast summaries were available.'}

## Matched likely-starter selections

`gain_per_pick` is the fixture-aware selection's realized points minus the
baseline selection's realized points. Intervals resample season-gameweeks after
averaging seeds.

{markdown_table(pooled, ['context_size', 'model', 'approach', 'comparison_feature_spec', 'baseline_points_per_pick', 'comparison_points_per_pick', 'gain_per_pick', 'gain_95', 'changed_pick_rate']) if not pooled.empty else 'No completed matched selection comparisons were available.'}

## Participation and conditional points

{markdown_table(participation, ['context_size', 'model', 'feature_spec', 'population', 'seeds', 'brier', 'roc_auc', 'conditional_mae', 'conditional_rmse']) if not participation.empty else 'No completed two-stage results were available.'}

## Failed cells

{failure_text}

Season-specific selection estimates are in
`selection/selection_summary.csv` and `selection/FINDINGS.md`. These rankings
use the restricted archived candidate pool and are not a transfer simulation.
"""
    (out / "REPORT.md").write_text(report)


def create_bundle(out: Path, config: dict[str, Any], data_root: Path,
                  selected_phases: list[str]) -> Path:
    prediction_paths: list[Path] = []
    runs: list[dict[str, Any]] = []
    for name in selected_phases:
        phase_dir = out / "phases" / name
        prediction_paths.extend(checkpoint_predictions(phase_dir))
        runs.extend(checkpoint_runs(phase_dir))
    predictions = load_combined_predictions(prediction_paths)
    combined_path = out / "combined_predictions.csv.gz"
    predictions.to_csv(combined_path, index=False, compression="gzip")

    selection_dir = out / "selection"
    outputs = analyze(predictions, top_k=int(config["top_k"]),
                      bootstrap_reps=int(config["bootstrap_reps"]))
    save_outputs(outputs, selection_dir, top_k=int(config["top_k"]))
    forecast = collect_forecast_summary(out, selected_phases)
    forecast.to_csv(out / "forecast_summary.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "selected_phases": selected_phases,
        "data_root": str(data_root.resolve()),
        "tabpfn_token_available": bool(os.environ.get("TABPFN_TOKEN")),
        "completed_runs": sum(row.get("status") == "ok" for row in runs),
        "failed_runs": [
            {key: row.get(key) for key in
             ("run_id", "error_type", "error", "seconds")}
            for row in runs if row.get("status") != "ok"
        ],
        "prediction_rows": int(len(predictions)),
        "config": config,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (out / "config_effective.json").write_text(json.dumps(config, indent=2))
    write_bundle_report(out, forecast, outputs, manifest)

    staging = out / "bundle"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for source in (out / "manifest.json", out / "config_effective.json",
                   out / "REPORT.md", out / "forecast_summary.csv", combined_path):
        shutil.copy2(source, staging / source.name)
    shutil.copytree(selection_dir, staging / "selection")
    for name in selected_phases:
        phase_dir = out / "phases" / name
        compact = staging / "phases" / name
        compact.mkdir(parents=True, exist_ok=True)
        for filename in ("summary.json", "metrics.csv", "overall_metrics.csv",
                         "model_vs_ridge.csv", "fixture_spec_improvements.csv"):
            source = phase_dir / filename
            if source.is_file():
                shutil.copy2(source, compact / filename)
    archive = Path(shutil.make_archive(str(out / "fpl_colab_results"), "zip", staging))
    print(f"Created {archive} ({archive.stat().st_size / 1_000_000:.1f} MB)")
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "colab_results")
    parser.add_argument("--phases", nargs="+",
                        help="Run only these named phases; defaults to all enabled phases")
    parser.add_argument("--bundle-only", action="store_true",
                        help="Build the bundle from completed checkpoints without fitting models")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a tiny Ridge-only end-to-end validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = dict(config)
        config.update({"test_seasons": ["2025-26"], "foundation_estimators": 1,
                       "bootstrap_reps": 100})
        config["phases"] = {
            "smoke": {
                "enabled": True, "models": ["ridge"],
                "approaches": ["direct", "two_stage"],
                "feature_specs": ["baseline", "elo", "interaction"],
                "context_sizes": [100], "seeds": [20261020], "test_limit": 400,
            }
        }
    enabled = [name for name, phase in config["phases"].items() if phase.get("enabled", True)]
    selected = args.phases or enabled
    unknown = sorted(set(selected) - set(config["phases"]))
    if unknown:
        raise ValueError(f"Unknown experiment phases: {unknown}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.bundle_only:
        for name in selected:
            print(f"Starting phase: {name}")
            phase = phase_args(name, config["phases"][name], config, args.data_root, args.out)
            summary, predictions = run(phase)
            save_results(phase, summary, predictions)
    create_bundle(args.out, config, args.data_root, selected)


if __name__ == "__main__":
    main()
