#!/usr/bin/env python3
"""Reproduce repeated TabICL calls in a fresh process, stopping at first failure."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from model_comparison import (DATA, FEATURE_SPECS, prepare_dataset, redact_secret,
                              release_model_memory, run_prediction, sample_training_rows)
from runtime_diagnostics import runtime_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=Path("colab_results/tabicl_diagnostic.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-size", type=int, default=5000)
    parser.add_argument("--test-rows", type=int, default=64)
    parser.add_argument("--include-tabpfn", action="store_true",
                        help="Alternate TabPFN and TabICL as in the original experiment")
    args = parser.parse_args()
    report = {"runtime": runtime_metadata(), "runs": [], "context_size": args.context_size,
              "test_rows": args.test_rows, "requested_device": args.device}
    if args.include_tabpfn and not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError("Load TABPFN_TOKEN from Colab Secrets before the mixed-model diagnostic")
    if args.include_tabpfn:
        os.environ["TABPFN_NO_BROWSER"] = "1"
    print(json.dumps(report["runtime"], indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False))

    # Identify the actual cached weights without downloading or modifying them.
    try:
        from huggingface_hub import hf_hub_download
        from tabicl import TabICLRegressor
        filename = TabICLRegressor().checkpoint_version
        path = Path(hf_hub_download("jingang/TabICL", filename, local_files_only=True))
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        report["tabicl_cached_weights"] = {"filename": filename, "bytes": path.stat().st_size, "sha256": digest}
    except Exception as exc:
        report["cache_inspection_error"] = redact_secret(str(exc))
    save()
    data = prepare_dataset(args.data_root)
    for season in ["2022-23", "2023-24"]:
        train = sample_training_rows(data.loc[data.season < season], args.context_size, 20261020)
        test = data.loc[data.season.eq(season)].head(args.test_rows)
        for approach in ["direct", "two_stage"]:
            for spec, features in FEATURE_SPECS.items():
                for model in (["tabpfn", "tabicl"] if args.include_tabpfn else ["tabicl"]):
                    row = dict(season=season, approach=approach, feature_spec=spec, model=model, diagnostics={})
                    start = time.perf_counter()
                    try:
                        prediction, _, _, _ = run_prediction(
                            model, approach, train, test, features, 20261020, 4,
                            args.device, row["diagnostics"])
                        if not np.isfinite(prediction).all():
                            raise ValueError("Nonfinite predictions")
                        row["status"] = "ok"
                    except Exception as exc:
                        row.update(status="error", error=redact_secret(str(exc)),
                                   traceback=redact_secret(traceback.format_exc()))
                    row["seconds"] = time.perf_counter()-start
                    report["runs"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
                    if row["status"] == "error":
                        raise SystemExit(f"Stopped at first failure; diagnostics saved to {args.out}")
                    release_model_memory()
    print(f"All diagnostic calls passed; saved to {args.out}")


if __name__ == "__main__":
    main()
