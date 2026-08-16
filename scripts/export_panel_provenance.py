#!/usr/bin/env python3
"""Export the recovered sentinel-panel provenance package.

The historical sampler survives in benchmark_eval git object 9ca37b3...,
while the ignored construction matrix does not.  This export freezes every
recoverable construction fact and the earliest complete post-selection
prediction matrix used to verify panel membership.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT.parent / "benchmark_eval"
OUT = ROOT / "assets" / "metrics_export"
RUN_ID = "20260308-204344Z"

CONSTRUCTION_MODELS = {
    "gemini-2.0-flash": 299,
    "gemini-2.5-flash": 299,
    "gemini-2.5-flash-preview-09-2025": 288,
    "gemini-2.5-pro": 299,
    "gemini-3-flash-preview": 299,
    "gemini-3-pro-preview": 1199,
    "gpt-4.1": 300,
    "gpt-4.1-mini": 300,
    "gpt-4.1-nano": 300,
    "gpt-4o": 300,
    "gpt-4o-mini": 1197,
    "gpt-5": 300,
    "gpt-5-mini": 300,
    "gpt-5-nano": 300,
    "gpt-5.1": 300,
    "gpt-5.2": 300,
}


def sha256_lines(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_lines_without_final_newline(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def disagreement_score(predictions: list[bool]) -> float:
    """Historical score: population SD times binary Shannon entropy."""
    values = np.asarray(predictions, dtype=float)
    p_true = float(values.mean())
    entropy = (
        0.0
        if p_true in (0.0, 1.0)
        else -(p_true * np.log(p_true) + (1.0 - p_true) * np.log(1.0 - p_true))
    )
    return float(values.std(ddof=0) * (entropy + 1e-10))


def parse_bool(comment: str, field: str) -> bool:
    match = re.search(rf"{field}=(True|False)", comment)
    if not match:
        raise ValueError(f"cannot parse {field} from score comment")
    return match.group(1) == "True"


def main() -> None:
    panel_source = (
        BENCHMARK / "notebooks" / "canonical_evidence_export" / "panel_ids.txt"
    )
    panel_ids = sorted(
        value.strip()
        for value in panel_source.read_text(encoding="utf-8").splitlines()
        if value.strip()
    )
    if len(panel_ids) != 300 or len(set(panel_ids)) != 300:
        raise AssertionError("expected exactly 300 unique panel identifiers")

    traces = pd.read_csv(
        BENCHMARK / "streamlit_app" / "langfuse_traces.csv",
        low_memory=False,
        dtype={"metadata.custom_id": str},
    )
    scores = pd.read_csv(
        BENCHMARK / "streamlit_app" / "langfuse_scores.csv", low_memory=False
    )
    run = traces.loc[
        traces["metadata.run_id"].eq(RUN_ID),
        ["id", "timestamp", "metadata.model", "metadata.custom_id"],
    ].rename(
        columns={
            "id": "trace_id",
            "metadata.model": "model",
            "metadata.custom_id": "custom_id",
        }
    )
    accuracy = (
        scores.loc[scores["name"].eq("accuracy"), ["traceId", "value", "comment"]]
        .sort_values("traceId")
        .drop_duplicates("traceId", keep="last")
    )
    predictions = run.merge(
        accuracy, left_on="trace_id", right_on="traceId", how="inner", validate="1:1"
    )
    predictions["expected"] = predictions["comment"].map(
        lambda value: parse_bool(str(value), "expected")
    )
    predictions["predicted"] = predictions["comment"].map(
        lambda value: parse_bool(str(value), "predicted")
    )
    predictions["correct"] = predictions["expected"].eq(predictions["predicted"])
    predictions = predictions[
        [
            "custom_id",
            "model",
            "expected",
            "predicted",
            "correct",
            "trace_id",
            "timestamp",
        ]
    ].sort_values(["custom_id", "model"])

    if set(predictions["custom_id"]) != set(panel_ids):
        raise AssertionError("earliest panel run does not reproduce frozen membership")
    if predictions["model"].nunique() != 14 or len(predictions) != 4194:
        raise AssertionError("unexpected earliest-run prediction coverage")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "panel_ids.txt").write_text(
        "".join(f"{value}\n" for value in panel_ids), encoding="utf-8"
    )
    predictions.to_csv(OUT / "panel_postselection_predictions.csv", index=False)

    provenance = {
        "status": (
            "Recovered provenance package. The ignored per-item construction "
            "matrix is not present, so the historical random draw cannot be replayed."
        ),
        "panel": {
            "items": 300,
            "sha256_sorted_ids_with_final_newline": sha256_lines(panel_ids),
            "sha256_sorted_ids_without_final_newline": (
                sha256_lines_without_final_newline(panel_ids)
            ),
            "identifier_file": "panel_ids.txt",
        },
        "historical_sampler": {
            "repository": "benchmark_eval",
            "source_at_commit": "52abcc2^:scripts/sample.py",
            "source_blob_sha1": "9ca37b38fb94e9e09ab728a26d8bc65068330ef6",
            "score": "population_std(boolean_predictions) * binary_shannon_entropy",
            "percentile_buckets": {
                "high": "score >= p80",
                "medium": "p50 <= score < p80",
                "low": "score < p50",
            },
            "target_counts": {"high": 201, "medium": 69, "low": 30},
            "random_generator": "numpy.random.default_rng",
            "seed": 42,
            "replacement": False,
            "final_shuffle": True,
            "top_up_order": ["remaining_scored_rows", "uncovered_rows"],
        },
        "recovered_execution_diagnostics": {
            "source_items": 1200,
            "covered_items": 1199,
            "positive_labels": 682,
            "negative_labels": 518,
            "p50": 0.0,
            "p80": 0.30005565652885874,
            "high_pool": 349,
            "medium_pool": 850,
            "low_pool": 0,
            "uncovered_items": 1,
        },
        "construction_models_and_prediction_counts": CONSTRUCTION_MODELS,
        "earliest_surviving_postselection_matrix": {
            "run_id": RUN_ID,
            "rows": int(len(predictions)),
            "models": int(predictions["model"].nunique()),
            "items": int(predictions["custom_id"].nunique()),
            "complete_14_model_items": int(
                predictions.groupby("custom_id")["model"].nunique().eq(14).sum()
            ),
            "file": "panel_postselection_predictions.csv",
            "note": (
                "These predictions verify the frozen panel but were recorded after "
                "selection and are not substituted for the missing construction matrix."
            ),
        },
        "non_recoverable_inputs": [
            "data/baseline_predictions.json",
            "the original ordered 1200-row source export",
        ],
    }
    (OUT / "panel_selection_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance["panel"], indent=2))
    print(f"wrote {len(predictions)} prediction rows")


if __name__ == "__main__":
    main()
