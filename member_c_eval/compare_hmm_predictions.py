"""Compare classifier-only and externally generated HMM/Viterbi predictions.

The HMM code itself belongs to the route-reconstruction implementation. It
should export a CSV with: trip_id, observation_id, predicted_arc_id. This script
compares that output with the classifier-only predictions and the per-point
reference arc labels exported by ml dataset.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from . import config as cfg
from .pathsnap_eval.route_metrics import route_continuity

HMM_PREDICTIONS_CSV = cfg.PROJECT_ROOT / "outputs" / "hmm" / "student_test_hmm_predictions.csv"


def _point_metrics(pred: pd.DataFrame, truth: pd.DataFrame, method: str) -> dict:
    merged = truth.merge(
        pred[[cfg.TRIP_COLUMN, cfg.OBSERVATION_COLUMN, "predicted_arc_id"]],
        on=[cfg.TRIP_COLUMN, cfg.OBSERVATION_COLUMN],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(f"No matching GPS observations found for {method} predictions.")
    if cfg.TRUE_ARC_COLUMN not in merged.columns:
        raise ValueError(
            "true_arc_id is missing from the processed student_test data. "
            "Your ml dataset.py export must preserve the reference arc for each GPS observation "
            "to evaluate HMM-decoded output honestly."
        )
    merged = merged.dropna(subset=[cfg.TRUE_ARC_COLUMN, "predicted_arc_id"]).copy()
    y_true = merged[cfg.TRUE_ARC_COLUMN].astype(int)
    y_pred = merged["predicted_arc_id"].astype(int)
    return {
        "method": method,
        "observation_count": int(len(merged)),
        "pointwise_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sequence_exact_match_rate": float(
            merged.assign(correct=(y_true.to_numpy() == y_pred.to_numpy()))
            .groupby(cfg.TRIP_COLUMN)["correct"].all().mean()
        ),
    }


def main() -> int:
    if not HMM_PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"HMM prediction file not found: {HMM_PREDICTIONS_CSV}\n"
            "Export your HMM/Viterbi output with trip_id, observation_id, predicted_arc_id."
        )

    selected_path = cfg.OUTPUT_DIR / "model_selection_summary.json"
    if not selected_path.exists():
        raise FileNotFoundError("Run run_model_comparison.py before comparing HMM output.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected_model"]
    safe_selected = "".join(c if c.isalnum() else "_" for c in selected).strip("_").lower()
    tuned_classifier_csv = (
        cfg.OUTPUT_DIR / "bayesian_optimisation" / "tuned_student_test_predictions.csv"
    )
    classifier_csv = (
        tuned_classifier_csv
        if tuned_classifier_csv.exists()
        else cfg.OUTPUT_DIR / "predictions" / f"student_test_predictions_{safe_selected}.csv"
    )
    if not classifier_csv.exists():
        raise FileNotFoundError(f"Classifier prediction file not found: {classifier_csv}")

    student_csv = Path(cfg.STUDENT_TEST_CSV)
    student = pd.read_csv(student_csv, dtype={cfg.TRIP_COLUMN: str})
    if cfg.TRUE_ARC_COLUMN not in student.columns:
        raise ValueError(
            "The processed student_test table needs true_arc_id for fair pointwise HMM evaluation."
        )
    truth = student[[cfg.TRIP_COLUMN, cfg.OBSERVATION_COLUMN, cfg.TRUE_ARC_COLUMN]].drop_duplicates(
        [cfg.TRIP_COLUMN, cfg.OBSERVATION_COLUMN]
    )

    # Preserve zero-padded trip folder IDs while reading the prediction files.
    classifier = pd.read_csv(classifier_csv, dtype={cfg.TRIP_COLUMN: str})
    hmm = pd.read_csv(HMM_PREDICTIONS_CSV, dtype={cfg.TRIP_COLUMN: str})
    required = {cfg.TRIP_COLUMN, cfg.OBSERVATION_COLUMN, "predicted_arc_id"}
    for name, frame in (("classifier", classifier), ("HMM", hmm)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} prediction CSV is missing: {sorted(missing)}")

    records = []
    for name, frame in (("Classifier only", classifier), ("Classifier + HMM/Viterbi", hmm)):
        record = _point_metrics(frame, truth, name)
        record.update(route_continuity(frame, cfg.DATA_ROOT, cfg.STUDENT_TEST_PARTITION, cfg))
        records.append(record)

    output_dir = cfg.OUTPUT_DIR / "hmm_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(output_dir / "classifier_vs_hmm.csv", index=False)
    print(result.to_string(index=False))
    print(f"Saved: {output_dir / 'classifier_vs_hmm.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
