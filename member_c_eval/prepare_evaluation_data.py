"""Build the evaluation CSVs from the team's existing PathSnap pipeline.

Run from the repository root:
    python -m member_c_eval.prepare_evaluation_data
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from src.common import load_trip, project
from src.data.labels import align_labels
from src.ml.dataset import build_trip_dataset

ROOT = Path.cwd()
TRAIN_CSV = ROOT / "results" / "ml" / "candidate_training_data.csv"
STUDENT_ROOT = ROOT / "data" / "student_test"
OUT_DIR = ROOT / "results" / "ml"
STUDENT_CSV = OUT_DIR / "student_test_candidate_data.csv"

RADIUS_M = 50.0
MAX_CANDIDATES = 30
MAX_LABEL_DISTANCE_M = 50.0
GAP_BREAK_S = 30.0


def _attach_reference_metadata(frame: pd.DataFrame, trip_dir: Path) -> pd.DataFrame:
    """Attach ground-truth arc and timestamp aligned to the cleaned GPS rows."""
    trip = project(load_trip(trip_dir, with_route=True))
    ref_arc_ids, _label_dist = align_labels(trip, return_dist=True)
    result = frame.copy()
    obs = result["observation_index"].to_numpy(dtype=int)
    if len(obs) and (obs.min() < 0 or obs.max() >= len(trip.t)):
        raise ValueError(
            f"Observation index is outside cleaned track for trip {trip.trip_id}."
        )
    result["trip_id"] = str(trip.trip_id)
    result["true_arc_id"] = ref_arc_ids[obs].astype(int)
    result["timestamp"] = trip.t[obs].astype(float)
    # Number of usable GPS observations after load_trip's invalid-row filtering.
    result["total_observations"] = int(len(trip.t))
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"{TRAIN_CSV} is missing. First run: python -m src.ml.train"
        )
    if not STUDENT_ROOT.exists():
        raise FileNotFoundError(
            f"{STUDENT_ROOT} is missing. Extract Student folder.zip so that "
            "its student_test folder is data/student_test."
        )

    print("Loading the processed training candidate dataset...", flush=True)
    train_df = pd.read_csv(TRAIN_CSV, dtype={"trip_id": str})
    if train_df.empty:
        raise RuntimeError("The training candidate CSV is empty.")

    train_frames = []
    for trip_id, part in train_df.groupby("trip_id", sort=True):
        trip_dir = ROOT / "data" / "train" / str(trip_id)
        if not trip_dir.exists():
            raise FileNotFoundError(f"Training trip folder not found: {trip_dir}")
        train_frames.append(_attach_reference_metadata(part, trip_dir))
    train_df = pd.concat(train_frames, ignore_index=True)
    train_df.to_csv(TRAIN_CSV, index=False)

    print("Building candidate rows for the 10 student_test trips...", flush=True)
    student_frames = []
    summaries = []
    trip_dirs = sorted(path for path in STUDENT_ROOT.iterdir() if path.is_dir())
    if len(trip_dirs) != 10:
        print(
            f"WARNING: expected 10 student_test trip folders, found {len(trip_dirs)}.",
            flush=True,
        )

    empty_trip_ids = []
    for trip_dir in trip_dirs:
        frame, summary = build_trip_dataset(
            trip_dir,
            radius_m=RADIUS_M,
            max_candidates=MAX_CANDIDATES,
            max_label_distance_m=MAX_LABEL_DISTANCE_M,
            gap_break_s=GAP_BREAK_S,
        )
        summary = dict(summary)
        summaries.append(summary)
        if not frame.empty:
            student_frames.append(_attach_reference_metadata(frame, trip_dir))
        else:
            empty_trip_ids.append(trip_dir.name)
        print(
            f"[{trip_dir.name}] candidate_rows={len(frame):,}; "
            f"skipped_far={summary.get('skipped_far_labels', 0)}; "
            f"missing_positive={summary.get('missing_positive', 0)}",
            flush=True,
        )

    if empty_trip_ids:
        raise RuntimeError(
            "No candidate rows were generated for these student_test trips: "
            f"{empty_trip_ids}. Inspect their dataset-build summaries before evaluation."
        )
    if len(student_frames) != len(trip_dirs):
        raise RuntimeError("Not all student_test trips contributed candidate rows.")

    student_df = pd.concat(student_frames, ignore_index=True)
    student_df.to_csv(STUDENT_CSV, index=False)
    pd.DataFrame(summaries).to_csv(
        OUT_DIR / "student_test_dataset_build_summary.csv", index=False
    )

    print("\nEvaluation data prepared.")
    print(
        f"Training: {len(train_df):,} candidate rows, "
        f"{train_df['trip_id'].nunique()} trips"
    )
    print(
        f"Student test: {len(student_df):,} candidate rows, "
        f"{student_df['trip_id'].nunique()} trips"
    )
    print(f"Training CSV: {TRAIN_CSV}")
    print(f"Student-test CSV: {STUDENT_CSV}")
    print(f"Student-test build summary: {OUT_DIR / 'student_test_dataset_build_summary.csv'}")
    print("Evaluator-test data is not loaded by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
