"""Build candidate-level ML training data from PathSnap trip folders."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.common import load_trip, project
from src.data.labels import align_labels
from src.ml.candidates import generate_candidates
from src.ml.features import extract_features, FEATURE_NAMES


def trip_number(trip_id: str):
    digits = ''.join(ch for ch in str(trip_id) if ch.isdigit())
    return int(digits) if digits else None


def build_trip_dataset(
    trip_dir,
    radius_m=50.0,
    max_candidates=30,
    max_label_distance_m=50.0,
    gap_break_s=30.0
):
    trip = project(load_trip(trip_dir, with_route=True))

    if trip.route is None:
        raise ValueError(
            f"Missing reference route for training trip {trip.trip_id}"
        )

    ref_arc_ids, label_dist = align_labels(trip, return_dist=True)

    cand = generate_candidates(
        trip,
        radius_m=radius_m,
        max_candidates=max_candidates
    )

    X, obs, arc_indices, distances = extract_features(
        trip, cand, gap_break_s=gap_break_s
    )

    if not len(X):
        return pd.DataFrame(), {
            "trip_id": trip.trip_id,
            "observations": len(trip.P),
            "candidate_rows": 0,
            "eligible_observations": int(
                np.sum(label_dist <= max_label_distance_m)
            ),
            "skipped_far_labels": int(
                np.sum(label_dist > max_label_distance_m)
            ),
            "missing_positive": 0
        }

    # Candidate arc indices correspond to rows in trip.arcs.
    candidate_actual_arc_ids = arc_indices

    # Generate binary candidate labels.
    y = (
        candidate_actual_arc_ids == ref_arc_ids[obs]
    ).astype(np.int8)

    # Identify eligible observations whose correct arc was not generated.
    eligible_obs = set(
        np.where(label_dist <= max_label_distance_m)[0]
    )

    candidate_positive_obs = set(obs[y == 1])

    missing_positive_obs = (
        eligible_obs - candidate_positive_obs
    )

    # Retain only candidates belonging to valid observations:
    # 1. Reference label is within the allowed distance.
    # 2. The correct arc exists in the candidate set.
    valid_obs = (
        (label_dist[obs] <= max_label_distance_m)
        & (~np.isin(obs, list(missing_positive_obs)))
    )

    # Construct the candidate dataframe.
    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    df["trip_id"] = trip.trip_id
    df["observation_index"] = obs
    df["arc_index"] = arc_indices
    df["candidate_distance_m"] = distances
    df["label"] = y

    # Remove invalid observations.
    df = df.loc[valid_obs].reset_index(drop=True)

    summary = {
        "trip_id": trip.trip_id,
        "observations": len(trip.P),
        "candidate_rows": int(len(df)),
        "eligible_observations": int(len(eligible_obs)),
        "skipped_far_labels": int(
            np.sum(label_dist > max_label_distance_m)
        ),
        "missing_positive": int(len(missing_positive_obs))
    }

    return df, summary


def build_training_dataset(
    train_root="data/train",
    exclude_trip_numbers=(33, 54),
    radius_m=50.0,
    max_candidates=30,
    max_label_distance_m=50.0,
    gap_break_s=30.0
):
    root = Path(train_root)

    frames = []
    summaries = []

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue

        if trip_number(d.name) in set(exclude_trip_numbers):
            summaries.append({
                "trip_id": d.name,
                "excluded": "quality exclusion",
                "candidate_rows": 0
            })
            continue

        frame, summary = build_trip_dataset(
            d,
            radius_m,
            max_candidates,
            max_label_distance_m,
            gap_break_s
        )

        summaries.append(summary)

        if len(frame):
            frames.append(frame)

        print(
            f"[{d.name}] "
            f"rows={len(frame):,} "
            f"skipped_far={summary.get('skipped_far_labels', 0)} "
            f"missing_positive={summary.get('missing_positive', 0)}",
            flush=True
        )

    if not frames:
        raise RuntimeError(
            "No candidate rows were produced. "
            "Check data path and candidate radius."
        )

    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(summaries)
    )
