"""Inference helpers for Member 3: candidate arc probabilities per GPS point."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from src.common import load_trip, project
from src.ml.candidates import generate_candidates
from src.ml.features import extract_features, FEATURE_NAMES


def load_saved_model(path="results/ml/best_model.joblib"):
    bundle = joblib.load(path)
    return bundle



def score_trip(trip_dir, model_bundle, radius_m=None, max_candidates=None, gap_break_s=30.0):
    """Scores candidates without loading .route (leakage-safe inference path).


    Returns a DataFrame with candidate arc IDs, raw classifier scores, and scores
    normalized within each observation for use as HMM emission-like weights.
    """
    model = model_bundle["model"] if isinstance(model_bundle, dict) else model_bundle
    radius_m = radius_m or (model_bundle.get("radius_m", 50.0) if isinstance(model_bundle, dict) else 50.0)
    max_candidates = max_candidates or (model_bundle.get("candidate_cap", 30) if isinstance(model_bundle, dict) else 30)
    trip = project(load_trip(trip_dir, with_route=False))
    cands = generate_candidates(trip, radius_m, max_candidates)
    X, obs, arc_rows, dist = extract_features(trip, cands, gap_break_s)
    if not len(X):
        return pd.DataFrame(columns=["trip_id", "observation_index", "arc_index", "distance_m", "probability", "normalized_probability"])
    raw = model.predict_proba(X)[:, 1]
    df = pd.DataFrame({"trip_id": trip.trip_id, "observation_index": obs, "arc_index": arc_rows,
                     "distance_m": dist, "probability": raw})
    sums = df.groupby(["trip_id", "observation_index"]).probability.transform("sum")
    df["normalized_probability"] = np.divide(df.probability, sums, out=np.zeros_like(raw), where=sums > 0)
    return df


def scores_as_nested_dict(scores: pd.DataFrame):
    """JSON-friendly observation -> candidate list shape for HMM integration."""
    output = {}
    for (trip_id, obs), group in scores.groupby(["trip_id", "observation_index"]):
        output.setdefault(str(trip_id), {})[int(obs)] = [
            {"arc_index": int(r.arc_index), "distance_m": float(r.distance_m),
             "classifier_probability": float(r.probability), "normalized_probability": float(r.normalized_probability)}
            for r in group.sort_values("normalized_probability", ascending=False).itertuples()
        ]
    return output
