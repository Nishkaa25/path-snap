"""Prediction pipeline. HARD RULE: nothing in this file reads the reference route."""
import json
import os
import time

import numpy as np

from src.hmm.graph import Graph
from src.hmm.viterbi import HMMConfig, decode


class StubModel:
    """Placeholder until C's artifacts/model.joblib exists (set PATHSNAP_STUB=1).
    Scores candidates by the first feature column as a negative distance."""

    def predict_proba(self, X):
        d = np.asarray(X)[:, 0]
        p = 1.0 / (1.0 + np.exp((d - 15.0) / 8.0))
        return np.column_stack([1 - p, p])


class Bundle:
    def __init__(self, art_dir="artifacts"):
        if os.getenv("PATHSNAP_STUB") == "1":
            self.model = StubModel()
            self.R, self.K, self.hmm = 50.0, 5, HMMConfig()
            return
        import joblib
        self.model = joblib.load(f"{art_dir}/model.joblib")
        cfg = json.load(open(f"{art_dir}/config.json"))
        self.R, self.K = cfg["R"], cfg["K"]
        self.hmm = HMMConfig(**cfg["hmm"])


def predict_trip(trip_dir, b: Bundle):
    # teammates' modules are imported here so the API (and /health) starts before they are merged
    from src.common import load_trip, project
    from src.features.candidates import generate_candidates
    from src.features.features import add_features, FEATURE_COLS

    tm = {}
    t0 = time.time()
    trip = project(load_trip(trip_dir, with_route=False))             # NO reference labels here
    tm["load_s"] = time.time() - t0
    t0 = time.time()
    cand = add_features(trip, generate_candidates(trip, b.R, b.K))    # no label argument
    tm["features_s"] = time.time() - t0
    t0 = time.time()
    p = b.model.predict_proba(cand[FEATURE_COLS].values)[:, 1]
    tm["model_s"] = time.time() - t0
    t0 = time.time()
    T = len(trip.P)
    arcs_o, p_o = [None] * T, [None] * T
    for o, grp in cand.assign(p=p).groupby("obs"):
        arcs_o[o], p_o[o] = grp["arc_id"].values, grp["p"].values
    clf = np.full(T, -1)                                              # classifier-only argmax
    for o in range(T):
        if arcs_o[o] is not None:
            clf[o] = arcs_o[o][int(np.argmax(p_o[o]))]
    g = Graph(trip.arcs, len(trip.nodes))
    hmm = decode(arcs_o, p_o, trip.t, g, b.hmm)
    tm["hmm_s"] = time.time() - t0
    return trip, clf, hmm, tm
