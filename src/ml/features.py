"""Candidate feature extraction; features use real timestamps and local geometry."""
from __future__ import annotations
import numpy as np

FEATURE_NAMES = [
    "distance_m", "distance_sq_m", "segment_length_m", "projection_fraction",
    "gps_speed_mps", "gps_heading_sin", "gps_heading_cos",
    "road_heading_sin", "road_heading_cos", "heading_difference_sin",
    "heading_difference_cos", "heading_available", "time_delta_s",
    "previous_gap_break", "next_gap_break",
]

def _bearing(v):
    # Bearing clockwise from north in degrees.
    return (np.degrees(np.arctan2(v[0], v[1])) + 360.0) % 360.0

def _angle_delta(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def extract_features(trip, candidates, gap_break_s: float = 30.0):
    """Return feature matrix per candidate and aligned observation/arc metadata.

    Candidate road orientation is treated as undirected: heading difference is the
    minimum of the forward and reverse segment bearings.
    """
    P, t, N, arcs = trip.P, np.asarray(trip.t), trip.N, trip.arcs
    T = len(P)
    dt_prev = np.zeros(T, dtype=float); dt_next = np.zeros(T, dtype=float)
    movement_prev = np.zeros((T, 2)); movement_next = np.zeros((T, 2))
    has_prev = np.zeros(T, bool); has_next = np.zeros(T, bool)
    if T > 1:
        dt_prev[1:] = np.diff(t)
        dt_next[:-1] = np.diff(t)
        has_prev[1:] = (dt_prev[1:] > 0) & (dt_prev[1:] <= gap_break_s)
        has_next[:-1] = (dt_next[:-1] > 0) & (dt_next[:-1] <= gap_break_s)
        movement_prev[1:] = np.diff(P, axis=0)
        movement_next[:-1] = np.diff(P, axis=0)
    X, obs_ids, arc_ids, distances = [], [], [], []
    for obs, c in enumerate(candidates):
        if has_prev[obs]:
            vec = movement_prev[obs]; gps_heading = _bearing(vec)
            gps_speed = float(np.linalg.norm(vec) / dt_prev[obs])
            heading_ok = True
        elif has_next[obs]:
            vec = movement_next[obs]; gps_heading = _bearing(vec)
            gps_speed = float(np.linalg.norm(vec) / dt_next[obs])
            heading_ok = True
        else:
            gps_heading, gps_speed, heading_ok = 0.0, 0.0, False
        ids = c["arc_indices"]
        if not len(ids):
            continue
        road_vec = N[arcs[ids, 1]] - N[arcs[ids, 0]]
        road_heading = np.array([_bearing(v) if np.linalg.norm(v) > 1e-9 else 0.0 for v in road_vec])
        # Roads can be traversed either direction.
        raw = np.abs(_angle_delta(gps_heading, road_heading))
        undirected_diff = np.minimum(raw, np.abs(180.0 - raw))
        dist = c["distance_m"]
        seglen = np.linalg.norm(road_vec, axis=1)
        dprev = float(dt_prev[obs]) if obs > 0 else 0.0
        dnext = float(dt_next[obs]) if obs < T - 1 else 0.0
        rows = np.column_stack([
            dist, dist**2, seglen, c["fraction"],
            np.full(len(ids), gps_speed),
            np.full(len(ids), np.sin(np.radians(gps_heading))),
            np.full(len(ids), np.cos(np.radians(gps_heading))),
            np.sin(np.radians(road_heading)), np.cos(np.radians(road_heading)),
            np.sin(np.radians(undirected_diff)), np.cos(np.radians(undirected_diff)),
            np.full(len(ids), float(heading_ok)), np.full(len(ids), dprev),
            np.full(len(ids), float(obs > 0 and dprev > gap_break_s)),
            np.full(len(ids), float(obs < T - 1 and dnext > gap_break_s)),
        ])
        X.append(rows); obs_ids.extend([obs] * len(ids)); arc_ids.extend(ids.tolist()); distances.extend(dist.tolist())
    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.array([], int), np.array([], int), np.array([], float)
    return np.vstack(X), np.asarray(obs_ids, int), np.asarray(arc_ids, int), np.asarray(distances, float)
