"""Per-observation reference labels from the ordered .route file."""
from __future__ import annotations
import numpy as np
from src.common import point_segment


def align_labels(trip, return_dist: bool = False):
    """Assign each GPS point a reference arc id using a MONOTONIC alignment over the route:
    the route index may stay or move forward, never backwards; cost = point-to-arc distance.
    Needs trip.route and project(trip). Returns arc ids (len T) [, distances in metres]."""
    P, route = trip.P, trip.route
    RA, RB = trip.N[trip.arcs[route, 0]], trip.N[trip.arcs[route, 1]]
    T, n = len(P), len(route)
    idx = np.arange(n)
    D = point_segment(P[0], RA, RB)[0]
    bp = np.zeros((T, n), dtype=np.int32)
    for t in range(1, T):
        pm = np.minimum.accumulate(D)                              # best cost over j <= k
        bp[t] = np.maximum.accumulate(np.where(D <= pm, idx, 0))   # argmin of that prefix
        D = point_segment(P[t], RA, RB)[0] + pm
    k = int(D.argmin())
    out = np.empty(T, dtype=int)
    for t in range(T - 1, -1, -1):
        out[t] = k
        if t > 0:
            k = bp[t, k]
    arcs = route[out]
    if not return_dist:
        return arcs
    d = point_segment(P, trip.N[trip.arcs[arcs, 0]], trip.N[trip.arcs[arcs, 1]])[0]
    return arcs, d
