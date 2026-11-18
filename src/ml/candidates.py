"""Nearby road-arc candidate generation using a midpoint spatial index."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from src.common import point_segment



class ArcCandidateIndex:
    """Index road arcs by midpoint; exact point-to-segment filtering removes false hits.

    The search radius is expanded by the longest arc half-length so a segment whose
    midpoint is far away but whose geometry reaches the GPS point is not missed.
    """
    def __init__(self, nodes_xy: np.ndarray, arcs: np.ndarray):
        self.nodes = np.asarray(nodes_xy, dtype=float)
        self.arcs = np.asarray(arcs, dtype=int)
        self.a = self.nodes[self.arcs[:, 0]]
        self.b = self.nodes[self.arcs[:, 1]]
        self.mid = (self.a + self.b) / 2.0
        self.length = np.linalg.norm(self.b - self.a, axis=1)
        self.half = self.length / 2.0
        self.tree = cKDTree(self.mid) if len(self.mid) else None
        self.max_half = float(self.half.max()) if len(self.half) else 0.0

    def query(self, point_xy: np.ndarray, radius_m: float = 50.0, max_candidates: int | None = None):
        p = np.asarray(point_xy, dtype=float)
        if self.tree is None:
            return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
        idx = np.asarray(self.tree.query_ball_point(p, radius_m + self.max_half), dtype=int)
        if not len(idx):
            # Fallback: ensure the nearest arcs are still considered if radius search is empty.
            _, idx = self.tree.query(p, k=min(8, len(self.mid)))
            idx = np.atleast_1d(idx).astype(int)
        d, frac = point_segment(p, self.a[idx], self.b[idx])
        keep = d <= radius_m
        idx, d, frac = idx[keep], np.asarray(d)[keep], np.asarray(frac)[keep]
        if max_candidates is not None and len(idx) > max_candidates:
            order = np.argsort(d)[:max_candidates]
            idx, d, frac = idx[order], d[order], frac[order]
        order = np.argsort(d)
        return idx[order], d[order], frac[order]


def generate_candidates(trip, radius_m: float = 50.0, max_candidates: int | None = 30):
    """Returns per-observation dicts containing arc indices, distances, and fractions."""
    if trip.P is None or trip.N is None:
        raise ValueError("Call src.common.project(trip) before candidate generation")
    index = ArcCandidateIndex(trip.N, trip.arcs)
    out = []
    for t, p in enumerate(trip.P):
        arc_idx, dist, frac = index.query(p, radius_m, max_candidates)
        out.append({"observation_index": t, "arc_indices": arc_idx, "distance_m": dist, "fraction": frac})
    return out
