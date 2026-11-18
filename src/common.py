"""Shared loader / projection / geometry helpers. Everyone imports from here."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class TripData:
    trip_id: str
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray                    # timestamps (s)
    nodes: np.ndarray                # (N,2) lon, lat ; row index = node id
    arcs: np.ndarray                 # (M,2) from_node, to_node ; row index = arc id
    route: np.ndarray | None         # reference arc ids (None on the prediction path)
    P: np.ndarray | None = None      # (T,2) GPS in metres (after project)
    N: np.ndarray | None = None      # (N,2) nodes in metres
    origin: tuple | None = None      # (lon0, lat0)
    n_dropped: int = 0               # invalid GPS rows removed


def load_trip(trip_dir, with_route: bool = True) -> TripData:
    """Load one trip folder. with_route=False is the leakage-safe path used by the API."""
    d = Path(trip_dir)
    tid = d.name
    trk = np.loadtxt(d / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(d / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(d / f"{tid}.arcs", dtype=int, ndmin=2)
    route = None
    rp = d / f"{tid}.route"
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=int, ndmin=1)
    ok = np.isfinite(trk).all(axis=1) & (np.abs(trk[:, 0]) <= 180) & (np.abs(trk[:, 1]) <= 90)
    n_dropped = int((~ok).sum())
    trk = trk[ok]
    return TripData(tid, trk[:, 0], trk[:, 1], trk[:, 2], nodes, arcs, route, n_dropped=n_dropped)


def project(trip: TripData) -> TripData:
    """Local equirectangular projection around the trip centroid (metres)."""
    lon0, lat0 = float(trip.lon.mean()), float(trip.lat.mean())
    kx = np.cos(np.radians(lat0)) * 111_320.0
    ky = 110_540.0
    trip.P = np.c_[(trip.lon - lon0) * kx, (trip.lat - lat0) * ky]
    trip.N = np.c_[(trip.nodes[:, 0] - lon0) * kx, (trip.nodes[:, 1] - lat0) * ky]
    trip.origin = (lon0, lat0)
    return trip


def point_segment(P, A, B):
    """Distance from points P to segments A-B (broadcastable). Returns (dist, frac)."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(-1), 1e-9)
    f = np.clip(((P - A) * AB).sum(-1) / L2, 0.0, 1.0)
    C = A + f[..., None] * AB
    return np.sqrt(((P - C) ** 2).sum(-1)), f
