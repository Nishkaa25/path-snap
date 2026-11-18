"""Dataset inventory + data-quality table (dataset PDF, section 4).
Usage: python -m src.data.eda --root data --out results/eda"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.common import load_trip, project
from src.data.labels import align_labels


def trip_row(part, d):
    tr = project(load_trip(d, with_route=True))
    T, N, M = len(tr.P), len(tr.nodes), len(tr.arcs)
    dt = np.diff(tr.t)
    step = np.linalg.norm(np.diff(tr.P, axis=0), axis=1)
    ext = tr.P.max(0) - tr.P.min(0)
    used = np.zeros(N, bool); used[tr.arcs.ravel()] = True
    g = coo_matrix((np.ones(M), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(N, N))
    ncomp, lab = connected_components(g, directed=False)
    giant = np.bincount(lab).max() / N
    lab_arc, dist = align_labels(tr, return_dist=True)
    rt = tr.route
    return dict(partition=part, trip_id=tr.trip_id, n_gps=T, dropped_gps=tr.n_dropped,
                n_nodes=N, n_arcs=M, route_len=len(rt),
                file_kb=round(sum(f.stat().st_size for f in Path(d).iterdir()) / 1024),
                lon0=round(tr.origin[0], 4), lat0=round(tr.origin[1], 4),
                extent_km=round(float(np.hypot(*ext)) / 1000, 2), track_km=round(float(step.sum()) / 1000, 2),
                dt_eq_1s=round(float((dt == 1).mean()), 4), dt_max_s=float(dt.max()) if len(dt) else 0,
                n_gaps_gt10s=int((dt > 10).sum()), time_not_increasing=int((dt <= 0).sum()),
                isolated_nodes=int((~used).sum()), n_components=int(ncomp), giant_comp_share=round(float(giant), 4),
                route_connected=round(float(np.mean(tr.arcs[rt[:-1], 1] == tr.arcs[rt[1:], 0])), 4) if len(rt) > 1 else 1.0,
                gps_to_ref_med_m=round(float(np.median(dist)), 2), gps_to_ref_p95_m=round(float(np.percentile(dist, 95)), 2),
                gps_to_ref_p99_m=round(float(np.percentile(dist, 99)), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data"); ap.add_argument("--out", default="results/eda")
    ap.add_argument("--parts", nargs="+", default=["train", "student_test"])
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = []
    for part in a.parts:
        for d in sorted((Path(a.root) / part).iterdir()):
            if d.is_dir():
                rows.append(trip_row(part, d)); print(part, d.name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "inventory.csv", index=False)
    summ = df.groupby("partition")[["n_gps", "n_nodes", "n_arcs", "route_len", "dt_eq_1s", "gps_to_ref_med_m", "gps_to_ref_p95_m"]].describe().T
    summ.to_csv(Path(a.out) / "summary_by_partition.csv")
    flag = df[(df.gps_to_ref_p95_m > 50) | (df.dt_max_s > 300) | (df.route_connected < 0.99) | (df.giant_comp_share < 0.9) | (df.dropped_gps > 0)]
    flag.to_csv(Path(a.out) / "flagged_trips.csv", index=False)
    print("\nTrips per partition:\n", df.partition.value_counts().to_string())
    print("\nFlagged trips (check on Folium):\n", flag[["partition", "trip_id", "gps_to_ref_p95_m", "dt_max_s", "n_components", "route_connected"]].to_string(index=False))


if __name__ == "__main__":
    main()
