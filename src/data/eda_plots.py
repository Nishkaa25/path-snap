"""EDA plots and extra tables (dataset PDF, section 4).

Usage (from the repo root):
    python -m src.data.eda_plots --root data --out results/eda

Writes PNG figures to <out>/figures and CSV tables to <out>.
The per-trip arrays (GPS-to-reference distances and time steps) are cached in
<out>/eda_cache.npz so a re-run is fast; use --recompute after changing the data.
Every figure says which partition (train / student_test) it uses.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from src.common import load_trip, project
from src.data.labels import align_labels

PARTS = ["train", "student_test"]
COL = {"train": "#2a78d6", "student_test": "#eb6834"}   # one colour per partition, in every figure
GPS_C, ROUTE_C, NET_C = "#2a78d6", "#008300", "#b9b8b2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"

MASK_M = 50.0          # GPS point is "far" if its distance to the reference arc exceeds this (metres)
EXCLUDE_SHARE = 0.5    # suggest excluding a trip when more than this share of points is far
MASK_SHARE = 0.02      # suggest masking far points when at least this share is far


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })


def trip_dirs(root, parts):
    for p in parts:
        for d in sorted((Path(root) / p).iterdir()):
            if d.is_dir():
                yield p, d


# ---------------------------------------------------------------- data collection
def collect(root, out, parts=PARTS, recompute=False):
    """Per trip: GPS-to-reference-arc distance (m) for every point, and time steps (s)."""
    cache = Path(out) / "eda_cache.npz"
    if cache.exists() and not recompute:
        return np.load(cache, allow_pickle=True)["data"].item()
    data = {}
    for p, d in trip_dirs(root, parts):
        tr = project(load_trip(d, with_route=True))
        _, dist = align_labels(tr, return_dist=True)
        data[(p, tr.trip_id)] = dict(dist=dist.astype(np.float32), dt=np.diff(tr.t).astype(np.float32))
        print("collected", p, tr.trip_id, flush=True)
    np.savez_compressed(cache, data=np.array(data, dtype=object))
    return data


def far_table(data):
    rows = []
    for (p, tid), v in data.items():
        d = v["dist"]
        share = float((d > MASK_M).mean())
        action = "exclude" if share > EXCLUDE_SHARE else ("mask far points" if share >= MASK_SHARE else "keep")
        rows.append(dict(partition=p, trip_id=tid, n_gps=len(d), median_m=round(float(np.median(d)), 2),
                         p95_m=round(float(np.percentile(d, 95)), 2), p99_m=round(float(np.percentile(d, 99)), 2),
                         share_gt_mask=round(share, 4), suggested_action=action))
    return pd.DataFrame(rows)


def missing_table(root, parts=PARTS):
    """Missing / invalid GPS rows, duplicate and non-increasing timestamps, big gaps (per trip)."""
    rows = []
    for p, d in trip_dirs(root, parts):
        raw = np.loadtxt(d / f"{d.name}.track", ndmin=2)
        tr = load_trip(d, with_route=False)
        dt = np.diff(tr.t)
        rows.append(dict(
            partition=p, trip_id=d.name, n_rows_raw=len(raw),
            nan_rows=int((~np.isfinite(raw).all(axis=1)).sum()),
            out_of_range_rows=int((np.isfinite(raw).all(axis=1) & ((np.abs(raw[:, 0]) > 180) | (np.abs(raw[:, 1]) > 90))).sum()),
            rows_dropped=tr.n_dropped, duplicate_timestamps=int((dt == 0).sum()),
            time_goes_backwards=int((dt < 0).sum()), share_dt_equal_1s=round(float((np.abs(dt - 1) < 1e-6).mean()), 4),
            n_gaps_gt10s=int((dt > 10).sum()), n_gaps_gt30s=int((dt > 30).sum()),
            max_gap_s=float(dt.max()) if len(dt) else 0.0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def plot_gaps(data, fig_dir):
    edges = [0.5, 1.5, 2.5, 5.5, 10.5, 30.5, 100.5, 1000.5, np.inf]
    labels = ["1 s", "2 s", "3-5 s", "6-10 s", "11-30 s", "31-100 s", "101-1000 s", ">1000 s"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    w = 0.38
    for i, p in enumerate(PARTS):
        dts = np.concatenate([v["dt"] for (pp, _), v in data.items() if pp == p])
        cnt = np.histogram(dts, bins=edges)[0]
        n_trips = sum(1 for (pp, _) in data if pp == p)
        a.bar(np.arange(8) + (i - 0.5) * w, np.where(cnt > 0, cnt / len(dts) * 100, np.nan), w * 0.92,
              color=COL[p], label=f"{p} ({n_trips} trips)")
    a.set_yscale("log"); a.set_ylim(1e-3, 300)
    a.set_xticks(range(8)); a.set_xticklabels(labels, rotation=30, ha="right")
    a.set_ylabel("share of time steps (%, log scale)"); a.set_title("Time step between GPS points")
    a.grid(axis="x", visible=False); a.legend(loc="upper right")
    mg = pd.DataFrame([(p, tid, float(v["dt"].max()) if len(v["dt"]) else 0.0) for (p, tid), v in data.items()],
                      columns=["p", "tid", "mx"])
    bins = np.logspace(0, 5.3, 28)
    for p in PARTS:
        b.hist(np.clip(mg[mg.p == p].mx, 1, None), bins=bins, color=COL[p], alpha=0.85, label=p, histtype="stepfilled",
               edgecolor="white", linewidth=0.6)
    b.set_xscale("log"); b.axvline(30, color=INK2, lw=1.2, ls="--")
    b.text(32, b.get_ylim()[1] * 0.93, "30 s", color=INK2, fontsize=9, va="top")
    for k, (_, r) in enumerate(mg.nlargest(3, "mx").iterrows()):
        b.annotate(f"trip {r.tid[-2:].lstrip('0') or '0'}: {r.mx:,.0f} s", (r.mx, 0.6), xytext=(-4, 22 + 20 * (2 - k)),
                   textcoords="offset points", ha="right", fontsize=8, color=INK2,
                   arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    b.set_xlabel("largest time gap in a trip (s, log scale)"); b.set_ylabel("number of trips")
    b.set_title("Largest gap per trip"); b.legend(loc="upper right")
    fig.suptitle("Timestamp gaps. Partitions: train (80 trips) and student_test (10 trips)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(Path(fig_dir) / "gap_histogram.png", dpi=200); plt.close(fig)


def plot_dist(data, ft, fig_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    bins = np.logspace(-1, 3.4, 50)
    for p in PARTS:
        x = np.clip(np.concatenate([v["dist"] for (pp, _), v in data.items() if pp == p]), 0.1, None)
        a.hist(x, bins=bins, weights=np.full(len(x), 100.0 / len(x)), color=COL[p], alpha=0.35, histtype="stepfilled")
        a.hist(x, bins=bins, weights=np.full(len(x), 100.0 / len(x)), color=COL[p], histtype="step", lw=1.6,
               label=f"{p}: median {np.median(x):.1f} m, p95 {np.percentile(x, 95):.1f} m")
    a.set_xscale("log"); a.axvline(MASK_M, color=INK2, ls="--", lw=1.2); a.set_ylim(0, 14)
    a.text(MASK_M * 1.08, a.get_ylim()[1] * 0.62, f"{MASK_M:.0f} m mask threshold", color=INK2, fontsize=9, va="top")
    a.set_xlabel("distance from GPS point to its reference arc (m, log scale)"); a.set_ylabel("share of GPS points (%)")
    a.set_title("GPS-to-reference distance, all points"); a.legend(loc="upper left", fontsize=8.5)
    s = ft.sort_values("p95_m").reset_index(drop=True)
    x = np.arange(len(s))
    for part, filled in (("train", True), ("student_test", False)):
        m = (s.partition == part).values
        for stat, marker, c in (("median_m", "o", "#1baf7a"), ("p95_m", "^", "#6250d6")):
            b.scatter(x[m], s.loc[m, stat], marker=marker, s=26, linewidths=1.4,
                      facecolors=c if filled else "none", edgecolors=c, zorder=3)
    b.set_yscale("log"); b.axhline(MASK_M, color=INK2, ls="--", lw=1.2)
    offs = [(-9, 0, "right"), (-9, 0, "right"), (0, 8, "center"), (-9, 0, "right"), (-9, 0, "right"), (0, 8, "center")]
    for (i, r), (dx, dy, ha) in zip(s.tail(6).iterrows(), offs):
        b.annotate(r.trip_id[-2:].lstrip("0") or "0", (i, r.p95_m), xytext=(dx, dy), textcoords="offset points",
                   ha=ha, va="center", fontsize=8, color=INK2)
    b.set_xlabel("trips, ranked by p95 (labels = trip id of the six worst)"); b.set_ylabel("distance (m, log scale)")
    b.set_xticks([]); b.grid(axis="x", visible=False)
    b.set_title("Per trip: median and p95")
    b.legend(handles=[Line2D([], [], marker="o", ls="", color="#1baf7a", label="median"),
                      Line2D([], [], marker="^", ls="", color="#6250d6", label="p95"),
                      Line2D([], [], marker="o", ls="", mfc="none", color=INK2, label="open marker = student_test"),
                      Line2D([], [], marker="o", ls="", color=INK2, label="filled marker = train")],
             loc="upper left", fontsize=8.5)
    fig.suptitle("Reference-label quality. Partitions: train (80 trips) and student_test (10 trips)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(Path(fig_dir) / "gps_to_reference_distance.png", dpi=200); plt.close(fig)


def plot_alignment(data, ft, fig_dir, normal=("train", "00000001"), problem=("train", "00000033")):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    ymax = 10 ** np.ceil(np.log10(max(data[normal]["dist"].max(), data[problem]["dist"].max())))
    for ax, key, title in ((axs[0], normal, "Normal trip"), (axs[1], problem, "Problem trip")):
        d = data[key]["dist"]
        ax.scatter(np.arange(len(d)), np.clip(d, 0.1, None), s=3, color=COL[key[0]], alpha=0.6, linewidths=0)
        ax.set_yscale("log"); ax.set_ylim(0.1, ymax); ax.axhline(MASK_M, color=INK2, ls="--", lw=1.2)
        ax.set_xlabel("GPS point index (time order)"); ax.set_ylabel("distance to assigned arc (m, log scale)")
        ax.set_title(f"{title}: {key[0]} {key[1]}")
        ax.text(0.97, 0.96, f"median {np.median(d):.1f} m\n{(d > MASK_M).mean() * 100:.0f}% of points > {MASK_M:.0f} m",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2)
    s = ft.sort_values("share_gt_mask", ascending=False).reset_index(drop=True)
    cols = [COL[p] for p in s.partition]
    axs[2].bar(np.arange(len(s)), s.share_gt_mask * 100, color=cols, width=0.85)
    axs[2].axhline(EXCLUDE_SHARE * 100, color=INK2, ls="--", lw=1.2)
    axs[2].text(len(s) * 0.45, EXCLUDE_SHARE * 100 + 2, "exclude above 50%", ha="left", fontsize=8.5, color=INK2)
    for i, r in s.head(2).iterrows():
        axs[2].text(i + 0.9, r.share_gt_mask * 100, "trip " + (r.trip_id.lstrip("0") or "0"), ha="left", va="center", fontsize=8.5, color=INK2)
    nxt = s.iloc[2:8]
    axs[2].text(0.30, 0.30, "next: trips " + ", ".join(t.lstrip("0") for t in nxt.trip_id) + f"\n({nxt.share_gt_mask.min() * 100:.0f}-{nxt.share_gt_mask.max() * 100:.0f}% far, mask only)",
                transform=axs[2].transAxes, fontsize=8.5, color=INK2, ha="left", va="center")
    axs[2].set_xticks([]); axs[2].grid(axis="x", visible=False)
    axs[2].set_xlabel("trips, ranked"); axs[2].set_ylabel(f"points more than {MASK_M:.0f} m from reference (%)")
    axs[2].set_title("Share of far points per trip")
    axs[2].legend(handles=[Line2D([], [], marker="s", ls="", color=COL[p], label=p) for p in PARTS], loc="upper right", fontsize=8.5)
    fig.suptitle("Alignment sanity check (monotone alignment to the reference route). Partitions: train and student_test",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(Path(fig_dir) / "alignment_check.png", dpi=200); plt.close(fig)


def _route_segments(tr):
    a = tr.arcs[tr.route]
    return np.stack([tr.N[a[:, 0]], tr.N[a[:, 1]]], axis=1) / 1000.0


def plot_trajectories(root, fig_dir, picks=(("train", "00000001"), ("train", "00000013"), ("train", "00000033"),
                                            ("train", "00000043"), ("train", "00000054"), ("train", "00000078"),
                                            ("train", "00000090"), ("student_test", "00000000"), ("student_test", "00000056"))):
    picks = [(p, t) for p, t in picks if (Path(root) / p / t).is_dir()]
    n = len(picks); nc = 3; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(12.5, 4.0 * nr))
    axs = np.atleast_1d(axs).ravel()
    for ax, (p, t) in zip(axs, picks):
        tr = project(load_trip(Path(root) / p / t, with_route=True))
        ax.add_collection(LineCollection(_route_segments(tr), colors=ROUTE_C, linewidths=2.2, alpha=0.8, zorder=2))
        ax.scatter(tr.P[:, 0] / 1000, tr.P[:, 1] / 1000, s=3, color=GPS_C, alpha=0.7, linewidths=0, zorder=3)
        ax.autoscale(); ax.set_aspect("equal", adjustable="datalim"); ax.margins(0.05)
        ax.set_title(f"{p} {t}  ({len(tr.P):,} points)", fontsize=10); ax.tick_params(labelsize=8)
        ax.set_xlabel("x (km)", fontsize=9); ax.set_ylabel("y (km)", fontsize=9)
    for ax in axs[n:]:
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GPS_C, label="GPS points"),
                        Line2D([], [], color=ROUTE_C, lw=2.5, label="reference route")], loc="lower center", ncol=2, fontsize=10)
    fig.suptitle("GPS trajectories (local projection, metres from each trip's centroid). Partitions: train and student_test",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(Path(fig_dir) / "trajectories.png", dpi=200); plt.close(fig)


def plot_network(root, fig_dir, picks=(("train", "00000075"), ("train", "00000008")), pad_m=300.0):
    fig, axs = plt.subplots(1, len(picks), figsize=(6.4 * len(picks), 6.4))
    for ax, (p, t) in zip(np.atleast_1d(axs), picks):
        tr = project(load_trip(Path(root) / p / t, with_route=True))
        lo, hi = tr.P.min(0) - pad_m, tr.P.max(0) + pad_m
        inside = ((tr.N >= lo) & (tr.N <= hi)).all(axis=1)
        sel = inside[tr.arcs[:, 0]] & inside[tr.arcs[:, 1]]
        seg = np.stack([tr.N[tr.arcs[sel, 0]], tr.N[tr.arcs[sel, 1]]], axis=1) / 1000.0
        ax.add_collection(LineCollection(seg, colors=NET_C, linewidths=0.6, zorder=1))
        ax.add_collection(LineCollection(_route_segments(tr), colors=ROUTE_C, linewidths=2.0, alpha=0.8, zorder=2))
        ax.scatter(tr.P[:, 0] / 1000, tr.P[:, 1] / 1000, s=3, color=GPS_C, alpha=0.7, linewidths=0, zorder=3)
        ax.set_xlim(lo[0] / 1000, hi[0] / 1000); ax.set_ylim(lo[1] / 1000, hi[1] / 1000); ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{p} {t}: {len(tr.arcs):,} arcs in file, {int(sel.sum()):,} drawn", fontsize=10)
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    fig.legend(handles=[Line2D([], [], color=NET_C, lw=1.5, label="road network (arcs)"),
                        Line2D([], [], color=ROUTE_C, lw=2.5, label="reference route"),
                        Line2D([], [], marker="o", ls="", color=GPS_C, label="GPS points")], loc="lower center", ncol=3, fontsize=9.5)
    fig.suptitle("Road network near the trip. Partition: train", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(Path(fig_dir) / "road_network.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data"); ap.add_argument("--out", default="results/eda")
    ap.add_argument("--parts", nargs="+", default=PARTS); ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    style()
    data = collect(a.root, out, a.parts, a.recompute)
    ft = far_table(data); ft.to_csv(out / "far_points_by_trip.csv", index=False)
    mt = missing_table(a.root, a.parts); mt.to_csv(out / "missing_values.csv", index=False)
    mt[mt.max_gap_s > 30].sort_values("max_gap_s", ascending=False).to_csv(out / "trips_with_gap_gt30s.csv", index=False)
    plot_gaps(data, fig_dir); plot_dist(data, ft, fig_dir); plot_alignment(data, ft, fig_dir)
    plot_trajectories(a.root, fig_dir); plot_network(a.root, fig_dir)
    print("\nTrips by suggested action:\n", ft.groupby(["partition", "suggested_action"]).size().to_string())
    print("\nMissing / invalid rows total:", int(mt.rows_dropped.sum()), "| duplicate timestamps:", int(mt.duplicate_timestamps.sum()),
          "| time goes backwards:", int(mt.time_goes_backwards.sum()))
    print("Figures written to", fig_dir)


if __name__ == "__main__":
    main()
