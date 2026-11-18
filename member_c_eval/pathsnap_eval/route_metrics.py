from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sort_observations(df: pd.DataFrame, observation_col: str) -> pd.DataFrame:
    result = df.copy()
    numeric_order = pd.to_numeric(result[observation_col], errors="coerce")
    if numeric_order.notna().all():
        result["__observation_order"] = numeric_order
    else:
        result["__observation_order"] = result[observation_col].astype(str)
    return result.sort_values("__observation_order", kind="stable")


def _load_arcs(data_root: Path, partition: str, trip_id: str) -> np.ndarray | None:
    path = data_root / partition / str(trip_id) / f"{trip_id}.arcs"
    if not path.exists():
        return None
    try:
        arcs = pd.read_csv(path, sep=r"\s+", header=None, names=["from_node", "to_node"])
    except Exception:
        return None
    if arcs.empty or arcs.shape[1] != 2:
        return None
    return arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)


def route_continuity(
    point_predictions: pd.DataFrame,
    data_root: Path,
    partition: str,
    cfg: Any,
) -> dict[str, float | int]:
    """Compute directed-edge continuity from the per-trip .arcs files.

    A repeated arc at two successive GPS observations is continuous. Otherwise,
    arc i -> arc i+1 is continuous only when to_node(i) == from_node(i+1).
    Pairs spanning timestamp gaps over MAX_CONTIGUOUS_GAP_SECONDS are skipped.
    """
    required = {cfg.TRIP_COLUMN, cfg.OBSERVATION_COLUMN, "predicted_arc_id"}
    if point_predictions.empty or not required.issubset(point_predictions.columns):
        return {"route_continuity": float("nan"), "continuity_pairs": 0, "invalid_arc_predictions": 0}

    all_pairs = 0
    connected_pairs = 0
    invalid_arc_predictions = 0
    trip_continuities: list[float] = []

    for trip_id, trip_df in point_predictions.groupby(cfg.TRIP_COLUMN, sort=False):
        arcs = _load_arcs(Path(data_root), partition, str(trip_id))
        if arcs is None:
            continue
        ordered = _sort_observations(trip_df, cfg.OBSERVATION_COLUMN)
        arc_ids = pd.to_numeric(ordered["predicted_arc_id"], errors="coerce").to_numpy()
        observation_indices = pd.to_numeric(
            ordered[cfg.OBSERVATION_COLUMN], errors="coerce"
        ).to_numpy()
        timestamps = (
            pd.to_numeric(ordered[cfg.TIMESTAMP_COLUMN], errors="coerce").to_numpy()
            if cfg.TIMESTAMP_COLUMN in ordered.columns
            else None
        )

        valid_edges: list[tuple[int, int] | None] = []
        for arc_id in arc_ids:
            if not np.isfinite(arc_id):
                valid_edges.append(None)
                invalid_arc_predictions += 1
                continue
            idx = int(arc_id)
            if idx < 0 or idx >= len(arcs):
                valid_edges.append(None)
                invalid_arc_predictions += 1
                continue
            valid_edges.append((int(arcs[idx, 0]), int(arcs[idx, 1])))

        trip_pairs = 0
        trip_connected = 0
        for i in range(len(valid_edges) - 1):
            previous_edge, next_edge = valid_edges[i], valid_edges[i + 1]
            if previous_edge is None or next_edge is None:
                continue
            # Do not treat rows separated by omitted observations as adjacent.
            if (
                np.isfinite(observation_indices[i])
                and np.isfinite(observation_indices[i + 1])
                and observation_indices[i + 1] - observation_indices[i] != 1
            ):
                continue
            if timestamps is not None and np.isfinite(timestamps[i]) and np.isfinite(timestamps[i + 1]):
                gap = float(timestamps[i + 1] - timestamps[i])
                if gap < 0 or gap > float(cfg.MAX_CONTIGUOUS_GAP_SECONDS):
                    continue
            trip_pairs += 1
            if previous_edge == next_edge or previous_edge[1] == next_edge[0]:
                trip_connected += 1

        if trip_pairs:
            trip_continuities.append(trip_connected / trip_pairs)
            all_pairs += trip_pairs
            connected_pairs += trip_connected

    score = connected_pairs / all_pairs if all_pairs else float("nan")
    return {
        "route_continuity": float(score),
        "continuity_pairs": int(all_pairs),
        "invalid_arc_predictions": int(invalid_arc_predictions),
    }
