from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .evaluator import predict_and_measure


def grouped_cross_validation(
    model_name: str,
    estimator,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Any,
    n_splits: int | None = None,
    params: dict | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate with GroupKFold so rows from one trip never cross folds."""
    n_splits = int(n_splits or cfg.CV_SPLITS)
    groups = train_df[cfg.TRIP_COLUMN].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"GroupKFold({n_splits}) needs at least {n_splits} unique trips; "
            f"only {len(unique_groups)} found."
        )

    X = train_df[feature_cols]
    y = train_df[cfg.TARGET_COLUMN].to_numpy(dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    fold_records = []

    for fold_number, (train_indices, valid_indices) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        fold_train = train_df.iloc[train_indices].copy()
        fold_valid = train_df.iloc[valid_indices].copy()

        train_trip_ids = set(fold_train[cfg.TRIP_COLUMN].astype(str).unique())
        valid_trip_ids = set(fold_valid[cfg.TRIP_COLUMN].astype(str).unique())
        overlap = train_trip_ids.intersection(valid_trip_ids)
        if overlap:
            raise RuntimeError(f"Trip leakage between CV fold {fold_number}: {overlap}")

        fold_model = clone(estimator)
        if params:
            fold_model.set_params(**params)

        fit_start = time.perf_counter()
        fold_model.fit(
            fold_train[feature_cols].replace([np.inf, -np.inf], np.nan),
            fold_train[cfg.TARGET_COLUMN].to_numpy(dtype=int),
        )
        fit_seconds = time.perf_counter() - fit_start

        metrics, _, _ = predict_and_measure(
            fold_model,
            fold_valid,
            feature_cols,
            cfg,
            cfg.TRAIN_PARTITION,
        )
        record = {
            "model": model_name,
            "fold": fold_number,
            "train_trip_count": len(train_trip_ids),
            "validation_trip_count": len(valid_trip_ids),
            "fit_time_seconds": float(fit_seconds),
        }
        record.update(metrics)
        fold_records.append(record)

    fold_df = pd.DataFrame(fold_records)
    summary = {
        "model": model_name,
        "cv_macro_f1_mean": float(fold_df["candidate_macro_f1"].mean()),
        "cv_macro_f1_std": float(fold_df["candidate_macro_f1"].std(ddof=1)),
        "cv_top1_accuracy_mean": float(fold_df["observation_top1_accuracy"].mean()),
        "cv_top1_accuracy_std": float(fold_df["observation_top1_accuracy"].std(ddof=1)),
        "cv_observation_arc_macro_f1_mean": float(fold_df["observation_arc_macro_f1"].mean()),
        "cv_observation_arc_macro_f1_std": float(fold_df["observation_arc_macro_f1"].std(ddof=1)),
        "cv_candidate_miss_rate_mean": float(fold_df["candidate_miss_rate"].mean()),
        "cv_route_continuity_mean": float(fold_df["route_continuity"].mean()),
        "cv_route_continuity_std": float(fold_df["route_continuity"].std(ddof=1)),
        "cv_log_loss_mean": float(fold_df["log_loss"].mean()),
        "cv_fit_time_seconds_mean": float(fold_df["fit_time_seconds"].mean()),
        "cv_inference_time_seconds_mean": float(fold_df["inference_time_seconds"].mean()),
    }
    return summary, fold_df
