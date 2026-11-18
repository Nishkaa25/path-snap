from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from .route_metrics import route_continuity


def _positive_scores(estimator, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ranking scores for class 1 and probabilities when available."""
    probabilities = None
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X))
        classes = list(getattr(estimator, "classes_", [0, 1]))
        if 1 in classes:
            positive_index = classes.index(1)
            scores = probabilities[:, positive_index]
        elif probabilities.shape[1] == 1:
            scores = probabilities[:, 0]
        else:
            raise ValueError("Estimator predict_proba() does not contain target class 1.")
        return np.asarray(scores, dtype=float), probabilities

    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float).reshape(-1), None

    return np.asarray(estimator.predict(X), dtype=float).reshape(-1), None


def _observation_predictions(
    df: pd.DataFrame,
    scores: np.ndarray,
    cfg: Any,
) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    work["__score"] = scores
    group_cols = [cfg.TRIP_COLUMN, cfg.OBSERVATION_COLUMN]
    # idxmax uses the first occurrence on ties, making results deterministic.
    best_indices = work.groupby(group_cols, sort=False)["__score"].idxmax()
    chosen = work.loc[best_indices].copy()
    chosen["predicted_arc_id"] = chosen[cfg.CANDIDATE_ARC_COLUMN].astype(int)
    chosen["predicted_score"] = chosen["__score"].astype(float)
    chosen["top1_correct"] = chosen[cfg.TARGET_COLUMN].astype(int).eq(1)
    chosen = chosen.drop(columns=["__score"])
    return chosen


def _metric_bundle(
    df: pd.DataFrame,
    predicted_labels: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray | None,
    cfg: Any,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    y_true = df[cfg.TARGET_COLUMN].to_numpy(dtype=int)
    y_pred = np.asarray(predicted_labels, dtype=int)

    candidate_accuracy = float(accuracy_score(y_true, y_pred))
    candidate_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    candidate_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    candidate_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    logloss_value = float("nan")
    if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] == 2:
        # The target is constrained to {0, 1}; sklearn classifiers sort these
        # integer classes as [0, 1], so this is a comparable candidate Log Loss.
        try:
            logloss_value = float(log_loss(y_true, probabilities, labels=[0, 1]))
        except (ValueError, TypeError):
            logloss_value = float("nan")

    point_preds = _observation_predictions(df, scores, cfg)
    positive_per_obs = df.groupby([cfg.TRIP_COLUMN, cfg.OBSERVATION_COLUMN], sort=False)[cfg.TARGET_COLUMN].max()

    # dataset.py intentionally omits observations whose positive reference arc
    # is too far away or absent from the generated candidate set. The prepared
    # CSV carries each trip's full usable GPS-observation count so those omitted
    # observations are counted as candidate misses and Top-1 errors.
    total_observations_col = getattr(cfg, "TOTAL_OBSERVATION_COLUMN", "total_observations")
    if total_observations_col in df.columns:
        total_observations = int(
            df.groupby(cfg.TRIP_COLUMN, sort=False)[total_observations_col].first().sum()
        )
    else:
        total_observations = int(len(positive_per_obs))

    correct_candidate_observations = int(positive_per_obs.sum()) if len(positive_per_obs) else 0
    candidate_miss_rate = (
        float(1.0 - correct_candidate_observations / total_observations)
        if total_observations > 0 else float("nan")
    )
    top1_correct_count = int(point_preds["top1_correct"].sum()) if len(point_preds) else 0
    top1_accuracy = (
        float(top1_correct_count / total_observations)
        if total_observations > 0 else float("nan")
    )

    obs_accuracy = float("nan")
    obs_precision = float("nan")
    obs_recall = float("nan")
    obs_f1 = float("nan")
    sequence_exact_match = float("nan")

    if cfg.TRUE_ARC_COLUMN in point_preds.columns and point_preds[cfg.TRUE_ARC_COLUMN].notna().all():
        y_arc_true = point_preds[cfg.TRUE_ARC_COLUMN].astype(int).to_numpy()
        y_arc_pred = point_preds["predicted_arc_id"].astype(int).to_numpy()
        obs_accuracy = float(accuracy_score(y_arc_true, y_arc_pred))
        obs_precision = float(precision_score(y_arc_true, y_arc_pred, average="macro", zero_division=0))
        obs_recall = float(recall_score(y_arc_true, y_arc_pred, average="macro", zero_division=0))
        obs_f1 = float(f1_score(y_arc_true, y_arc_pred, average="macro", zero_division=0))
        point_preds["true_arc_id"] = y_arc_true
        point_preds["observation_arc_correct"] = y_arc_true == y_arc_pred
        exact_by_trip = point_preds.groupby(cfg.TRIP_COLUMN, sort=False)["observation_arc_correct"].all()
        sequence_exact_match = float(exact_by_trip.mean()) if len(exact_by_trip) else float("nan")

    # Report Macro-F1 at the per-observation arc level when true_arc_id exists;
    # otherwise report candidate-row Macro-F1 and label it clearly in outputs.
    report_macro_f1 = obs_f1 if np.isfinite(obs_f1) else candidate_f1
    report_precision = obs_precision if np.isfinite(obs_precision) else candidate_precision
    report_recall = obs_recall if np.isfinite(obs_recall) else candidate_recall

    metrics = {
        "accuracy": candidate_accuracy,
        "precision_macro": report_precision,
        "recall_macro": report_recall,
        "macro_f1": report_macro_f1,
        "candidate_accuracy": candidate_accuracy,
        "candidate_precision_macro": candidate_precision,
        "candidate_recall_macro": candidate_recall,
        "candidate_macro_f1": candidate_f1,
        "observation_arc_accuracy": obs_accuracy,
        "observation_arc_precision_macro": obs_precision,
        "observation_arc_recall_macro": obs_recall,
        "observation_arc_macro_f1": obs_f1,
        "observation_top1_accuracy": top1_accuracy,
        "candidate_miss_rate": candidate_miss_rate,
        "log_loss": logloss_value,
        "sequence_exact_match_rate": sequence_exact_match,
    }
    return metrics, point_preds, confusion_matrix(y_true, y_pred, labels=[0, 1])


def fit_and_evaluate(
    estimator,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Any,
    eval_partition: str,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, Any]:
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_df[cfg.TARGET_COLUMN].to_numpy(dtype=int)
    X_eval = eval_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_eval = eval_df[cfg.TARGET_COLUMN].to_numpy(dtype=int)

    start_fit = time.perf_counter()
    estimator.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start_fit

    start_inference = time.perf_counter()
    predicted_labels = np.asarray(estimator.predict(X_eval), dtype=int)
    scores, probabilities = _positive_scores(estimator, X_eval)
    inference_seconds = time.perf_counter() - start_inference

    metrics, point_preds, cm = _metric_bundle(
        eval_df, predicted_labels, scores, probabilities, cfg
    )
    continuity = route_continuity(point_preds, cfg.DATA_ROOT, eval_partition, cfg)
    metrics.update(continuity)
    metrics["training_time_seconds"] = float(training_seconds)
    metrics["inference_time_seconds"] = float(inference_seconds)
    metrics["inference_ms_per_candidate"] = (
        float(inference_seconds * 1000 / max(len(eval_df), 1))
    )
    # point_preds contains one selected candidate per GPS observation.
    return metrics, point_preds, cm, estimator


def predict_and_measure(
    estimator,
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Any,
    partition: str,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    """Evaluate an already-fitted estimator (used during CV folds)."""
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_true = df[cfg.TARGET_COLUMN].to_numpy(dtype=int)
    start = time.perf_counter()
    predicted_labels = np.asarray(estimator.predict(X), dtype=int)
    scores, probabilities = _positive_scores(estimator, X)
    inference_seconds = time.perf_counter() - start
    metrics, point_preds, cm = _metric_bundle(df, predicted_labels, scores, probabilities, cfg)
    continuity = route_continuity(point_preds, cfg.DATA_ROOT, partition, cfg)
    metrics.update(continuity)
    metrics["inference_time_seconds"] = float(inference_seconds)
    metrics["inference_ms_per_candidate"] = float(inference_seconds * 1000 / max(len(df), 1))
    return metrics, point_preds, cm


def evaluate_predictions(
    estimator,
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Any,
    partition: str,
):
    """Public helper to score the student-test table with a fitted estimator."""
    return predict_and_measure(estimator, df, feature_cols, cfg, partition)
