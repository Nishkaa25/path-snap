
"""
GroupKFold cross-validation for PathSnap.

- Splits data by trip_id to prevent trip-level data leakage.
- Computes Macro F1 and Log Loss for each fold.
- Saves out-of-fold (OOF) predictions.
- Optionally computes Top-1 accuracy when an observation ID column is supplied.
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss


def run_group_cv(
    df,
    model,
    feature_cols,
    target_col="label",
    group_col="trip_id",
    observation_col=None,
    n_splits=5,
    threshold=0.5,
):
    """
    Run GroupKFold cross-validation.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed candidate-level dataset.
    model : sklearn-compatible estimator
        Classifier implementing fit() and predict_proba().
    feature_cols : list
        Names of model input feature columns.
    target_col : str
        Binary target column (1 = reference candidate, 0 = other).
    group_col : str
        Trip identifier used for grouping.
    observation_col : str or None
        GPS observation identifier, used for Top-1 accuracy.
    n_splits : int
        Number of cross-validation folds.
    threshold : float
        Probability threshold for binary classification.

    Returns
    -------
    metrics_df : pandas.DataFrame
        Metrics for each fold and their mean/std.
    oof_df : pandas.DataFrame
        Out-of-fold predictions for every input row.
    """

    required_cols = feature_cols + [target_col, group_col]

    if observation_col is not None:
        required_cols.append(observation_col)

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[group_col].nunique() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique trips, "
            f"but found {df[group_col].nunique()}."
        )

    X = df[feature_cols]
    y = df[target_col].astype(int)
    groups = df[group_col]

    splitter = GroupKFold(n_splits=n_splits)

    fold_results = []
    oof_parts = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Create a fresh model for every fold.
        fold_model = clone(model)

        # Train only on the training trips.
        fold_model.fit(X_train, y_train)

        # Predict probabilities for validation candidates.
        probabilities = fold_model.predict_proba(X_val)

        classes = list(fold_model.classes_)

        if 1 not in classes:
            raise ValueError("The training fold contains no positive class (1).")

        positive_col = classes.index(1)
        y_prob = probabilities[:, positive_col]

        # Convert probabilities into binary predictions.
        y_pred = (y_prob >= threshold).astype(int)

        # Candidate-level evaluation.
        macro_f1 = f1_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )

        fold_logloss = log_loss(
            y_val,
            probabilities,
            labels=classes,
        )

        # Top-1: select the highest-scoring candidate per GPS observation.
        top1 = np.nan

        if observation_col is not None:
            validation_rows = df.iloc[val_idx].copy()
            validation_rows["_probability"] = y_prob

            group_keys = [group_col, observation_col]

            best_candidates = validation_rows.loc[
                validation_rows.groupby(group_keys)["_probability"].idxmax()
            ]

            top1 = float((best_candidates[target_col] == 1).mean())

        fold_results.append({
            "fold": fold,
            "train_trips": groups.iloc[train_idx].nunique(),
            "validation_trips": groups.iloc[val_idx].nunique(),
            "macro_f1": macro_f1,
            "log_loss": fold_logloss,
            "top1_accuracy": top1,
        })

        # Store predictions against their original dataframe indices.
        oof_fold = df.iloc[val_idx][
            [group_col, target_col]
        ].copy()

        if observation_col is not None:
            oof_fold[observation_col] = df.iloc[val_idx][observation_col].values

        oof_fold["fold"] = fold
        oof_fold["probability"] = y_prob
        oof_fold["prediction"] = y_pred

        oof_parts.append(oof_fold)

        print(
            f"Fold {fold}/{n_splits} | "
            f"Macro F1: {macro_f1:.4f} | "
            f"Log Loss: {fold_logloss:.4f} | "
            f"Top-1: {top1:.4f}"
        )

    # Combine results from all folds.
    metrics_df = pd.DataFrame(fold_results)

    # Add mean and standard deviation summaries.
    numeric_metrics = [
        "macro_f1",
        "log_loss",
        "top1_accuracy",
    ]

    summary = {
        "fold": "mean",
        "train_trips": np.nan,
        "validation_trips": np.nan,
    }

    for metric in numeric_metrics:
        summary[metric] = metrics_df[metric].mean()

    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([summary])],
        ignore_index=True,
    )

    std_summary = {
        "fold": "std",
        "train_trips": np.nan,
        "validation_trips": np.nan,
    }

    for metric in numeric_metrics:
        std_summary[metric] = metrics_df.loc[
            metrics_df["fold"] != "mean", metric
        ].std()

    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([std_summary])],
        ignore_index=True,
    )

    oof_df = pd.concat(oof_parts).sort_index()

    return metrics_df, oof_df
