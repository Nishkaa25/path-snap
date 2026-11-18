from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

META_COLUMNS = {
    "trip_id",
    "observation_id",
    "candidate_arc_id",
    "target",
    "true_arc_id",
    "timestamp",
}


def _load_module_from_path(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset script not found: {path}\n"
            "Check DATASET_SCRIPT in config.py."
        )
    spec = importlib.util.spec_from_file_location("pathsnap_user_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import dataset script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_table(df: pd.DataFrame, name: str, cfg: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}.")

    required = [
        cfg.TRIP_COLUMN,
        cfg.OBSERVATION_COLUMN,
        cfg.CANDIDATE_ARC_COLUMN,
        cfg.TARGET_COLUMN,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}.\n"
            "See README_FIRST.md for the required candidate-table schema."
        )

    result = df.copy()
    if result[[cfg.TRIP_COLUMN, cfg.OBSERVATION_COLUMN]].isna().any().any():
        raise ValueError(f"{name} contains missing trip_id or observation_id values.")
    result[cfg.TRIP_COLUMN] = result[cfg.TRIP_COLUMN].astype(str)
    result[cfg.CANDIDATE_ARC_COLUMN] = pd.to_numeric(
        result[cfg.CANDIDATE_ARC_COLUMN], errors="raise"
    ).astype("int64")
    result[cfg.TARGET_COLUMN] = pd.to_numeric(
        result[cfg.TARGET_COLUMN], errors="raise"
    ).astype("int64")

    target_values = set(result[cfg.TARGET_COLUMN].dropna().unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{name}.{cfg.TARGET_COLUMN} must contain only 0 and 1; found {target_values}."
        )
    if len(target_values) < 2:
        raise ValueError(
            f"{name} must contain both target classes 0 and 1."
        )

    if cfg.TRUE_ARC_COLUMN in result.columns:
        result[cfg.TRUE_ARC_COLUMN] = pd.to_numeric(
            result[cfg.TRUE_ARC_COLUMN], errors="coerce"
        )

    if cfg.TIMESTAMP_COLUMN in result.columns:
        result[cfg.TIMESTAMP_COLUMN] = pd.to_numeric(
            result[cfg.TIMESTAMP_COLUMN], errors="coerce"
        )

    # The model sees only numerical feature columns. Metadata, candidate IDs,
    # labels, and optional ground truth are never passed to the estimator.
    feature_cols = cfg.FEATURE_COLUMNS
    if feature_cols is None:
        excluded = {
            cfg.TRIP_COLUMN,
            cfg.OBSERVATION_COLUMN,
            cfg.CANDIDATE_ARC_COLUMN,
            cfg.TARGET_COLUMN,
            cfg.TRUE_ARC_COLUMN,
            cfg.TIMESTAMP_COLUMN,
        }
        feature_cols = [
            c for c in result.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(result[c])
        ]
    else:
        missing_features = [c for c in feature_cols if c not in result.columns]
        if missing_features:
            raise ValueError(f"{name} is missing configured feature columns: {missing_features}")

    if not feature_cols:
        raise ValueError(
            f"No numeric feature columns detected in {name}. "
            "Set FEATURE_COLUMNS in config.py or fix the dataset export."
        )

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(result[c])]
    if non_numeric:
        raise ValueError(
            "All features must be numerical/encoded before model evaluation. "
            f"These configured feature columns are not numeric: {non_numeric}"
        )

    result.attrs["feature_columns"] = list(feature_cols)
    return result


def load_eval_data(cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the 80-trip train table and 10-trip student-test table."""
    if cfg.DATA_SOURCE.lower() == "csv":
        if not Path(cfg.TRAIN_CSV).exists() or not Path(cfg.STUDENT_TEST_CSV).exists():
            raise FileNotFoundError(
                "Processed candidate CSVs were not found. Expected:\n"
                f"  {cfg.TRAIN_CSV}\n  {cfg.STUDENT_TEST_CSV}\n\n"
                "Export the processed candidate tables from ml dataset.py, or set "
                "DATA_SOURCE='module' and add load_pathsnap_eval_data()."
            )
        # Trip IDs are zero-padded folder names (for example 00000008).
        # Read them as strings so pandas does not strip leading zeroes.
        train_df = pd.read_csv(cfg.TRAIN_CSV, dtype={cfg.TRIP_COLUMN: str})
        student_df = pd.read_csv(cfg.STUDENT_TEST_CSV, dtype={cfg.TRIP_COLUMN: str})
    elif cfg.DATA_SOURCE.lower() == "module":
        module = _load_module_from_path(Path(cfg.DATASET_SCRIPT))
        if not hasattr(module, cfg.DATASET_FUNCTION):
            raise AttributeError(
                f"{cfg.DATASET_SCRIPT} must define {cfg.DATASET_FUNCTION}(). "
                "See README_FIRST.md for the required return format."
            )
        loaded = getattr(module, cfg.DATASET_FUNCTION)()
        if isinstance(loaded, dict):
            train_df = loaded.get("train")
            student_df = loaded.get("student_test")
        elif isinstance(loaded, (tuple, list)) and len(loaded) == 2:
            train_df, student_df = loaded
        else:
            raise TypeError(
                f"{cfg.DATASET_FUNCTION}() must return a dict with 'train' and "
                "'student_test', or a 2-tuple of DataFrames."
            )
    else:
        raise ValueError("DATA_SOURCE must be either 'csv' or 'module'.")

    train_df = _validate_table(train_df, "train candidate table", cfg)
    student_df = _validate_table(student_df, "student_test candidate table", cfg)

    train_features = train_df.attrs["feature_columns"]
    student_features = student_df.attrs["feature_columns"]
    if train_features != student_features:
        raise ValueError(
            "Train and student-test feature columns are not identical or in the same order.\n"
            f"Train: {train_features}\nStudent test: {student_features}"
        )

    train_trips = set(train_df[cfg.TRIP_COLUMN].unique())
    student_trips = set(student_df[cfg.TRIP_COLUMN].unique())
    overlap = train_trips.intersection(student_trips)
    if overlap:
        raise ValueError(
            "Data leakage risk: trip IDs appear in both train and student_test: "
            f"{sorted(overlap)[:10]}"
        )

    if len(train_trips) != 80:
        warnings.warn(
            f"Guidelines specify 80 training trips; loaded {len(train_trips)}.",
            RuntimeWarning,
        )
    if len(student_trips) != 10:
        warnings.warn(
            f"Guidelines specify 10 student_test trips; loaded {len(student_trips)}.",
            RuntimeWarning,
        )

    return train_df, student_df, train_features


def get_xy_groups(df: pd.DataFrame, feature_cols: list[str], cfg: Any):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[cfg.TARGET_COLUMN].to_numpy(dtype=int)
    groups = df[cfg.TRIP_COLUMN].astype(str).to_numpy()
    return X, y, groups
