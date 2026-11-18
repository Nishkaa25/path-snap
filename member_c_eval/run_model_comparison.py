from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import config as cfg
from .pathsnap_eval.cross_validation import grouped_cross_validation
from .pathsnap_eval.data import load_eval_data
from .pathsnap_eval.evaluator import fit_and_evaluate
from .pathsnap_eval.models import get_model_registry
from .pathsnap_eval.plots import make_comparison_plots, save_confusion_matrix


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def main() -> int:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = cfg.OUTPUT_DIR / "models"
    prediction_dir = cfg.OUTPUT_DIR / "predictions"
    figure_dir = cfg.OUTPUT_DIR / "figures"
    for path in (model_dir, prediction_dir, figure_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_df, student_df, feature_cols = load_eval_data(cfg)
    print(f"Training trips: {train_df[cfg.TRIP_COLUMN].nunique()} | rows: {len(train_df):,}")
    print(f"Student-test trips: {student_df[cfg.TRIP_COLUMN].nunique()} | rows: {len(student_df):,}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    registry = get_model_registry(cfg)
    cv_summaries = []
    all_cv_folds = []
    test_summaries = []
    comparison_records = []

    for model_name, estimator in registry.items():
        print(f"\n[1/2] Group CV: {model_name}")
        cv_summary, cv_folds = grouped_cross_validation(
            model_name, estimator, train_df, feature_cols, cfg
        )
        cv_summaries.append(cv_summary)
        all_cv_folds.append(cv_folds)

        print(f"[2/2] Fit on all training trips and evaluate on student_test: {model_name}")
        fitted = clone(estimator)
        test_metrics, point_predictions, cm, fitted = fit_and_evaluate(
            fitted,
            train_df,
            student_df,
            feature_cols,
            cfg,
            cfg.STUDENT_TEST_PARTITION,
        )
        model_path = model_dir / f"{_safe_name(model_name)}.joblib"
        # Save the exact estimator configuration used for the run. For a
        # Pipeline this includes the preprocessing and classifier parameters.
        model_params = fitted.get_params(deep=True) if hasattr(fitted, "get_params") else {}
        estimator_for_family = fitted
        if hasattr(fitted, "named_steps") and fitted.named_steps:
            estimator_for_family = list(fitted.named_steps.values())[-1]
        algorithm_family = type(estimator_for_family).__name__
        config_text = json.dumps(model_params, sort_keys=True, default=str)
        joblib.dump(
            {
                "model": fitted,
                "feature_columns": feature_cols,
                "model_name": model_name,
                "target_column": cfg.TARGET_COLUMN,
                "trip_column": cfg.TRIP_COLUMN,
                "observation_column": cfg.OBSERVATION_COLUMN,
                "candidate_arc_column": cfg.CANDIDATE_ARC_COLUMN,
            },
            model_path,
        )
        point_predictions.to_csv(
            prediction_dir / f"student_test_predictions_{_safe_name(model_name)}.csv",
            index=False,
        )
        save_confusion_matrix(cm, model_name, figure_dir)

        record = {
            "model": model_name,
            "algorithm_family": algorithm_family,
            "initial_configuration": config_text,
        }
        record.update(cv_summary)
        record.update({f"student_{key}": value for key, value in test_metrics.items()})
        record["saved_model"] = str(model_path.relative_to(cfg.PROJECT_ROOT))
        comparison_records.append(record)
        test_summaries.append(test_metrics)

        print(
            f"  CV Macro-F1={cv_summary['cv_macro_f1_mean']:.4f} "
            f"(+/- {cv_summary['cv_macro_f1_std']:.4f}) | "
            f"Student Top-1={test_metrics['observation_top1_accuracy']:.4f} | "
            f"Student Macro-F1={test_metrics['macro_f1']:.4f}"
        )

    results = pd.DataFrame(comparison_records)
    results = results.sort_values("cv_macro_f1_mean", ascending=False, na_position="last")
    results_path = cfg.OUTPUT_DIR / "model_comparison.csv"
    results.to_csv(results_path, index=False)
    configuration_summary = {
        row["model"]: {
            "algorithm_family": row["algorithm_family"],
            "initial_configuration": json.loads(row["initial_configuration"]),
        }
        for _, row in results.iterrows()
    }
    with (cfg.OUTPUT_DIR / "model_configurations.json").open("w", encoding="utf-8") as f:
        json.dump(configuration_summary, f, indent=2, default=str)
    pd.concat(all_cv_folds, ignore_index=True).to_csv(
        cfg.OUTPUT_DIR / "cross_validation_fold_results.csv", index=False
    )
    make_comparison_plots(results, figure_dir)

    # The top model is selected from training-trip CV only. Student-test metrics
    # are reported for development comparison, not used by the HPO objective.
    best = results.iloc[0]
    summary = {
        "selected_by": "highest mean GroupKFold candidate-row Macro-F1 on training trips",
        "selected_model": str(best["model"]),
        "cv_macro_f1_mean": float(best["cv_macro_f1_mean"]),
        "cv_macro_f1_std": float(best["cv_macro_f1_std"]),
        "n_training_trips": int(train_df[cfg.TRIP_COLUMN].nunique()),
        "n_student_test_trips": int(student_df[cfg.TRIP_COLUMN].nunique()),
        "feature_columns": feature_cols,
        "random_state": cfg.RANDOM_STATE,
        "cv_splits": cfg.CV_SPLITS,
        "note": "Evaluator-testing data was not loaded or used.",
    }
    with (cfg.OUTPUT_DIR / "model_selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nModel comparison complete.")
    print(results[[
        "model",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "student_observation_top1_accuracy",
        "student_macro_f1",
        "student_log_loss",
        "student_route_continuity",
        "student_training_time_seconds",
        "student_inference_time_seconds",
    ]].to_string(index=False))
    print(f"\nSaved results: {results_path}")
    print(f"Selected (CV only): {summary['selected_model']}")
    print("Next: python run_bayesian_tuning.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
