from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone

from . import config as cfg
from .pathsnap_eval.cross_validation import grouped_cross_validation
from .pathsnap_eval.data import load_eval_data
from .pathsnap_eval.evaluator import fit_and_evaluate
from .pathsnap_eval.models import get_model_registry
from .pathsnap_eval.plots import save_confusion_matrix
from .pathsnap_eval.tuning import suggest_params


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def main() -> int:
    comparison_path = cfg.OUTPUT_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"{comparison_path} not found. Run python run_model_comparison.py first."
        )

    train_df, student_df, feature_cols = load_eval_data(cfg)
    registry = get_model_registry(cfg)
    comparison = pd.read_csv(comparison_path)

    if cfg.TUNING_MODEL_NAME:
        model_name = cfg.TUNING_MODEL_NAME
    else:
        # Select solely by CV score on the 80 training trips.
        best_row = comparison.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
        model_name = str(best_row["model"])

    if model_name not in registry:
        raise KeyError(f"TUNING_MODEL_NAME={model_name!r} is not in the model registry.")
    base_estimator = registry[model_name]
    print(f"Bayesian optimisation target: {model_name}")
    print("Objective: mean candidate-row Macro-F1 across GroupKFold on training trips.")
    print("The student_test partition is not passed to the objective.")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tune_dir = cfg.OUTPUT_DIR / "bayesian_optimisation"
    tune_dir.mkdir(parents=True, exist_ok=True)
    storage_path = tune_dir / cfg.OPTUNA_STORAGE_FILENAME
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study_name = f"{cfg.OPTUNA_STUDY_NAME_PREFIX}_{_safe_name(model_name)}_seed{cfg.RANDOM_STATE}"

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name, cfg)
        fold_summary, fold_results = grouped_cross_validation(
            model_name,
            base_estimator,
            train_df,
            feature_cols,
            cfg,
            params=params,
        )
        trial.set_user_attr("cv_macro_f1_std", fold_summary["cv_macro_f1_std"])
        trial.set_user_attr("cv_top1_accuracy_mean", fold_summary["cv_top1_accuracy_mean"])
        trial.set_user_attr("params_used", params)
        return float(fold_summary["cv_macro_f1_mean"])

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_STATE),
        storage=storage_url,
        load_if_exists=True,
    )
    before = len(study.trials)
    study.optimize(objective, n_trials=int(cfg.N_BAYESIAN_TRIALS), show_progress_bar=False)
    added = len(study.trials) - before

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    trials_df.to_csv(tune_dir / "bayesian_trials.csv", index=False)

    best_params = dict(study.best_trial.params)
    tuned_estimator = clone(base_estimator)
    tuned_estimator.set_params(**best_params)

    # Refit the tuned estimator on all 80 permitted training trips, then use
    # the ten student-testing trips once for development-stage evaluation.
    test_metrics, test_predictions, cm, tuned_estimator = fit_and_evaluate(
        tuned_estimator,
        train_df,
        student_df,
        feature_cols,
        cfg,
        cfg.STUDENT_TEST_PARTITION,
    )
    best_cv_summary, best_cv_folds = grouped_cross_validation(
        model_name,
        base_estimator,
        train_df,
        feature_cols,
        cfg,
        params=best_params,
    )
    best_cv_folds.to_csv(tune_dir / "best_parameters_cv_folds.csv", index=False)

    model_path = tune_dir / f"tuned_{_safe_name(model_name)}.joblib"
    joblib.dump(
        {
            "model": tuned_estimator,
            "feature_columns": feature_cols,
            "model_name": model_name,
            "best_params": best_params,
        },
        model_path,
    )
    test_predictions.to_csv(tune_dir / "tuned_student_test_predictions.csv", index=False)
    save_confusion_matrix(cm, f"Tuned {model_name}", tune_dir)

    baseline_rows = comparison[comparison["model"] == model_name]
    if baseline_rows.empty:
        raise ValueError(f"Base comparison result for {model_name!r} is missing.")
    baseline = baseline_rows.iloc[0]

    compare_metrics = [
        ("CV Macro-F1", float(baseline["cv_macro_f1_mean"]), float(best_cv_summary["cv_macro_f1_mean"])),
        ("Student-test Top-1 Accuracy", float(baseline["student_observation_top1_accuracy"]), float(test_metrics["observation_top1_accuracy"])),
        ("Student-test Macro-F1", float(baseline["student_macro_f1"]), float(test_metrics["macro_f1"])),
        ("Student-test Log Loss", float(baseline["student_log_loss"]), float(test_metrics["log_loss"])),
        ("Student-test Inference Time (s)", float(baseline["student_inference_time_seconds"]), float(test_metrics["inference_time_seconds"])),
        ("Student-test Route Continuity", float(baseline["student_route_continuity"]), float(test_metrics["route_continuity"])),
    ]
    compare_df = pd.DataFrame(compare_metrics, columns=["metric", "untuned", "tuned"])
    compare_df.to_csv(tune_dir / "tuned_vs_untuned_comparison.csv", index=False)

    result = {
        "model": model_name,
        "optimiser": "Optuna TPE sampler (Bayesian optimisation)",
        "objective": "Maximise mean candidate-row Macro-F1 over GroupKFold on training trips",
        "trials_added_this_run": int(added),
        "total_trials_in_study": int(len(study.trials)),
        "best_trial_number": int(study.best_trial.number),
        "best_cv_macro_f1": float(study.best_value),
        "best_cv_macro_f1_std": float(best_cv_summary["cv_macro_f1_std"]),
        "best_params": best_params,
        "student_test_metrics": test_metrics,
        "saved_model": str(model_path),
        "trial_database": str(storage_path),
        "note": "Evaluator-testing data was not loaded or used.",
    }
    with (tune_dir / "best_tuning_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)

    print("\nBayesian optimisation complete.")
    print(f"Trials added this run: {added}; total retained trials: {len(study.trials)}")
    print(f"Best CV Macro-F1: {study.best_value:.4f}")
    print("Best parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    print("\nUntuned vs tuned:")
    print(compare_df.to_string(index=False))
    print(f"\nSaved tuning record: {tune_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
