from __future__ import annotations

from typing import Any


def _custom_suggestions(trial, spec: dict) -> dict:
    params = {}
    for name, definition in spec.items():
        kind = definition[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(definition[1]), int(definition[2]))
        elif kind == "float":
            log_scale = len(definition) > 3 and definition[3] == "log"
            params[name] = trial.suggest_float(
                name, float(definition[1]), float(definition[2]), log=log_scale
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(definition[1]))
        else:
            raise ValueError(f"Unknown custom search-space type {kind!r} for {name!r}.")
    return params


def suggest_params(trial, model_name: str, cfg: Any) -> dict:
    """Bayesian search spaces for package baselines or user-defined estimators."""
    if model_name in cfg.CUSTOM_SEARCH_SPACES:
        return _custom_suggestions(trial, cfg.CUSTOM_SEARCH_SPACES[model_name])

    if model_name == "Logistic Regression":
        return {
            "model__C": trial.suggest_float("model__C", 1e-3, 100.0, log=True),
            "model__class_weight": trial.suggest_categorical(
                "model__class_weight", [None, "balanced"]
            ),
        }
    if model_name == "K-Nearest Neighbours":
        return {
            "model__n_neighbors": trial.suggest_int("model__n_neighbors", 3, 25),
            "model__weights": trial.suggest_categorical("model__weights", ["uniform", "distance"]),
            "model__p": trial.suggest_int("model__p", 1, 2),
        }
    if model_name == "Decision Tree":
        return {
            "model__max_depth": trial.suggest_categorical(
                "model__max_depth", [None, 4, 6, 8, 12, 16, 24, 32]
            ),
            "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 25),
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 12),
            "model__criterion": trial.suggest_categorical(
                "model__criterion", ["gini", "entropy", "log_loss"]
            ),
        }
    if model_name == "Random Forest":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 150, 600, step=50),
            "model__max_depth": trial.suggest_categorical(
                "model__max_depth", [None, 8, 12, 16, 24, 32]
            ),
            "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 10),
            "model__max_features": trial.suggest_categorical(
                "model__max_features", ["sqrt", "log2", 0.7]
            ),
            "model__class_weight": trial.suggest_categorical(
                "model__class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
    if model_name == "Extra Trees":
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 150, 600, step=50),
            "model__max_depth": trial.suggest_categorical(
                "model__max_depth", [None, 8, 12, 16, 24, 32]
            ),
            "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 20),
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 10),
            "model__max_features": trial.suggest_categorical(
                "model__max_features", ["sqrt", "log2", 0.7]
            ),
        }
    if model_name == "HistGradientBoosting":
        return {
            "model__learning_rate": trial.suggest_float("model__learning_rate", 0.02, 0.25, log=True),
            "model__max_iter": trial.suggest_int("model__max_iter", 100, 500, step=50),
            "model__max_leaf_nodes": trial.suggest_int("model__max_leaf_nodes", 15, 63),
            "model__l2_regularization": trial.suggest_float(
                "model__l2_regularization", 1e-6, 10.0, log=True
            ),
        }

    raise ValueError(
        f"No Bayesian search space is configured for {model_name!r}. Add it to "
        "CUSTOM_SEARCH_SPACES in config.py."
    )
