from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _pipeline(classifier, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", classifier))
    return Pipeline(steps)


def default_model_registry(random_state: int = 42) -> dict[str, Any]:
    """Six supervised classifiers with probability estimates."""
    return {
        "Logistic Regression": _pipeline(
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                solver="lbfgs",
                random_state=random_state,
            ),
            scale=True,
        ),
        "K-Nearest Neighbours": _pipeline(
            KNeighborsClassifier(n_neighbors=7, weights="distance", p=2),
            scale=True,
        ),
        "Decision Tree": _pipeline(
            DecisionTreeClassifier(
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=random_state,
            )
        ),
        "Random Forest": _pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "Extra Trees": _pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "HistGradientBoosting": _pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                l2_regularization=0.0,
                random_state=random_state,
            )
        ),
    }


def _load_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Custom model source not found: {path}")
    spec = importlib.util.spec_from_file_location(f"pathsnap_models_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom model source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_model_registry(cfg: Any) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    if cfg.USE_DEFAULT_BASELINES:
        registry.update(default_model_registry(cfg.RANDOM_STATE))

    for source_path, function_name in cfg.CUSTOM_MODEL_EXPORTS:
        path = Path(source_path)
        module = _load_module(path)
        if not hasattr(module, function_name):
            raise AttributeError(
                f"{path} must define {function_name}() returning a dict of "
                "unfitted scikit-learn estimators. See README_FIRST.md."
            )
        custom = getattr(module, function_name)()
        if not isinstance(custom, dict) or not custom:
            raise TypeError(f"{path}:{function_name}() must return a non-empty dict.")
        for name, estimator in custom.items():
            if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
                raise TypeError(
                    f"Custom model {name!r} is not a scikit-learn-style estimator. "
                    "It must implement fit() and predict()."
                )
            registry[str(name)] = estimator

    if len(registry) < 5:
        raise ValueError(
            f"Only {len(registry)} models are registered. PathSnap requires at least five. "
            "Enable default baselines or export additional models from train.py and "
            "training_remainig.py."
        )
    return registry
