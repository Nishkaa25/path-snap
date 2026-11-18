"""Train and compare five candidate classifiers with trip-grouped holdout."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.ml.features import FEATURE_NAMES
from src.ml.dataset import build_training_dataset


def models(seed=42):
    return {
        "LogisticRegression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=seed),
        "SVM_RBF": Pipeline([("scale", StandardScaler()), ("model", SVC(C=2.0, gamma="scale", kernel="rbf", probability=True, class_weight="balanced", random_state=seed))]),
        "MLP": Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, learning_rate_init=0.001, max_iter=400, early_stopping=True, random_state=seed))]),
    }


def _metrics(y, pred, prob):
    out = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "log_loss": log_loss(y, prob, labels=[0, 1])}
    try: out["roc_auc"] = roc_auc_score(y, prob)
    except ValueError: out["roc_auc"] = float("nan")
    return out


def top1_observation_hit(df, scores):
    tmp = df[["trip_id", "observation_index", "label"]].copy(); tmp["score"] = scores
    idx = tmp.groupby(["trip_id", "observation_index"])["score"].idxmax()
    return float(tmp.loc[idx, "label"].mean())


def train_and_compare(df, out_dir="results/ml", seed=42, tune=False):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X = df[FEATURE_NAMES].to_numpy(dtype=float); y = df.label.to_numpy(int); groups = df.trip_id.astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    results, fitted = [], {}
    search_spaces = {
        "LogisticRegression": {"model__C": [0.05, 0.1, 0.5, 1.0, 3.0, 10.0]},
        "RandomForest": {"n_estimators": [150, 250, 400], "max_depth": [None, 10, 20, 35], "min_samples_leaf": [1, 2, 4]},
        "HistGradientBoosting": {"learning_rate": [0.03, 0.05, 0.08, 0.1], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 0.1, 1.0]},
        "SVM_RBF": {"model__C": [0.5, 1.0, 2.0, 5.0, 10.0], "model__gamma": ["scale", 0.001, 0.01, 0.1]},
        "MLP": {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "model__alpha": [0.0001, 0.001, 0.01]},
    }
    train_groups = groups[train_idx]
    cv_splits = min(3, len(set(train_groups)))
    for name, model in models(seed).items():
        start = time.perf_counter()
        if tune and cv_splits >= 2:
            cv = GroupKFold(n_splits=cv_splits)
            search = RandomizedSearchCV(model, search_spaces[name], n_iter=5, scoring="f1",
                cv=cv, random_state=seed, n_jobs=-1, refit=True, verbose=0)
            search.fit(X[train_idx], y[train_idx], groups=train_groups)
            model = search.best_estimator_
            with open(out / f"{name}_tuning.json", "w", encoding="utf-8") as f:
                json.dump({"best_params": search.best_params_, "best_cv_f1": float(search.best_score_)}, f, indent=2)
        else:
            model.fit(X[train_idx], y[train_idx])
        elapsed = time.perf_counter() - start
        prob = model.predict_proba(X[val_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        met = _metrics(y[val_idx], pred, prob)
        met["top1_observation_hit"] = top1_observation_hit(df.iloc[val_idx].reset_index(drop=True), prob)
        met["fit_seconds"] = elapsed; met["model"] = name
        results.append(met); fitted[name] = model
        print(f"{name}: F1={met['f1']:.4f}, top1={met['top1_observation_hit']:.4f}, fit={elapsed:.1f}s", flush=True)
    table = pd.DataFrame(results).sort_values(["top1_observation_hit", "f1"], ascending=False)
    table.to_csv(out / "model_comparison.csv", index=False)
    table.to_json(out / "model_comparison.json", orient="records", indent=2)
    best_name = table.iloc[0]["model"]
    joblib.dump({"model": fitted[best_name], "model_name": best_name, "feature_names": FEATURE_NAMES,
                "radius_m": 50.0, "candidate_cap": 30}, out / "best_model.joblib")
    with open(out / "validation_trip_ids.json", "w", encoding="utf-8") as f:
        json.dump({"train_trip_ids": sorted(set(groups[train_idx])), "validation_trip_ids": sorted(set(groups[val_idx])),
                   "split_seed": seed}, f, indent=2)
    print(f"Best model by observation Top-1 then candidate F1: {best_name}")
    return table, fitted[best_name], best_name



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default="data/train")
    ap.add_argument("--out", default="results/ml")
    ap.add_argument("--exclude", nargs="*", type=int, default=[33, 54])
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--max-candidates", type=int, default=30)
    ap.add_argument("--max-label-distance", type=float, default=50.0)
    ap.add_argument("--gap-break", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune", action="store_true", help="Run grouped randomized hyperparameter search (slower)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df, summaries = build_training_dataset(args.train_root, tuple(args.exclude), args.radius,
                                          args.max_candidates, args.max_label_distance, args.gap_break)
    df.to_csv(out / "candidate_training_data.csv", index=False)
    summaries.to_csv(out / "dataset_build_summary.csv", index=False)
    print(f"Candidate dataset: {len(df):,} rows; positives={int(df.label.sum()):,}; trips={df.trip_id.nunique()}")
    train_and_compare(df, out, args.seed, tune=args.tune)

if __name__ == "__main__":
    main()
