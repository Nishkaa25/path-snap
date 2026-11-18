
"""Train remaining PathSnap models using the existing candidate dataset."""
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from src.ml.features import FEATURE_NAMES

DATA = Path("results/ml/candidate_training_data.csv")
OUT = Path("results/ml")
SEED = 42

def sample_observations(df, n, seed=42):
    keys = df[["trip_id", "observation_index"]].drop_duplicates()
    if len(keys) <= n:
        return df
    chosen = keys.sample(n=n, random_state=seed)
    return df.merge(chosen, on=["trip_id", "observation_index"], how="inner")

def top1_hit(df, scores):
    temp = df[["trip_id", "observation_index", "label"]].copy()
    temp["score"] = scores
    idx = temp.groupby(["trip_id", "observation_index"])["score"].idxmax()
    return float(temp.loc[idx, "label"].mean())

def evaluate(name, df, scores, pred):
    return {
        "model": name,
        "accuracy": accuracy_score(df.label, pred),
        "precision": precision_score(df.label, pred, zero_division=0),
        "recall": recall_score(df.label, pred, zero_division=0),
        "f1": f1_score(df.label, pred, zero_division=0),
        "top1_observation_hit": top1_hit(df, scores),
    }

def main():
    print("Loading existing dataset...", flush=True)
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} rows", flush=True)

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df.label.to_numpy(dtype=int)
    groups = df.trip_id.astype(str).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()

    results_path = OUT / "remaining_models_comparison.csv"
    results = []

    # ---------------- MLP ----------------
    print("\nPreparing MLP sample...", flush=True)
    mlp_train = sample_observations(train, 20000, SEED)
    X_mlp = mlp_train[FEATURE_NAMES].to_numpy(dtype=float)
    y_mlp = mlp_train.label.to_numpy(dtype=int)

    mlp = Pipeline([
        ("scale", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            random_state=SEED,
            verbose=True
        ))
    ])

    print(f"Training MLP on {len(mlp_train):,} rows...", flush=True)
    start = time.perf_counter()
    mlp.fit(X_mlp, y_mlp)
    elapsed = time.perf_counter() - start

    mlp_scores = mlp.predict_proba(val[FEATURE_NAMES].to_numpy(dtype=float))[:, 1]
    mlp_pred = (mlp_scores >= 0.5).astype(int)
    met = evaluate("MLP", val, mlp_scores, mlp_pred)
    met["fit_seconds"] = elapsed
    results.append(met)
    pd.DataFrame(results).to_csv(results_path, index=False)
    joblib.dump(mlp, OUT / "MLP_model.joblib")
    print(f"\nMLP F1={met['f1']:.4f}, Top-1={met['top1_observation_hit']:.4f}, time={elapsed:.1f}s", flush=True)

    # ---------------- SVM ----------------
    print("\nPreparing SVM sample...", flush=True)
    svm_train = sample_observations(train, 1500, SEED)
    svm_val = sample_observations(val, 1500, SEED)

    X_svm = svm_train[FEATURE_NAMES].to_numpy(dtype=float)
    y_svm = svm_train.label.to_numpy(dtype=int)
    X_svm_val = svm_val[FEATURE_NAMES].to_numpy(dtype=float)
    y_svm_val = svm_val.label.to_numpy(dtype=int)

    svm = Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(
            C=2.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            random_state=SEED
        ))
    ])

    print(f"Training SVM on {len(svm_train):,} rows...", flush=True)
    start = time.perf_counter()
    svm.fit(X_svm, y_svm)
    elapsed = time.perf_counter() - start

    svm_scores = svm.decision_function(X_svm_val)
    svm_pred = (svm_scores >= 0).astype(int)
    met = evaluate("SVM_RBF_sampled", svm_val, svm_scores, svm_pred)
    met["fit_seconds"] = elapsed
    results.append(met)
    pd.DataFrame(results).to_csv(results_path, index=False)
    joblib.dump(svm, OUT / "SVM_RBF_sampled_model.joblib")
    print(f"\nSVM F1={met['f1']:.4f}, Top-1={met['top1_observation_hit']:.4f}, time={elapsed:.1f}s", flush=True)

    print("\nRemaining-model results saved to:", results_path)
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
