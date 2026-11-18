"""Evaluate saved model on a held-out trip partition. Use this only once model selection is done."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.ml.dataset import build_trip_dataset
from src.ml.train import _metrics, top1_observation_hit


def evaluate_partition(root, bundle_path, out="results/ml/student_test"):
    bundle = joblib.load(bundle_path); model = bundle["model"]
    outdir = Path(out); outdir.mkdir(parents=True, exist_ok=True)
    frames=[]
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir(): continue
        df, summary = build_trip_dataset(d, radius_m=bundle.get("radius_m",50),
            max_candidates=bundle.get("candidate_cap",30), max_label_distance_m=1e9, gap_break_s=30)
        if len(df):
            frames.append(df); print(f"[{d.name}] candidates={len(df)}", flush=True)
    if not frames: raise RuntimeError("No evaluation candidate rows built")
    data=pd.concat(frames, ignore_index=True)
    X=data[bundle["feature_names"]].to_numpy(float); y=data.label.to_numpy(int)
    prob=model.predict_proba(X)[:,1]; pred=(prob>=0.5).astype(int)
    result=_metrics(y,pred,prob); result["top1_observation_hit"]=top1_observation_hit(data,prob)
    pd.DataFrame([result]).to_csv(outdir/"metrics.csv",index=False)
    data["score"]=prob; data.to_csv(outdir/"candidate_scores.csv",index=False)
    print(pd.Series(result).to_string())
    return result

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="data/student_test"); ap.add_argument("--model",default="results/ml/best_model.joblib"); ap.add_argument("--out",default="results/ml/student_test")
    a=ap.parse_args(); evaluate_partition(a.root,a.model,a.out)
