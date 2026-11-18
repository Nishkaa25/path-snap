"""
PathSnap final metrics evaluation.
Uses the existing dataset; all models share a GroupShuffleSplit validation set.
MLP and SVM use observation-level training samples for runtime practicality.
"""
from pathlib import Path
import time, json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from src.ml.features import FEATURE_NAMES

DATA=Path("results/ml/candidate_training_data.csv")
OUT=Path("results/ml/final_metrics")
SEED=42

def sample_obs(frame, n, seed=SEED):
    keys=frame[["trip_id","observation_index"]].drop_duplicates()
    if len(keys)<=n: return frame.copy()
    chosen=keys.sample(n=n, random_state=seed)
    return frame.merge(chosen, on=["trip_id","observation_index"], how="inner")

def top1(frame, scores):
    tmp=frame[["trip_id","observation_index","label"]].copy()
    tmp["score"]=scores
    winners=tmp.groupby(["trip_id","observation_index"])["score"].idxmax()
    return float(tmp.loc[winners,"label"].mean())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA)
    missing=[f for f in FEATURE_NAMES if f not in df.columns]
    if missing: raise ValueError(f"Missing feature columns: {missing}")
    y=df.label.astype(int).to_numpy()
    groups=df.trip_id.astype(str).to_numpy()
    splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=SEED)
    train_idx,val_idx=next(splitter.split(df[FEATURE_NAMES],y,groups))
    train=df.iloc[train_idx].reset_index(drop=True)
    val=df.iloc[val_idx].reset_index(drop=True)
    Xv=val[FEATURE_NAMES].to_numpy(float); yv=val.label.astype(int).to_numpy()
    print(f"Train {len(train):,} rows/{train.trip_id.nunique()} trips; validation {len(val):,} rows/{val.trip_id.nunique()} trips")

    specs=[
      ("LogisticRegression",Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=SEED))]),train),
      ("RandomForest",RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED),train),
      ("HistGradientBoosting",HistGradientBoostingClassifier(max_iter=200,learning_rate=0.08,max_leaf_nodes=31,random_state=SEED),train),
      ("SVM_RBF_sampled",Pipeline([("scale",StandardScaler()),("model",SVC(C=2.0,gamma="scale",kernel="rbf",probability=True,class_weight="balanced",random_state=SEED))]),sample_obs(train,1500)),
      ("MLP_sampled",Pipeline([("scale",StandardScaler()),("model",MLPClassifier(hidden_layer_sizes=(64,32),alpha=0.001,learning_rate_init=0.001,max_iter=400,early_stopping=True,random_state=SEED))]),sample_obs(train,20000)),
    ]
    rows=[]
    for name,model,subtrain in specs:
        Xt=subtrain[FEATURE_NAMES].to_numpy(float); yt=subtrain.label.astype(int).to_numpy()
        print(f"\\nTraining {name}: {len(Xt):,} candidate rows...")
        t=time.perf_counter(); model.fit(Xt,yt); fit=time.perf_counter()-t
        t=time.perf_counter(); prob=model.predict_proba(Xv)[:,1]; predtime=time.perf_counter()-t
        pred=(prob>=0.5).astype(int)
        tn,fp,fn,tp=confusion_matrix(yv,pred,labels=[0,1]).ravel()
        r={"model":name,"train_rows":len(Xt),"validation_rows":len(yv),
           "accuracy":accuracy_score(yv,pred),"precision":precision_score(yv,pred,zero_division=0),
           "recall":recall_score(yv,pred,zero_division=0),"f1":f1_score(yv,pred,zero_division=0),
           "roc_auc":roc_auc_score(yv,prob),"log_loss":log_loss(yv,prob,labels=[0,1]),
           "top1_observation_hit":top1(val,prob),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
           "fit_seconds":fit,"predict_seconds_full_validation":predtime}
        rows.append(r)
        pd.DataFrame([r]).to_csv(OUT/f"{name}_metrics.csv",index=False)
        print(f"Accuracy {r['accuracy']:.4f} | Precision {r['precision']:.4f} | Recall {r['recall']:.4f} | F1 {r['f1']:.4f} | Top-1 {r['top1_observation_hit']:.4f} | {fit:.1f}s")
    result=pd.DataFrame(rows).sort_values(["top1_observation_hit","f1"],ascending=False)
    result.to_csv(OUT/"all_models_metrics.csv",index=False)
    result.to_json(OUT/"all_models_metrics.json",orient="records",indent=2)
    info={"seed":SEED,"split":"GroupShuffleSplit 80/20 by trip_id","train_trip_ids":sorted(train.trip_id.astype(str).unique().tolist()),
           "validation_trip_ids":sorted(val.trip_id.astype(str).unique().tolist()),"features":FEATURE_NAMES,
           "note":"All models use the same complete validation set; MLP/SVM training uses observation-level samples."}
    (OUT/"split_and_methodology.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print("\\nFINAL TABLE\\n",result.to_string(index=False))
    print("Saved in",OUT)

if __name__=="__main__": main()
