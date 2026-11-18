import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.pipeline import Bundle, predict_trip
from api.scoring import score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")
EVAL_DIR = Path(os.getenv("EVAL_DIR", "data/evaluator"))
state = {}


@asynccontextmanager
async def lifespan(app):
    state["bundle"] = Bundle()
    log.info("model + HMM config loaded")
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Req(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    if not EVAL_DIR.is_dir():
        return {"trips": []}
    return {"trips": sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir())}


@app.post("/predict")
def predict(req: Req, request: Request):
    if not re.fullmatch(r"\d{8}", req.trip_id) or not (EVAL_DIR / req.trip_id).is_dir():
        raise HTTPException(404, "unknown trip")                  # also blocks path traversal
    t0 = time.time()
    d = EVAL_DIR / req.trip_id
    log.info("REQUEST from=%s trip=%s", request.client.host, req.trip_id)
    trip, clf, hmm, tm = predict_trip(d, state["bundle"])
    metrics = score(d, clf, hmm, trip.arcs)                       # reference used ONLY here, after prediction
    seq = [a for i, a in enumerate(hmm) if a >= 0 and (i == 0 or a != hmm[i - 1])]
    # NOTE: assumes trip.nodes columns are (lon, lat); verify against project() with one real trip
    poly = [[float(trip.nodes[trip.arcs[a, 0], 1]), float(trip.nodes[trip.arcs[a, 0], 0])] for a in seq]
    if seq:
        last = seq[-1]
        poly.append([float(trip.nodes[trip.arcs[last, 1], 1]), float(trip.nodes[trip.arcs[last, 1], 0])])
    step = max(1, len(trip.lat) // 2000)
    tm["total_s"] = time.time() - t0
    log.info("DONE trip=%s obs=%d top1_clf=%.3f top1_hmm=%.3f cont_hmm=%.3f time=%.1fs",
             req.trip_id, len(trip.P), metrics["top1_classifier"], metrics["top1_hmm"],
             metrics["continuity_hmm"], tm["total_s"])
    return dict(trip_id=req.trip_id, metrics=metrics, timing=tm,
                center=[float(trip.lat.mean()), float(trip.lon.mean())],
                gps=[[float(a), float(b)] for a, b in zip(trip.lat[::step], trip.lon[::step])],
                pred_polyline=poly)
