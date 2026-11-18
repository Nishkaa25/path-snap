# PathSnap

Map-matching of GPS traces to road arcs: candidate generation, a classifier scoring each candidate,
and an HMM/Viterbi decoder that enforces road-network connectivity. Served through a FastAPI service
with a Streamlit UI.

## Repo layout

```
src/common.py        load_trip, project, point_segment
src/data/            labels.py (align_labels), eda.py
src/features/        candidates.py, features.py
src/hmm/             graph.py (CSR road graph), viterbi.py (decoder)
src/eval/            metrics.py
api/                 main.py (FastAPI), pipeline.py (prediction), scoring.py (reference scoring)
artifacts/           model.joblib, config.json   (config.json holds R, K and the "hmm" settings)
results/             eda/, oof_best.parquet, hmm_tuning.csv
tests/               test_viterbi.py
```

## Data placement (never committed)

```
data/train/<trip_id>/
data/student_test/<trip_id>/
data/evaluator/<trip_id>/      # four files per trip; path configurable with EVAL_DIR
```

## Install

```
pip install -r requirements.txt
```

## Reproduce

TODO: EDA, build dataset, cross-validation, Bayesian optimisation, decoder tuning (fill in commands).

## Run the API

```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Set `EVAL_DIR` to the evaluator data folder. `PATHSNAP_STUB=1` runs with a stub model (no artifacts needed).
Endpoints: `GET /health`, `GET /trips`, `POST /predict` with body `{"trip_id": "00000076"}`.

## Run the UI

```
API_URL=http://<api-laptop-ip>:8000 streamlit run ui/app.py
```

## Two-laptop deployment

1. Both laptops on the same network (use a phone hotspot or Ethernet if campus Wi-Fi isolates clients).
2. Laptop 1: find its IP (`ipconfig` on Windows, `ip a` on Linux/Mac).
3. Windows firewall: `netsh advfirewall firewall add rule name=PathSnap dir=in action=allow protocol=TCP localport=8000`
4. Laptop 2: open `http://<ip>:8000/health` in a browser first.
5. Start the UI with `API_URL=http://<ip>:8000`.
6. Try all ten evaluator trips and keep the API terminal (REQUEST / DONE log lines) visible.

## Integrity

The prediction path loads trips with `with_route=False`. The reference route is read only in
`api/scoring.py`, after prediction, and only numbers are returned to the UI.
