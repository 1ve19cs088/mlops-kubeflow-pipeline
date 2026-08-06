# mlops-kubeflow-pipeline

A configuration-driven MLOps framework: ingestion → validation → feature
engineering → training → evaluation → FastAPI serving, designed to be
reusable across datasets and eventually orchestrated with Kubeflow
Pipelines.

## Building the serving image

The serving image bakes in a pre-trained model and its contract rather
than mounting them at deploy time (see `docker/Dockerfile.serving`), so
the following must be true **before** running `docker build`:

1. **The training pipeline must have been run at least once:**
   ```bash
   python -m src.pipeline.run_pipeline
   ```
2. **That run must have produced these three files** (the build will
   fail without them):
   - `model/model.pkl`
   - `artifacts/training_report.json`
   - `artifacts/metrics.json`
3. **The build must be run from the repository root**, not from
   inside `docker/` — this is what makes the Dockerfile's `COPY`
   instructions resolve correctly and what makes the root-level
   `.dockerignore` apply.

Build:
```bash
docker build -f docker/Dockerfile.serving -t mlops-kubeflow-pipeline-serving:latest .
```

Run:
```bash
docker run --rm -p 8000:8000 mlops-kubeflow-pipeline-serving:latest
```

Verify:
```bash
curl http://localhost:8000/v1/health
```

## Dashboard

A lightweight web dashboard (`dashboard/`) sits on top of the serving
API as a **separate process** — it never imports `app/` or `src/`,
only calls the existing `/v1/*` endpoints over HTTP:

```
Browser
   ↓
Dashboard UI (dashboard/, its own FastAPI app + Jinja2 templates)
   ↓  HTTP calls via dashboard/api_client.py
Existing FastAPI endpoints (app/, /v1/health, /v1/metadata, /v1/metrics, /v1/predict, /v1/predict/batch)
   ↓
Current model (model/model.pkl)
```

Every page duplicates zero business logic — it renders whatever the
existing API already returns. The prediction form is generated
entirely from `/v1/metadata`'s `feature_names`/`feature_dtypes`, so a
different dataset with a different number of features needs no
template changes. The Deployment page reads this repo's own
`kubernetes/*.yaml` manifests and safe local environment markers
(`/.dockerenv`, the Kubernetes serviceaccount path) rather than
querying a live cluster, consistent with this project's rule that
automated code never talks to any Kubernetes cluster except the local
`kind-ai-agent` one.

**Pages:** Home, Model Metrics, Prediction, Batch Prediction,
Deployment, System Status.

### Running it

Requires the serving API to be running first (see above), then:
```bash
pip install -r requirements/dashboard.txt
uvicorn dashboard.main:app --host 127.0.0.1 --port 8080
```
Open `http://127.0.0.1:8080`.

By default the dashboard calls the API at `http://localhost:8000`.
Override with:
```bash
export API_BASE_URL=http://localhost:8000
```

### Screenshots

_TODO: add screenshots of each page here (Home, Metrics, Predict, Batch Predict, Deployment, System Status)._
