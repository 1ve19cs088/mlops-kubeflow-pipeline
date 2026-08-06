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
