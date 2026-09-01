# churn-model — GitHub Actions + MLflow + Evidently CI/CD demo

A minimal but complete ML pipeline showing how to wire together:

- **GitHub Actions** — orchestrates every step, and blocks the pipeline when a check fails
- **MLflow** — tracks every training run (params, metrics, model artifact) and manages promotion via the Model Registry
- **Evidently** — gates the pipeline in two places: before training (input data drift) and after training (prediction drift / model quality)

```
                ┌─────────────────┐
   push/PR ───► │ validate-data    │  Evidently: is incoming data
                │ (Evidently)      │  too different from reference?
                └────────┬─────────┘
                         │ pass
                         ▼
                ┌─────────────────┐
                │ train-and-       │  sklearn model, logged to MLflow.
                │ validate-        │  Evidently: has the model's OWN
                │ predictions      │  prediction behavior drifted vs.
                └────────┬─────────┘  a stored baseline of predictions?
                         │ pass (main branch only)
                         ▼
                ┌─────────────────┐
                │ register-model   │  MLflow Model Registry: register,
                │                  │  promote to Production only if it
                └────────┬─────────┘  beats the current Production f1
                         │
                         ▼
                ┌─────────────────┐
                │ deploy (mock)    │  stands in for docker build/push +
                └──────────────────┘  rollout
```

## Why two Evidently checks?

- **`validate_data.py`** (before training) catches *input* drift — e.g. a new
  cohort of customers, a broken upstream ETL job, a schema change. Failing
  fast here avoids wasting compute training on bad data.
- **`validate_predictions.py`** (after training) catches *model behavior*
  drift — e.g. the model's predicted churn rate suddenly jumps even though
  the inputs look fine. This can happen from a code bug, a bad label join,
  or a regression introduced by a "improvement." Data drift and prediction
  drift are not redundant — each catches failure modes the other misses.

## Repo layout

```
churn-model/
├── data/
│   ├── reference.csv              # baseline distribution (checked in)
│   ├── current.csv                # "new" incoming data (checked in; regenerate to test drift)
│   └── predictions_baseline.csv   # created automatically on first run
├── src/
│   ├── generate_data.py           # synthetic data generator (makes this repo runnable standalone)
│   ├── validate_data.py           # Evidently gate #1: input data drift
│   ├── train.py                   # trains model, logs to MLflow, writes predictions.csv
│   ├── validate_predictions.py    # Evidently gate #2: prediction/target drift
│   └── register_model.py          # MLflow Model Registry: register + conditionally promote
├── requirements.txt
└── .github/workflows/ci-cd.yml
```

## Running it locally

```bash
pip install -r requirements.txt

# 1. (data/*.csv is already checked in, but you can regenerate it)
python src/generate_data.py

# 2. Gate 1: check input data drift
python src/validate_data.py
# -> exits 1 and writes evidently_data_drift_report.html if drift is detected

# 3. Train + log to MLflow
python src/train.py
# -> writes predictions.csv, run_id.txt, and a run under ./mlruns/

# 4. Gate 2: check prediction drift vs. the stored baseline
python src/validate_predictions.py
# -> first run creates data/predictions_baseline.csv and passes;
#    subsequent runs compare against it

# 5. Register (and maybe promote) the model
python src/register_model.py

# Inspect everything in the MLflow UI:
mlflow ui --backend-store-uri file:./mlruns
```

## Simulating a failure on purpose

To see the pipeline actually block a bad deploy, regenerate `current.csv`
with simulated drift and re-run:

```bash
DRIFT=1 python src/generate_data.py
python src/validate_data.py   # now exits 1 — pipeline stops here
```

In GitHub Actions, trigger this the same way via the manual
`workflow_dispatch` input `simulate_drift: true` on the Actions tab.

## What's mocked vs. real

- **Real**: data generation, drift detection math, model training, MLflow
  tracking/registry, GitHub Actions orchestration and artifact upload — all
  of this actually runs and produces real reports/metrics.
- **Mocked**: the final `deploy` job just prints the commands it *would* run
  (`docker build`, `docker push`, `kubectl rollout restart`) rather than
  actually shipping to infrastructure, since that part is entirely specific
  to wherever you deploy models.

## Adapting this to a real project

- Point `MLFLOW_TRACKING_URI` at a shared MLflow server (not `file:./mlruns`)
  so runs persist across CI jobs instead of living only on the ephemeral
  GitHub Actions runner.
- Store `data/reference.csv` and the prediction baseline somewhere durable
  (S3/GCS/a feature store) rather than committing CSVs to git, once your
  data stops being toy-sized.
- Add secrets (`MLFLOW_TRACKING_URI`, cloud credentials) via GitHub encrypted
  secrets, referenced as `${{ secrets.MLFLOW_TRACKING_URI }}` in the workflow.
- Replace the mock `deploy` job with your actual build/push/rollout steps.
