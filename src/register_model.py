"""
Registers the model from the current run into the MLflow Model Registry,
but only if its f1_score beats the current Production model's f1_score
(or if there is no Production model yet).

This is the final gate: even after passing data drift + prediction drift +
the raw quality threshold in train.py, a model still has to beat what's
already deployed to get promoted.
"""
import sys
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-model"


def get_production_f1(client: MlflowClient) -> float | None:
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    except Exception:
        return None
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get("f1_score")


def main() -> int:
    with open("run_id.txt") as f:
        run_id = f.read().strip()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    run = client.get_run(run_id)
    new_f1 = run.data.metrics["f1_score"]
    prod_f1 = get_production_f1(client)

    print(f"New candidate f1_score: {new_f1:.4f}")
    print(f"Current Production f1_score: {prod_f1}")

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {mv.version} (stage: None)")

    if prod_f1 is None or new_f1 > prod_f1:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Promoted version {mv.version} to Production.")
    else:
        print(
            f"Version {mv.version} registered but NOT promoted "
            f"(f1 {new_f1:.4f} did not beat Production f1 {prod_f1:.4f})."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
