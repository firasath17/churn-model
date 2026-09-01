"""
Trains the churn model and logs everything (params, metrics, model,
predictions) to MLflow.

Writes predictions.csv, which validate_predictions.py uses for the
second Evidently gate (prediction drift / target drift), and writes
run_id.txt so downstream CI steps know which MLflow run to register.
"""
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "churn-prediction"
F1_THRESHOLD = 0.45  # quality gate — build fails if the new model doesn't clear this

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "is_paperless",
]


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv("data/current.csv")
    X = df[FEATURE_COLUMNS]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {"n_estimators": 200, "max_depth": 6, "random_state": 42}

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("training_rows", len(X_train))

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # Save predictions on the held-out test set for the prediction-drift
        # check. This mirrors what you'd log in production: features +
        # ground truth (when available) + model output, all side by side.
        pred_df = X_test.copy()
        pred_df["churn_actual"] = y_test.values
        pred_df["churn_predicted"] = preds
        pred_df["churn_probability"] = pred_probs
        pred_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"run_id={run.info.run_id}")
        print(json.dumps(metrics, indent=2))

        if metrics["f1_score"] < F1_THRESHOLD:
            raise SystemExit(
                f"Model f1_score {metrics['f1_score']:.3f} is below "
                f"threshold {F1_THRESHOLD}. Failing build."
            )


if __name__ == "__main__":
    main()
