"""
Gate #2: Prediction / target drift check.

This runs AFTER training, using predictions.csv written by train.py.
Where validate_data.py checks whether the *inputs* have drifted,
this checks whether the *model's behavior* has drifted — comparing
the current run's predicted churn probabilities and error patterns
against a stored baseline of predictions from a known-good run.

Why both checks matter:
  - Data drift can happen with no prediction drift (irrelevant feature shifted).
  - Prediction drift can happen with no data drift (the model itself regressed,
    e.g. after a code change or retraining on a bad label set).

On the very first run there's no baseline yet, so this script creates one
and passes. On subsequent runs it compares against that stored baseline.
"""
import os
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset, ClassificationPreset

PREDICTIONS_PATH = "predictions.csv"
BASELINE_PATH = "data/predictions_baseline.csv"
REPORT_PATH = "evidently_prediction_drift_report.html"


def main() -> int:
    current = pd.read_csv(PREDICTIONS_PATH)
    current = current.rename(
        columns={"churn_actual": "target", "churn_predicted": "prediction"}
    )

    if not os.path.exists(BASELINE_PATH):
        # First-ever run: nothing to compare against yet. Establish this
        # run's predictions as the baseline and let the pipeline proceed.
        current.to_csv(BASELINE_PATH, index=False)
        print(f"No baseline found. Wrote {BASELINE_PATH} as the new baseline. Passing.")
        return 0

    reference = pd.read_csv(BASELINE_PATH)

    report = Report(metrics=[TargetDriftPreset(), ClassificationPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(REPORT_PATH)

    result = report.as_dict()
    # TargetDriftPreset's first metric reports drift on the `prediction` column
    prediction_drift = result["metrics"][0]["result"].get("drift_detected", False)

    # Pull current-run classification metrics for a plain-language print,
    # so a human reading CI logs doesn't have to open the HTML report
    # just to see if quality moved.
    classification_metrics = next(
        (m["result"]["current"] for m in result["metrics"] if "current" in m.get("result", {})),
        None,
    )
    if classification_metrics:
        print("Current run classification metrics:", classification_metrics)

    print(f"Prediction drift detected: {prediction_drift}")
    print(f"Full report written to {REPORT_PATH}")

    if prediction_drift:
        print(
            "FAILING: model's prediction distribution has drifted vs. baseline.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
