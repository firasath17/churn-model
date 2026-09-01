"""
Gate #1: Data drift check.

Compares the incoming `current.csv` against the trusted `reference.csv`
using Evidently. If the input feature distributions have drifted too far,
this script exits non-zero, which fails the GitHub Actions step and stops
the pipeline before we waste compute training on bad data.

Outputs an HTML report (evidently_data_drift_report.html) that CI uploads
as a build artifact so a human can inspect exactly what drifted.
"""
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REFERENCE_PATH = "data/reference.csv"
CURRENT_PATH = "data/current.csv"
REPORT_PATH = "evidently_data_drift_report.html"

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "is_paperless",
]


def main() -> int:
    reference = pd.read_csv(REFERENCE_PATH)[FEATURE_COLUMNS]
    current = pd.read_csv(CURRENT_PATH)[FEATURE_COLUMNS]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(REPORT_PATH)

    result = report.as_dict()
    dataset_drift = result["metrics"][0]["result"]["dataset_drift"]
    n_drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]
    n_total = result["metrics"][0]["result"]["number_of_columns"]

    print(f"Drifted columns: {n_drifted}/{n_total}")
    print(f"Dataset-level drift detected: {dataset_drift}")
    print(f"Full report written to {REPORT_PATH}")

    if dataset_drift:
        print("FAILING: input data has drifted beyond threshold.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
