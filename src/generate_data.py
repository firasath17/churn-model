"""
Generates synthetic churn data so this repo is runnable end-to-end
without needing a real dataset.

Run this once to create data/reference.csv (the stable baseline).
Run it again with DRIFT=1 to simulate a shift in the incoming data,
which is what should trip the Evidently drift check in CI.

Usage:
    python src/generate_data.py                 # writes reference + non-drifted current
    DRIFT=1 python src/generate_data.py          # writes a drifted current.csv only
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")


def make_data(n: int, drift: bool = False, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)

    tenure_months = r.normal(24 if not drift else 14, 10, n).clip(0, 72)
    monthly_charges = r.normal(65 if not drift else 85, 20, n).clip(10, 150)
    support_calls = r.poisson(1.5 if not drift else 3.2, n)
    contract_type = r.choice(
        [0, 1, 2], size=n, p=[0.5, 0.3, 0.2] if not drift else [0.3, 0.3, 0.4]
    )
    is_paperless = r.integers(0, 2, n)

    # Ground-truth relationship used to label churn (kept stable even under
    # covariate drift, so drift here is a pure data-distribution shift,
    # not a change in the underlying label-generating process).
    logit = (
        -2.5
        - 0.08 * tenure_months
        + 0.035 * monthly_charges
        + 0.9 * support_calls
        - 0.7 * contract_type
        + 0.3 * is_paperless
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = r.binomial(1, prob)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months.round(1),
            "monthly_charges": monthly_charges.round(2),
            "support_calls": support_calls,
            "contract_type": contract_type,
            "is_paperless": is_paperless,
            "churn": churn,
        }
    )


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    drift = os.environ.get("DRIFT", "0") == "1"

    if drift:
        current = make_data(500, drift=True, seed=99)
        current.to_csv(os.path.join(DATA_DIR, "current.csv"), index=False)
        print(f"Wrote DRIFTED current.csv ({len(current)} rows) to {DATA_DIR}")
    else:
        reference = make_data(2000, drift=False, seed=1)
        current = make_data(500, drift=False, seed=2)
        reference.to_csv(os.path.join(DATA_DIR, "reference.csv"), index=False)
        current.to_csv(os.path.join(DATA_DIR, "current.csv"), index=False)
        print(f"Wrote reference.csv ({len(reference)} rows) and current.csv ({len(current)} rows) to {DATA_DIR}")
