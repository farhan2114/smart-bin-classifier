
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "reports" / "metrics" / "order_evaluation_results.csv"
OUTPUT_PATH = ROOT / "reports" / "metrics" / "five_failure_cases.csv"


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {RESULTS_PATH}"
        )

    results = pd.read_csv(RESULTS_PATH)

    # Valid orders predicted as FAIL are false rejections.
    failures = results[
        (results["order_type"] == "valid")
        & (results["predicted_result"] == "FAIL")
    ].copy()

    cases = []

    for _, row in failures.head(5).iterrows():
        ordered = json.loads(row["ordered_quantities"])
        predicted = json.loads(row["predicted_quantities"])

        differences = []
        for asin, ordered_quantity in ordered.items():
            predicted_quantity = predicted.get(asin, 0)

            if ordered_quantity != predicted_quantity:
                differences.append({
                    "asin": asin,
                    "ordered_quantity": ordered_quantity,
                    "predicted_quantity": predicted_quantity,
                })

        cases.append({
            "image_name": row["image_name"],
            "failure_type": "Valid order incorrectly rejected",
            "number_of_quantity_mismatches": len(differences),
            "mismatched_products": json.dumps(differences),
            "analysis_note": (
                "The model's predicted catalogue quantities did not "
                "exactly match the valid order. Inspect the image and "
                "per-product quantity differences to identify the cause."
            ),
        })

    output = pd.DataFrame(cases)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("False rejections available:", len(failures))
    print("\nFive example failure cases:")
    if output.empty:
        print("No false rejection cases found.")
    else:
        print(output.to_string(index=False))

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
