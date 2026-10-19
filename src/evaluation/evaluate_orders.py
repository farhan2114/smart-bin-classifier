
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import TEST_DIR, ARTIFACTS_DIR, IMAGE_SIZE

CATALOGUE_PATH = ROOT / "data" / "processed" / "catalogue.csv"
MODEL_PATH = ARTIFACTS_DIR / "resnet18_quantity.pth"
OUTPUT_DIR = ROOT / "reports" / "metrics"

BATCH_SIZE = 16
MAX_QUANTITY = 5
DEVICE = torch.device("cpu")


class TestDataset(Dataset):
    def __init__(self):
        self.metadata = pd.read_csv(TEST_DIR / "metadata.csv")
        self.catalogue = pd.read_csv(CATALOGUE_PATH)
        self.asins = self.catalogue["asin"].astype(str).str.strip().tolist()

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image_path = TEST_DIR / "images" / str(row["image_name"])

        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))

        # Labels are returned only for scoring, never passed to the model.
        products = json.loads(row["products_json"])
        quantities = {asin: 0 for asin in self.asins}

        if isinstance(products, dict):
            for asin_key, product in products.items():
                if not isinstance(product, dict):
                    continue

                asin = str(product.get("asin", asin_key)).strip()
                if asin in quantities:
                    quantities[asin] += int(product.get("quantity", 0))

        labels = [quantities[asin] for asin in self.asins]

        return image, torch.tensor(labels, dtype=torch.long), str(row["image_name"])


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = TestDataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if checkpoint["catalogue_asins"] != dataset.asins:
        raise ValueError("Model catalogue does not match catalogue.csv")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(dataset.asins) * (MAX_QUANTITY + 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_actual = []
    all_predicted = []
    all_names = []

    print("Running predictions on the held-out test images...")

    with torch.no_grad():
        for images, labels, image_names in loader:
            images = images.to(DEVICE)

            logits = model(images).view(
                images.size(0),
                len(dataset.asins),
                MAX_QUANTITY + 1,
            )

            predictions = logits.argmax(dim=2)

            all_actual.append(labels.numpy())
            all_predicted.append(predictions.cpu().numpy())
            all_names.extend(image_names)

    y_true = np.concatenate(all_actual, axis=0)
    y_pred = np.concatenate(all_predicted, axis=0)

    # Each test bin produces one intended-valid and one intended-invalid order.
    # The image model sees pixels only. Labels are used to create and score orders.
    records = []

    for index, image_name in enumerate(all_names):
        actual_quantities = y_true[index].copy()
        predicted_quantities = y_pred[index].copy()

        # Intended-valid order: quantities match the labelled catalogue contents.
        valid_order = actual_quantities.copy()

        # Intended-invalid order: alter one product quantity.
        invalid_order = actual_quantities.copy()
        candidates = np.flatnonzero(actual_quantities < MAX_QUANTITY)

        if len(candidates) == 0:
            candidates = np.flatnonzero(actual_quantities > 0)

        product_index = int(candidates[0])

        if invalid_order[product_index] < MAX_QUANTITY:
            invalid_order[product_index] += 1
        else:
            invalid_order[product_index] -= 1

        for order_type, ordered_quantities, intended_result in [
            ("valid", valid_order, "PASS"),
            ("invalid", invalid_order, "FAIL"),
        ]:
            start_time = time.perf_counter()

            # The predicted bin contents determine the system's decision.
            predicted_result = (
                "PASS"
                if np.array_equal(predicted_quantities, ordered_quantities)
                else "FAIL"
            )

            elapsed_seconds = time.perf_counter() - start_time

            records.append({
                "image_name": image_name,
                "order_type": order_type,
                "intended_result": intended_result,
                "predicted_result": predicted_result,
                "correct_decision": predicted_result == intended_result,
                "ordered_quantities": json.dumps(
                    dict(zip(dataset.asins, ordered_quantities.tolist()))
                ),
                "predicted_quantities": json.dumps(
                    dict(zip(dataset.asins, predicted_quantities.tolist()))
                ),
                "altered_product": (
                    dataset.asins[product_index]
                    if order_type == "invalid"
                    else ""
                ),
                "decision_seconds": elapsed_seconds,
            })

    results = pd.DataFrame(records)
    results_path = OUTPUT_DIR / "order_evaluation_results.csv"
    results.to_csv(results_path, index=False)

    y_expected = results["intended_result"]
    y_decided = results["predicted_result"]

    labels = ["PASS", "FAIL"]
    matrix = confusion_matrix(
        y_expected,
        y_decided,
        labels=labels,
    )

    correct = accuracy_score(y_expected, y_decided)

    invalid_orders = results[results["order_type"] == "invalid"]
    valid_orders = results[results["order_type"] == "valid"]

    false_passes = (
        (invalid_orders["predicted_result"] == "PASS").sum()
    )
    false_fails = (
        (valid_orders["predicted_result"] == "FAIL").sum()
    )

    print("\nORDER VALIDATION RESULTS")
    print("Test images:", len(dataset))
    print("Orders evaluated:", len(results))
    print(f"Overall PASS/FAIL accuracy: {correct:.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted)")
    print("Labels:", labels)
    print(matrix)

    print("\nClassification report:")
    print(classification_report(
        y_expected,
        y_decided,
        labels=labels,
        zero_division=0,
    ))

    print("\nSafety-focused metrics")
    print(
        f"False PASS rate on invalid orders: "
        f"{false_passes}/{len(invalid_orders)} "
        f"({false_passes / len(invalid_orders):.2%})"
    )
    print(
        f"False FAIL rate on valid orders: "
        f"{false_fails}/{len(valid_orders)} "
        f"({false_fails / len(valid_orders):.2%})"
    )
    print(
        "Mean decision time: "
        f"{results['decision_seconds'].mean():.6f} seconds"
    )

    print("\nSaved detailed results to:", results_path)


if __name__ == "__main__":
    main()
