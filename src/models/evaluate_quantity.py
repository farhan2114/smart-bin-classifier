
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import TEST_DIR, ARTIFACTS_DIR, IMAGE_SIZE

MODEL_PATH = ARTIFACTS_DIR / "resnet18_quantity.pth"
CATALOGUE_PATH = ROOT / "data" / "processed" / "catalogue.csv"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_QUANTITY = 5
NUM_QUANTITY_CLASSES = MAX_QUANTITY + 1


class QuantityDataset(Dataset):
    def __init__(self):
        metadata_path = TEST_DIR / "metadata.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Test metadata not found: {metadata_path}"
            )

        if not CATALOGUE_PATH.exists():
            raise FileNotFoundError(
                f"Catalogue not found: {CATALOGUE_PATH}"
            )

        self.metadata = pd.read_csv(metadata_path)
        self.catalogue = pd.read_csv(CATALOGUE_PATH)
        self.asins = self.catalogue["asin"].astype(str).tolist()

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

        if not image_path.exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        products = json.loads(row["products_json"])
        labels = []

        for asin in self.asins:
            product = products.get(asin)
            quantity = int(product["quantity"]) if product else 0

            if not 0 <= quantity <= MAX_QUANTITY:
                raise ValueError(
                    f"Invalid quantity {quantity} for ASIN {asin} "
                    f"in image {row['image_name']}"
                )

            labels.append(quantity)

        return (
            self.transform(image),
            torch.tensor(labels, dtype=torch.long),
        )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    dataset = QuantityDataset()

    if len(dataset) == 0:
        raise ValueError("The test dataset contains no images.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(dataset.asins) * NUM_QUANTITY_CLASSES,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    actual_batches = []
    predicted_batches = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            logits = model(images).view(
                images.size(0),
                len(dataset.asins),
                NUM_QUANTITY_CLASSES,
            )

            # Argmax quantity prediction for this evaluation.
            predictions = logits.argmax(dim=2)

            actual_batches.append(labels.numpy())
            predicted_batches.append(predictions.cpu().numpy())

    y_true = np.concatenate(actual_batches, axis=0)
    y_pred = np.concatenate(predicted_batches, axis=0)

    print("\nFINAL TEST QUANTITY METRICS")
    print(f"Images evaluated: {len(y_true)}")
    print(f"Catalogue products: {len(dataset.asins)}")
    print(
        "Whole-vector exact match: "
        f"{np.all(y_true == y_pred, axis=1).mean():.4f}"
    )
    print(
        "Quantity MAE, all products: "
        f"{mean_absolute_error(y_true.ravel(), y_pred.ravel()):.4f}"
    )

    present = y_true > 0
    print(f"Actual present product entries: {present.sum()}")

    if present.any():
        print(
            "MAE on actually present products: "
            f"{mean_absolute_error(y_true[present], y_pred[present]):.4f}"
        )
        print(
            "Exact quantity accuracy on actually present products: "
            f"{accuracy_score(y_true[present], y_pred[present]):.4f}"
        )

    print("\nPER-PRODUCT TEST RESULTS")

    for index, asin in enumerate(dataset.asins):
        true_q = y_true[:, index]
        pred_q = y_pred[:, index]
        actual_present = true_q > 0

        present_accuracy = (
            np.mean(true_q[actual_present] == pred_q[actual_present])
            if actual_present.any()
            else float("nan")
        )

        print(
            f"{asin} | "
            f"present_bins={actual_present.sum():3d} | "
            f"quantity_MAE={mean_absolute_error(true_q, pred_q):.4f} | "
            f"present_quantity_accuracy={present_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
