
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
)

from config import VALIDATION_DIR, ARTIFACTS_DIR

MODEL_PATH = ARTIFACTS_DIR / "resnet18_finetuned.pth"
CATALOGUE_PATH = PROJECT_ROOT / "data" / "processed" / "catalogue.csv"
DEVICE = torch.device("cpu")
BATCH_SIZE = 16


def load_labels(metadata_path, catalogue_asins):
    df = pd.read_csv(metadata_path)
    labels = []

    for _, row in df.iterrows():
        try:
            products = json.loads(row["products_json"])
        except (json.JSONDecodeError, TypeError):
            products = {}

        quantities = {asin: 0 for asin in catalogue_asins}

        if isinstance(products, dict):
            for asin_key, product in products.items():
                if not isinstance(product, dict):
                    continue

                asin = str(product.get("asin", asin_key)).strip()
                if asin in quantities:
                    try:
                        quantities[asin] += max(
                            0, int(product.get("quantity", 0))
                        )
                    except (ValueError, TypeError):
                        pass

        labels.append([
            int(quantities[asin] > 0) for asin in catalogue_asins
        ])

    return df, np.asarray(labels, dtype=np.int64)


class ValidationDataset(Dataset):
    def __init__(self, split_dir, asins, transform):
        self.split_dir = Path(split_dir)
        self.df, self.labels = load_labels(
            self.split_dir / "metadata.csv", asins
        )
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.split_dir / "images" / str(
            self.df.iloc[index]["image_name"]
        )
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, self.labels[index]


def main():
    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=False
    )
    asins = checkpoint["catalogue_asins"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ValidationDataset(VALIDATION_DIR, asins, transform)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(asins))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_y, all_prob = [], []

    with torch.no_grad():
        for images, labels in loader:
            probabilities = torch.sigmoid(model(images.to(DEVICE)))
            all_y.append(labels.numpy())
            all_prob.append(probabilities.cpu().numpy())

    y_true = np.vstack(all_y)
    y_prob = np.vstack(all_prob)

    thresholds = np.arange(0.05, 0.96, 0.05)
    best_thresholds = []

    print("Per-product threshold tuning")
    print("-" * 65)

    for i, asin in enumerate(asins):
        best_threshold = 0.5
        best_f1 = -1

        for threshold in thresholds:
            pred = (y_prob[:, i] >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true[:, i], pred,
                average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        best_thresholds.append(best_threshold)
        print(f"{asin}: threshold={best_threshold:.2f}, F1={best_f1:.4f}")

    best_thresholds = np.asarray(best_thresholds)
    y_pred = (y_prob >= best_thresholds.reshape(1, -1)).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    exact = accuracy_score(y_true, y_pred)

    print("\nTuned validation results")
    print(f"Micro precision: {precision:.4f}")
    print(f"Micro recall: {recall:.4f}")
    print(f"Micro F1: {f1:.4f}")
    print(f"Exact-match accuracy: {exact:.4f}")

    np.savez_compressed(
        ARTIFACTS_DIR / "resnet18_finetuned_validation_predictions.npz",
        catalogue_asins=np.asarray(asins),
        y_validation=y_true,
        y_prob=y_prob,
        best_thresholds=best_thresholds,
        y_pred_tuned=y_pred,
    )

    print("\nSaved tuned validation predictions.")


if __name__ == "__main__":
    main()
