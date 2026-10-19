
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

from config import VALIDATION_DIR, ARTIFACTS_DIR, IMAGE_SIZE

MODEL_PATH = ARTIFACTS_DIR / "resnet18_quantity.pth"
CATALOGUE_PATH = ROOT / "data" / "processed" / "catalogue.csv"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_QUANTITY = 5


class QuantityDataset(Dataset):
    def __init__(self):
        self.metadata = pd.read_csv(VALIDATION_DIR / "metadata.csv")
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
        image = Image.open(
            VALIDATION_DIR / "images" / row["image_name"]
        ).convert("RGB")

        products = json.loads(row["products_json"])
        labels = []

        for asin in self.asins:
            product = products.get(asin)
            quantity = int(product["quantity"]) if product else 0
            labels.append(quantity)

        return self.transform(image), torch.tensor(labels, dtype=torch.long)


def main():
    dataset = QuantityDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=False
    )

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.asins) * 6)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    actual_batches = []
    predicted_batches = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images).view(images.size(0), len(dataset.asins), 6)
            predictions = logits.argmax(dim=2)

            actual_batches.append(labels.numpy())
            predicted_batches.append(predictions.cpu().numpy())

    y_true = np.concatenate(actual_batches)
    y_pred = np.concatenate(predicted_batches)

    print("\nVALIDATION QUANTITY METRICS")
    print(f"Images evaluated: {len(y_true)}")
    print(f"Whole-vector exact match: {np.all(y_true == y_pred, axis=1).mean():.4f}")
    print(f"Quantity MAE, all products: {mean_absolute_error(y_true.ravel(), y_pred.ravel()):.4f}")

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

    print("\nPER-PRODUCT RESULTS")
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
