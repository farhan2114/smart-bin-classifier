
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from config import TRAIN_DIR, VALIDATION_DIR, ARTIFACTS_DIR, RANDOM_SEED


CATALOGUE_PATH = PROJECT_ROOT / "data" / "processed" / "catalogue.csv"
MODEL_PATH = ARTIFACTS_DIR / "resnet18_presence.pth"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cpu")


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
            int(quantities[asin] > 0)
            for asin in catalogue_asins
        ])

    return df, np.asarray(labels, dtype=np.float32)


class BinDataset(Dataset):
    def __init__(self, split_dir, catalogue_asins, transform):
        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / "images"
        self.df, self.labels = load_labels(
            self.split_dir / "metadata.csv",
            catalogue_asins,
        )
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_name = str(self.df.iloc[index]["image_name"])
        image_path = self.image_dir / image_name

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return image, label


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    catalogue = pd.read_csv(CATALOGUE_PATH)
    catalogue_asins = catalogue["asin"].astype(str).str.strip().tolist()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = BinDataset(TRAIN_DIR, catalogue_asins, transform)
    val_dataset = BinDataset(VALIDATION_DIR, catalogue_asins, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Device:", DEVICE)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Catalogue products:", len(catalogue_asins))

    # Load ImageNet-pretrained weights.
    # The first run may download the weights from the internet.
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not download pretrained ResNet-18 weights. "
            "Check internet access and rerun."
        ) from exc

    # Freeze the pretrained backbone and train only the final layer.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(catalogue_asins))
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_f1 = -1.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_dataset)

        model.eval()
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(DEVICE))
                probabilities = torch.sigmoid(logits)

                all_labels.append(labels.numpy())
                all_probabilities.append(probabilities.cpu().numpy())

        y_true = np.vstack(all_labels)
        y_prob = np.vstack(all_probabilities)

        # Initial threshold only. We'll tune thresholds later if useful.
        y_pred = (y_prob >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )
        exact_match = accuracy_score(y_true, y_pred)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_precision={precision:.4f} | "
            f"val_recall={recall:.4f} | "
            f"val_f1={f1:.4f} | "
            f"val_exact_match={exact_match:.4f}"
        )

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "catalogue_asins": catalogue_asins,
                    "image_size": IMAGE_SIZE,
                    "threshold": 0.5,
                    "best_validation_f1": best_val_f1,
                },
                MODEL_PATH,
            )

    print("\nBest validation micro F1:", round(best_val_f1, 4))
    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    main()
