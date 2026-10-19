
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
INITIAL_MODEL_PATH = ARTIFACTS_DIR / "resnet18_presence.pth"
OUTPUT_MODEL_PATH = ARTIFACTS_DIR / "resnet18_finetuned.pth"

BATCH_SIZE = 16
EPOCHS = 5
BACKBONE_LR = 0.00001
CLASSIFIER_LR = 0.0005
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


def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_labels), np.vstack(all_probabilities)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INITIAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Initial checkpoint not found: {INITIAL_MODEL_PATH}\n"
            "Run train_resnet.py first."
        )

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
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    checkpoint = torch.load(
        INITIAL_MODEL_PATH, map_location=DEVICE, weights_only=False
    )

    if checkpoint["catalogue_asins"] != catalogue_asins:
        raise ValueError("Checkpoint catalogue does not match catalogue.csv")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(catalogue_asins))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze the early layers and fine-tune the final residual block.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
        {"params": model.fc.parameters(), "lr": CLASSIFIER_LR},
    ])

    best_val_f1 = -1.0

    print("Device:", DEVICE)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Fine-tuning layer4 and fc")
    print("Epochs:", EPOCHS)

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
        y_true, y_prob = evaluate(model, val_loader)

        # Fixed threshold for comparable epoch selection.
        y_pred = (y_prob >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
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
            torch.save({
                "model_state_dict": model.state_dict(),
                "catalogue_asins": catalogue_asins,
                "image_size": IMAGE_SIZE,
                "threshold": 0.5,
                "best_validation_f1": best_val_f1,
                "fine_tuned_layers": ["layer4", "fc"],
            }, OUTPUT_MODEL_PATH)

    print("\nBest validation micro F1 at threshold 0.5:",
          round(best_val_f1, 4))
    print("Saved fine-tuned checkpoint:", OUTPUT_MODEL_PATH)


if __name__ == "__main__":
    main()
