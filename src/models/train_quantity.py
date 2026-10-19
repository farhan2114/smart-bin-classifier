
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

from config import TRAIN_DIR, VALIDATION_DIR, ARTIFACTS_DIR, RANDOM_SEED


CATALOGUE_PATH = PROJECT_ROOT / "data" / "processed" / "catalogue.csv"
PRESENCE_MODEL_PATH = ARTIFACTS_DIR / "resnet18_finetuned.pth"
OUTPUT_MODEL_PATH = ARTIFACTS_DIR / "resnet18_quantity.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
MAX_QUANTITY = 5

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cpu")


def load_quantity_labels(metadata_path, catalogue_asins):
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
                if asin not in quantities:
                    continue

                try:
                    quantity = int(product.get("quantity", 0))
                except (ValueError, TypeError):
                    continue

                if quantity < 0:
                    raise ValueError(
                        f"Negative quantity for {asin} in "
                        f"{row['image_name']}"
                    )

                quantities[asin] += quantity

        if any(q > MAX_QUANTITY for q in quantities.values()):
            raise ValueError(
                f"Catalogue product quantity exceeds {MAX_QUANTITY} "
                f"in {row['image_name']}"
            )

        labels.append([quantities[asin] for asin in catalogue_asins])

    return df, np.asarray(labels, dtype=np.int64)


class QuantityDataset(Dataset):
    def __init__(self, split_dir, catalogue_asins, transform):
        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / "images"
        self.df, self.labels = load_quantity_labels(
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
            image = self.transform(image.convert("RGB"))

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label



def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    actual_batches = []
    predicted_batches = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            batch_size = images.size(0)
            number_of_products = labels.size(1)

            logits = model(images).view(
                batch_size,
                number_of_products,
                MAX_QUANTITY + 1,
            )

            # CrossEntropyLoss expects class scores shaped [N, C]
            # and integer targets shaped [N].
            loss = criterion(
                logits.reshape(-1, MAX_QUANTITY + 1),
                labels.reshape(-1),
            )

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=2)
            actual_batches.append(labels.cpu().numpy())
            predicted_batches.append(predictions.cpu().numpy())

    y_true = np.concatenate(actual_batches, axis=0)
    y_pred = np.concatenate(predicted_batches, axis=0)

    exact_match = np.all(y_true == y_pred, axis=1).mean()
    mae = np.abs(y_true - y_pred).mean()

    return total_loss / len(loader.dataset), exact_match, mae

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PRESENCE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {PRESENCE_MODEL_PATH}. "
            "Run finetune_resnet.py first."
        )

    catalogue = pd.read_csv(CATALOGUE_PATH)
    catalogue_asins = catalogue["asin"].astype(str).str.strip().tolist()
    number_of_products = len(catalogue_asins)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = QuantityDataset(
        TRAIN_DIR, catalogue_asins, transform
    )
    val_dataset = QuantityDataset(
        VALIDATION_DIR, catalogue_asins, transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    checkpoint = torch.load(
        PRESENCE_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if checkpoint["catalogue_asins"] != catalogue_asins:
        raise ValueError("Checkpoint catalogue does not match catalogue.csv")

    # Load the fine-tuned visual features.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, number_of_products)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Replace the presence head with 6 quantity classes per product.
    model.fc = nn.Linear(
        model.fc.in_features,
        number_of_products * (MAX_QUANTITY + 1),
    )
    model = model.to(DEVICE)
    # Give rare quantity classes more influence during training.
    class_counts = np.zeros(MAX_QUANTITY + 1, dtype=np.float64)

    for _, labels in train_loader:
        for quantity in range(MAX_QUANTITY + 1):
            class_counts[quantity] += (labels.numpy() == quantity).sum()

    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    class_weights = class_weights / class_weights.mean()

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print("Quantity class counts:", class_counts.astype(int))
    print("Quantity class weights:", np.round(class_weights, 3))

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")

    print("Device:", DEVICE)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Catalogue products:", number_of_products)
    print("Quantity classes per product:", MAX_QUANTITY + 1)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images).view(
                images.size(0),
                number_of_products,
                MAX_QUANTITY + 1,
            )

            loss = criterion(
                logits.reshape(-1, MAX_QUANTITY + 1),
                labels.reshape(-1),
            )

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)

        train_loss = total_train_loss / len(train_dataset)
        val_loss, exact_match, mae = evaluate(
            model, val_loader, criterion
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"quantity_exact_match={exact_match:.4f} | "
            f"quantity_MAE={mae:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "catalogue_asins": catalogue_asins,
                "image_size": IMAGE_SIZE,
                "max_quantity": MAX_QUANTITY,
                "quantity_classes": MAX_QUANTITY + 1,
                "best_validation_loss": best_val_loss,
            }, OUTPUT_MODEL_PATH)

    print("\nBest validation loss:", round(best_val_loss, 4))
    print("Saved quantity model:", OUTPUT_MODEL_PATH)


if __name__ == "__main__":
    main()
