
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "data/splits"
CATALOGUE_PATH = ROOT / "data/processed/catalogue.csv"

EXPECTED_COUNTS = {
    "train": 1400,
    "validation": 200,
    "test": 200,
    "blind": 200,
}

errors = []
all_image_names = []
asin_frequency = Counter()

with CATALOGUE_PATH.open(encoding="utf-8", newline="") as f:
    catalogue_rows = list(csv.DictReader(f))

catalogue_asins = {row["asin"] for row in catalogue_rows}

print("Checking catalogue...")
print("Catalogue products:", len(catalogue_asins))

for row in catalogue_rows:
    frequency = int(row["selected_bin_frequency"])
    if frequency < 100:
        errors.append(
            f"Catalogue product {row['asin']} appears only {frequency} times."
        )

for split, expected_count in EXPECTED_COUNTS.items():
    split_dir = SPLITS_DIR / split
    image_dir = split_dir / "images"
    images = sorted(image_dir.glob("*"))

    print(f"\nChecking {split}...")
    print("Image files:", len(images))

    if len(images) != expected_count:
        errors.append(
            f"{split}: expected {expected_count} images, found {len(images)}."
        )

    for image in images:
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            errors.append(f"Unexpected image file: {image}")

    if split == "blind":
        order_path = split_dir / "image_order.csv"

        if not order_path.exists():
            errors.append("Blind image_order.csv is missing.")
        else:
            with order_path.open(encoding="utf-8", newline="") as f:
                order_rows = list(csv.DictReader(f))

            ordered_names = {r["image_name"] for r in order_rows}
            actual_names = {p.name for p in images}

            if ordered_names != actual_names:
                errors.append("Blind image order does not match blind images.")

        if (split_dir / "metadata.csv").exists():
            errors.append(
                "Blind metadata.csv exists. Keep blind labels separate."
            )

        continue

    metadata_path = split_dir / "metadata.csv"

    if not metadata_path.exists():
        errors.append(f"{split}: metadata.csv is missing.")
        continue

    with metadata_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != expected_count:
        errors.append(
            f"{split}: expected {expected_count} metadata rows, found {len(rows)}."
        )

    metadata_names = {row["image_name"] for row in rows}
    actual_names = {p.name for p in images}

    if metadata_names != actual_names:
        errors.append(f"{split}: image files and metadata filenames differ.")

    for row in rows:
        try:
            products = json.loads(row["products_json"])
            expected_quantity = int(row["expected_quantity"])
            quantities = [
                int(product.get("quantity", 0))
                for product in products.values()
            ]

            if not 1 <= expected_quantity <= 5:
                errors.append(
                    f"{split}/{row['image_name']}: quantity out of range."
                )

            if sum(quantities) != expected_quantity:
                errors.append(
                    f"{split}/{row['image_name']}: quantity mismatch."
                )

            for asin in products:
                asin_frequency[asin] += 1

        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(
                f"{split}/{row['image_name']}: invalid metadata ({exc})."
            )

        all_image_names.append(row["image_name"])

if len(all_image_names) != len(set(all_image_names)):
    errors.append("Duplicate images found across train/validation/test.")

print("\nCatalogue frequency check:")
for asin in sorted(catalogue_asins):
    print(f"{asin}: {asin_frequency[asin]} bins across labelled splits")

print("\nValidation result:")
if errors:
    print(f"FAILED: {len(errors)} issue(s)")
    for error in errors[:30]:
        print("-", error)
    if len(errors) > 30:
        print(f"... and {len(errors) - 30} more")
else:
    print("PASSED: image counts, metadata, and split checks look good.")
