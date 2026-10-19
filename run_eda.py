
import json
import sqlite3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

from config import (
    PROJECT_ROOT,
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    REPORTS_DIR,
)

DB_PATH = PROJECT_ROOT / "data" / "raw" / "metadata_extracted" / "metadata.db"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SMART BIN CLASSIFIER: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Load selection and catalogue
selected = pd.read_csv(PROCESSED_DIR / "selected_bins.csv")
catalogue = pd.read_csv(PROCESSED_DIR / "catalogue.csv")

required_columns = {
    "image_name", "archive_path", "expected_quantity", "products_json"
}
missing_columns = required_columns - set(selected.columns)

if missing_columns:
    raise ValueError(f"Missing selected_bins.csv columns: {missing_columns}")

print(f"\nSelected bins: {len(selected)}")
print(f"Catalogue products: {len(catalogue)}")

# 2. Parse products from the selected-bin manifest.
# This is restricted to the 2,000 selected bins.
product_records = []
empty_product_count = 0
invalid_json_count = 0
quantity_mismatch_count = 0

for _, row in selected.iterrows():
    try:
        products = json.loads(row["products_json"])
        if not isinstance(products, dict):
            products = {}
            invalid_json_count += 1
    except (json.JSONDecodeError, TypeError):
        products = {}
        invalid_json_count += 1

    if not products:
        empty_product_count += 1

    total_quantity = 0

    for product_key, product in products.items():
        if not isinstance(product, dict):
            continue

        asin = product.get("ASIN", product.get("asin", product_key))
        raw_quantity = product.get(
            "QUANTITY",
            product.get("quantity", 0)
        )

        try:
            quantity = int(raw_quantity)
        except (TypeError, ValueError):
            quantity = 0

        total_quantity += quantity

        product_records.append({
            "image_name": str(row["image_name"]),
            "asin": str(asin) if asin else None,
            "quantity": quantity,
            "expected_quantity": row["expected_quantity"],
        })

    try:
        if (
            pd.notna(row["expected_quantity"])
            and int(row["expected_quantity"]) != total_quantity
        ):
            quantity_mismatch_count += 1
    except (TypeError, ValueError):
        pass

products_df = pd.DataFrame(
    product_records,
    columns=["image_name", "asin", "quantity", "expected_quantity"]
)

products_df = products_df[products_df["asin"].notna()].copy()

# Restrict model-target analysis to the 10 catalogue products.
catalogue_asins = set(catalogue["asin"].astype(str).str.strip())

products_df["asin"] = products_df["asin"].astype(str).str.strip()

products_df = products_df[
    products_df["asin"].isin(catalogue_asins)
].copy()

print(f"Product records for catalogue: {len(products_df)}")
print(f"Distinct catalogue ASINs found: {products_df['asin'].nunique()}")
products_df.to_csv(
    METRICS_DIR / "selected_product_records.csv",
    index=False
)

# 3. Product frequency across selected bins
product_frequency = (
    products_df.groupby("asin")["image_name"]
    .nunique()
    .sort_values(ascending=False)
    .rename("bin_count")
    .reset_index()
)

product_frequency.to_csv(
    METRICS_DIR / "product_frequency.csv",
    index=False
)

plt.figure(figsize=(12, 6))
plt.bar(
    product_frequency["asin"],
    product_frequency["bin_count"]
)
plt.title("Product Frequency Across Selected Bins")
plt.xlabel("Product ASIN")
plt.ylabel("Number of Distinct Bins")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "product_frequency.png", dpi=160)
plt.close()

# 4. Product quantity distribution
valid_quantities = products_df[
    products_df["quantity"].between(1, 5)
]

quantity_counts = (
    valid_quantities["quantity"]
    .value_counts()
    .reindex([1, 2, 3, 4, 5], fill_value=0)
    .sort_index()
)

quantity_counts.rename_axis("quantity").reset_index(
    name="record_count"
).to_csv(
    METRICS_DIR / "quantity_distribution.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.bar(quantity_counts.index.astype(str), quantity_counts.values)
plt.title("Product Quantity Distribution")
plt.xlabel("Product Quantity")
plt.ylabel("Number of Product Records")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "quantity_distribution.png", dpi=160)
plt.close()

# 5. Image quality checks on labelled splits only
print("\nChecking labelled images...")

split_directories = {
    "train": TRAIN_DIR,
    "validation": VALIDATION_DIR,
    "test": TEST_DIR,
}

image_records = []
unreadable_images = []

for split_name, split_dir in split_directories.items():
    images_dir = split_dir / "images"

    if not images_dir.exists():
        print(f"WARNING: Missing image directory: {images_dir}")
        continue

    for image_path in images_dir.iterdir():
        if not image_path.is_file():
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format
                stats = ImageStat.Stat(image.convert("L"))

                image_records.append({
                    "split": split_name,
                    "image": image_path.name,
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "file_size_kb": round(
                        image_path.stat().st_size / 1024, 2
                    ),
                    "brightness_mean": round(stats.mean[0], 2),
                    "brightness_std": round(stats.stddev[0], 2),
                })

        except Exception as error:
            unreadable_images.append({
                "split": split_name,
                "image": image_path.name,
                "error": str(error),
            })

image_df = pd.DataFrame(image_records)

image_df.to_csv(METRICS_DIR / "image_quality.csv", index=False)
pd.DataFrame(unreadable_images).to_csv(
    METRICS_DIR / "unreadable_images.csv",
    index=False
)

if not image_df.empty:
    plt.figure(figsize=(8, 5))
    plt.hist(image_df["file_size_kb"], bins=30)
    plt.title("Image File Size Distribution")
    plt.xlabel("File Size (KB)")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "image_file_sizes.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(image_df["brightness_mean"], bins=30)
    plt.title("Image Brightness Distribution")
    plt.xlabel("Mean Grayscale Brightness")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "image_brightness.png", dpi=160)
    plt.close()

# 6. Check bin ID leakage across labelled splits.
# Use each split's metadata.csv if it exists.
split_ids = {}

for split_name, split_dir in split_directories.items():
    metadata_path = split_dir / "metadata.csv"

    if not metadata_path.exists():
        continue

    split_metadata = pd.read_csv(metadata_path)

    # Accommodate different metadata column naming conventions.
    id_column = next(
        (
            column for column in ["image_name", "bin_id", "image_fname"]
            if column in split_metadata.columns
        ),
        None
    )

    if id_column:
        split_ids[split_name] = set(
            split_metadata[id_column].astype(str).str.strip()
        )

leakage_records = []
names = list(split_ids.keys())

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        first, second = names[i], names[j]
        overlap = split_ids[first] & split_ids[second]

        leakage_records.append({
            "split_1": first,
            "split_2": second,
            "overlapping_image_names": len(overlap),
        })

leakage_df = pd.DataFrame(leakage_records)
leakage_df.to_csv(
    METRICS_DIR / "split_leakage_check.csv",
    index=False
)

# 7. Save summary
summary = {
    "selected_bins": len(selected),
    "catalogue_products": len(catalogue),
    "parsed_product_records": len(products_df),
    "invalid_products_json_records": invalid_json_count,
    "empty_product_records": empty_product_count,
    "quantity_mismatch_bins": quantity_mismatch_count,
    "readable_labelled_images": len(image_df),
    "unreadable_labelled_images": len(unreadable_images),
    "split_pairs_with_overlap": (
        int((leakage_df["overlapping_image_names"] > 0).sum())
        if not leakage_df.empty else 0
    ),
}

pd.Series(summary, name="value").to_csv(
    METRICS_DIR / "eda_summary.csv",
    header=True
)

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nProduct frequency:")
print(product_frequency.to_string(index=False))

print("\nSplit leakage checks:")
print(
    leakage_df.to_string(index=False)
    if not leakage_df.empty
    else "No split metadata available for comparison."
)

print("\nEDA completed.")
print(f"Charts saved to: {FIGURES_DIR}")
print(f"Metrics saved to: {METRICS_DIR}")
