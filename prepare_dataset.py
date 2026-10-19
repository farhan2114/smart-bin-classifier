
import csv
import json
import random
import zipfile
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent
SELECTED_CSV = PROJECT_ROOT / "data/processed/selected_bins.csv"
ZIP_PATH = PROJECT_ROOT / "data/raw/archive (5).zip"
SPLITS_DIR = PROJECT_ROOT / "data/splits"

SPLIT_COUNTS = {
    "train": 1400,
    "validation": 200,
    "test": 200,
    "blind": 200,
}

random.seed(SEED)

# Read the selected records.
with SELECTED_CSV.open(encoding="utf-8", newline="") as f:
    records = list(csv.DictReader(f))

if len(records) != 2000:
    raise ValueError(f"Expected 2000 records, found {len(records)}")

# Shuffle once using a reproducible seed.
random.shuffle(records)

# Assign the split membership.
splits = {}
offset = 0

for split_name, count in SPLIT_COUNTS.items():
    splits[split_name] = records[offset:offset + count]
    offset += count

# Check for duplicate filenames across splits.
all_names = [
    row["image_name"]
    for rows in splits.values()
    for row in rows
]

if len(all_names) != len(set(all_names)):
    raise ValueError("Duplicate image names detected across splits")

print("Split counts:")
for split_name, rows in splits.items():
    print(f"  {split_name}: {len(rows)}")

# Create folders and extract only selected images.
with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    archive_names = set(archive.namelist())

    for split_name, rows in splits.items():
        image_dir = SPLITS_DIR / split_name / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Save labels separately. Blind labels must not be used by the app.
        if split_name != "blind":
            metadata_path = SPLITS_DIR / split_name / "metadata.csv"

            with metadata_path.open(
                "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_name",
                        "expected_quantity",
                        "products_json",
                    ],
                )
                writer.writeheader()

                for row in rows:
                    writer.writerow({
                        "image_name": row["image_name"],
                        "expected_quantity": row["expected_quantity"],
                        "products_json": row["products_json"],
                    })

        for index, row in enumerate(rows, start=1):
            archive_path = row["archive_path"]

            if archive_path not in archive_names:
                raise FileNotFoundError(
                    f"Missing archive member: {archive_path}"
                )

            output_path = image_dir / row["image_name"]

            if not output_path.exists():
                with archive.open(archive_path) as source:
                    with output_path.open("wb") as destination:
                        destination.write(source.read())

            if index % 100 == 0:
                print(
                    f"{split_name}: extracted {index}/{len(rows)} images",
                    flush=True,
                )

        # Save blind image IDs only, not their metadata or labels.
        if split_name == "blind":
            with (SPLITS_DIR / "blind/image_order.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["image_name"])
                for row in rows:
                    writer.writerow([row["image_name"]])

print("\nDataset preparation complete.")
print("Images extracted: 2000")
print("Blind labels were not written to the blind split.")
