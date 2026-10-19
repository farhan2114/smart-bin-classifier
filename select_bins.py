
import csv
import json
import random
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

SEED = 42
TARGET_BINS = 2000
MIN_FREQUENCY = 100
PRIORITY_COUNT = 10

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data/raw/metadata_extracted/metadata.db"
ZIP_PATH = ROOT / "data/raw/archive (5).zip"
OUTPUT_DIR = ROOT / "data/processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng = random.Random(SEED)

print("1/4 Reading eligible metadata...", flush=True)

connection = sqlite3.connect(str(DB_PATH))
records = []

for image_name, raw in connection.execute(
    "SELECT name, metadata FROM metadata"
):
    try:
        data = json.loads(raw)
        products = data.get("BIN_FCSKU_DATA") or {}
        expected = int(data.get("EXPECTED_QUANTITY"))
        quantities = [int(p.get("quantity", 0)) for p in products.values()]
    except (ValueError, TypeError, json.JSONDecodeError):
        continue

    if not products or not 1 <= expected <= 5:
        continue
    if any(q < 0 for q in quantities) or sum(quantities) != expected:
        continue

    records.append({
        "image_name": image_name,
        "expected_quantity": expected,
        "products": products,
    })

connection.close()

frequency = Counter()
for record in records:
    frequency.update(record["products"].keys())

priority_asins = [
    asin for asin, count in frequency.most_common()
    if count >= MIN_FREQUENCY
][:PRIORITY_COUNT]

if len(priority_asins) < PRIORITY_COUNT:
    raise RuntimeError("Not enough frequent products for the catalogue.")

print("Priority ASINs:", priority_asins, flush=True)
print("2/4 Selecting bins to meet catalogue frequency...", flush=True)

rng.shuffle(records)
selected = []
selected_names = set()
selected_frequency = Counter()

# First satisfy the 100-bin minimum for every priority product.
for record in records:
    if all(selected_frequency[a] >= MIN_FREQUENCY for a in priority_asins):
        break

    name = record["image_name"]
    if name in selected_names:
        continue

    contributes = any(
        asin in record["products"]
        and selected_frequency[asin] < MIN_FREQUENCY
        for asin in priority_asins
    )

    if contributes:
        selected.append(record)
        selected_names.add(name)
        for asin in record["products"]:
            selected_frequency[asin] += 1

missing = {
    asin: selected_frequency[asin]
    for asin in priority_asins
    if selected_frequency[asin] < MIN_FREQUENCY
}

if missing:
    raise RuntimeError(f"Could not meet catalogue minimum: {missing}")

if len(selected) > TARGET_BINS:
    raise RuntimeError(
        f"Frequency requirements needed {len(selected)} bins, "
        f"more than target {TARGET_BINS}."
    )

# Fill remaining places with eligible unique bins.
for record in records:
    if len(selected) >= TARGET_BINS:
        break
    if record["image_name"] in selected_names:
        continue

    selected.append(record)
    selected_names.add(record["image_name"])
    for asin in record["products"]:
        selected_frequency[asin] += 1

if len(selected) != TARGET_BINS:
    raise RuntimeError(f"Selected only {len(selected)} bins.")

# Recalculate frequencies across all selected bins.
selected_frequency = Counter()
product_details = {}

for record in selected:
    for asin, product in record["products"].items():
        selected_frequency[asin] += 1
        product_details[asin] = product

catalogue = {
    asin for asin in priority_asins
    if selected_frequency[asin] >= MIN_FREQUENCY
}

if len(catalogue) != PRIORITY_COUNT:
    raise RuntimeError("Final catalogue frequency verification failed.")

print("3/4 Indexing image ZIP...", flush=True)

with zipfile.ZipFile(ZIP_PATH) as archive:
    archive_index = {}
    for member in archive.namelist():
        if member.lower().endswith((".jpg", ".jpeg", ".png")):
            archive_index[Path(member).name] = member

    missing_images = [
        r["image_name"] for r in selected
        if r["image_name"] not in archive_index
    ]

    if missing_images:
        raise RuntimeError(f"Missing image files: {missing_images[:5]}")

    print("4/4 Saving selection and catalogue...", flush=True)

    with (OUTPUT_DIR / "selected_bins.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_name", "archive_path",
            "expected_quantity", "products_json"
        ])
        writer.writeheader()
        for r in selected:
            writer.writerow({
                "image_name": r["image_name"],
                "archive_path": archive_index[r["image_name"]],
                "expected_quantity": r["expected_quantity"],
                "products_json": json.dumps(r["products"]),
            })

with (OUTPUT_DIR / "catalogue.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=[
        "asin", "product_name", "selected_bin_frequency"
    ])
    writer.writeheader()
    for asin in sorted(catalogue):
        writer.writerow({
            "asin": asin,
            "product_name": product_details[asin].get("name", ""),
            "selected_bin_frequency": selected_frequency[asin],
        })

print("\nSelection successful.")
print("Selected bins:", len(selected))
print("Catalogue products:", len(catalogue))
for asin in sorted(catalogue):
    print(f"{asin}: {selected_frequency[asin]} selected bins")
print("Images extracted: 0")
