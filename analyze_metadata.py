
import sqlite3
import json
from collections import Counter
from pathlib import Path

DB_PATH = Path("data/raw/metadata_extracted/metadata.db")

connection = sqlite3.connect(str(DB_PATH))

total_rows = connection.execute(
    "SELECT COUNT(*) FROM metadata"
).fetchone()[0]

print(f"Total metadata records: {total_rows:,}")

quantity_counts = Counter()
valid_records = []
invalid_json = 0
empty_products = 0
quantity_mismatches = 0

cursor = connection.execute("SELECT name, metadata FROM metadata")

for image_name, raw_metadata in cursor:
    try:
        record = json.loads(raw_metadata)
    except (json.JSONDecodeError, TypeError):
        invalid_json += 1
        continue

    products = record.get("BIN_FCSKU_DATA") or {}
    expected = record.get("EXPECTED_QUANTITY")

    if not products:
        empty_products += 1
        continue

    try:
        expected = int(expected)
        product_quantities = [
            int(product.get("quantity", 0))
            for product in products.values()
        ]
    except (TypeError, ValueError):
        continue

    quantity_counts[expected] += 1

    if expected < 1 or expected > 5:
        continue

    if any(quantity < 0 for quantity in product_quantities):
        continue

    if sum(product_quantities) != expected:
        quantity_mismatches += 1
        continue

    valid_records.append({
        "image_name": image_name,
        "expected_quantity": expected,
        "products": products,
    })

connection.close()

print(f"Records with empty product data: {empty_products:,}")
print(f"Invalid JSON records: {invalid_json:,}")
print("\nExpected quantity distribution:")
for quantity, count in sorted(quantity_counts.items()):
    print(f"  {quantity}: {count:,}")

print(f"\nEligible records after quantity checks: {len(valid_records):,}")
print(f"Records excluded for quantity mismatch: {quantity_mismatches:,}")

# Count each ASIN once per bin, regardless of units in that bin.
asin_bin_counts = Counter()

for record in valid_records:
    for asin in record["products"]:
        asin_bin_counts[asin] += 1

frequent_asins = {
    asin: count
    for asin, count in asin_bin_counts.items()
    if count >= 100
}

print(f"\nDistinct ASINs in eligible records: {len(asin_bin_counts):,}")
print(f"ASINs appearing in at least 100 eligible bins: {len(frequent_asins):,}")

print("\nTop 15 ASINs by eligible-bin frequency:")
for asin, count in asin_bin_counts.most_common(15):
    print(f"  {asin}: {count:,} bins")

print("\nAnalysis complete. No images were extracted or modified.")
