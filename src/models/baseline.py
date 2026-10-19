
import sys
from pathlib import Path

from PIL import report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
)

CATALOGUE_PATH = PROJECT_ROOT / "data" / "processed" / "catalogue.csv"




def load_labels(split_dir, catalogue_asins):
    """Create a multi-label presence matrix from split metadata."""
    metadata_path = split_dir / "metadata.csv"
    df = pd.read_csv(metadata_path)

    labels = []

    for _, row in df.iterrows():
        try:
            products = json.loads(row["products_json"])
        except (json.JSONDecodeError, TypeError):
            products = {}

        if not isinstance(products, dict):
            products = {}

        quantities = {asin: 0 for asin in catalogue_asins}

        for asin_key, product in products.items():
            if not isinstance(product, dict):
                continue

            asin = str(product.get("asin", asin_key)).strip()

            if asin not in quantities:
                continue

            try:
                quantity = int(product.get("quantity", 0))
            except (ValueError, TypeError):
                quantity = 0

            quantities[asin] += max(quantity, 0)

        labels.append([
            int(quantities[asin] > 0)
            for asin in catalogue_asins
        ])

    return df, np.asarray(labels, dtype=np.int64)



def main():
    catalogue = pd.read_csv(CATALOGUE_PATH)
    catalogue_asins = catalogue["asin"].astype(str).str.strip().tolist()

    train_df, y_train = load_labels(TRAIN_DIR, catalogue_asins)
    val_df, y_val = load_labels(VALIDATION_DIR, catalogue_asins)

    print(f"Catalogue products: {len(catalogue_asins)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Training label matrix: {y_train.shape}")
    print(f"Validation label matrix: {y_val.shape}")

    # Model 1: a frequency-based baseline.
    # Predict each product as present if it appears in at least half
    # of the training images.
    train_prevalence = y_train.mean(axis=0)
    threshold = 0.5
    y_pred_frequency = np.tile(
        (train_prevalence >= threshold).astype(int),
        (len(y_val), 1),
    )

    # Model 2: image-feature baseline using RGB colour histograms.
    # These features are simple and computationally inexpensive.
    from PIL import Image

    def image_features(image_path):
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((128, 128))
            image_array = np.asarray(image)

        features = []
        for channel in range(3):
            hist, _ = np.histogram(
                image_array[:, :, channel],
                bins=32,
                range=(0, 256),
                density=True,
            )
            features.extend(hist.tolist())

        return features

    def build_features(df, split_dir):
        image_dir = split_dir / "images"
        rows = []

        for image_name in df["image_name"]:
            path = image_dir / str(image_name)
            if not path.exists():
                raise FileNotFoundError(f"Missing image: {path}")
            rows.append(image_features(path))

        return np.asarray(rows, dtype=np.float32)

    print("\nExtracting training image features...")
    X_train = build_features(train_df, TRAIN_DIR)

    print("Extracting validation image features...")
    X_val = build_features(val_df, VALIDATION_DIR)

    # Train one binary logistic regression model per catalogue product.
    classifier = MultiOutputClassifier(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        )
    )

    print("\nTraining image-feature baseline...")
    classifier.fit(X_train, y_train)
    
    # Get the probability that each catalogue product is present.
    probability_outputs = classifier.predict_proba(X_val)

    y_prob = np.column_stack([
        proba[:, list(est.classes_).index(1)]
        if 1 in est.classes_
        else np.zeros(len(X_val))
        for est, proba in zip(
            classifier.estimators_,
            probability_outputs,
        )
    ])

    # Original predictions using the standard 0.5 threshold.
    y_pred_image = (y_prob >= 0.5).astype(int)

    # Tune one threshold per product using validation F1.
    # This is for model selection, not an unbiased final test score.
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_thresholds = []

    for i, asin in enumerate(catalogue_asins):
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in thresholds:
            prediction = (y_prob[:, i] >= threshold).astype(int)

            _, _, f1, _ = precision_recall_fscore_support(
                y_val[:, i],
                prediction,
                average="binary",
                zero_division=0,
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        best_thresholds.append(best_threshold)

    best_thresholds = np.asarray(best_thresholds)
    y_pred_tuned = (
        y_prob >= best_thresholds.reshape(1, -1)
    ).astype(int)

    print("\nPer-product validation thresholds")
    for asin, threshold in zip(catalogue_asins, best_thresholds):
        print(f"{asin}: {threshold:.2f}")



    def report(name, y_true, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )

        exact_match = accuracy_score(y_true, y_pred)

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Micro precision: {precision:.4f}")
        print(f"Micro recall:    {recall:.4f}")
        print(f"Micro F1:        {f1:.4f}")
        print(f"Exact-match accuracy: {exact_match:.4f}")

    report("Frequency baseline", y_val, y_pred_frequency)
    report("RGB histogram + logistic regression (threshold 0.5)", y_val, y_pred_image)
    report("RGB histogram + logistic regression (tuned thresholds)", y_val, y_pred_tuned)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(
        classifier,
        ARTIFACTS_DIR / "rgb_histogram_baseline.joblib",
    )

    np.savez_compressed(
        ARTIFACTS_DIR / "baseline_label_data.npz",
        catalogue_asins=np.asarray(catalogue_asins),
        train_prevalence=train_prevalence,
        y_validation=y_val,
        y_pred_frequency=y_pred_frequency,
        y_pred_image=y_pred_image,
        y_prob=y_prob,
        best_thresholds=best_thresholds,
        y_pred_tuned=y_pred_tuned,
    )

    print("\nSaved model to artifacts/rgb_histogram_baseline.joblib")
    print("Saved validation predictions to artifacts/baseline_label_data.npz")


if __name__ == "__main__":
    main()
