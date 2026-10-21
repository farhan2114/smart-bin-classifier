
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "artifacts" / "resnet18_quantity.pth"
CATALOGUE_PATH = ROOT / "data" / "processed" / "catalogue.csv"
DEMO_IMAGES = ROOT / "data" / "demo_images"
LOG_PATH = ROOT / "logs" / "validation_log.csv"

DEVICE = torch.device("cpu")
MAX_QUANTITY = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

st.set_page_config(
    page_title="Smart Bin Classifier",
    page_icon="📦",
    layout="wide",
)


# --------------------------------------------------
# Load catalogue
# --------------------------------------------------

@st.cache_data
def load_catalogue():
    if not CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"Catalogue not found: {CATALOGUE_PATH}"
        )

    catalogue = pd.read_csv(CATALOGUE_PATH)

    if catalogue.empty:
        raise ValueError("The catalogue file is empty.")

    if "asin" in catalogue.columns:
        asin_column = "asin"
    elif "ASIN" in catalogue.columns:
        asin_column = "ASIN"
    else:
        asin_column = catalogue.columns[0]

    asins = catalogue[asin_column].astype(str).tolist()
    asins = [asin.strip() for asin in asins if asin.strip()]

    if not asins:
        raise ValueError("No product ASINs were found in the catalogue.")

    if len(asins) != len(set(asins)):
        raise ValueError("The catalogue contains duplicate ASINs.")

    return asins, catalogue


def get_product_name_map(catalogue_df):
    """Return a readable product name for each ASIN, falling back to the ASIN."""
    name_map = {}

    asin_column = next(
        (column for column in ("asin", "ASIN") if column in catalogue_df.columns),
        catalogue_df.columns[0] if len(catalogue_df.columns) else None,
    )

    if asin_column is None:
        return name_map

    for _, row in catalogue_df.iterrows():
        asin = str(row.get(asin_column, "")).strip()
        raw_name = row.get("product_name", row.get("Product Name", ""))
        name = "" if pd.isna(raw_name) else str(raw_name).strip()

        if asin:
            name_map[asin] = name if name and name.lower() != "nan" else asin

    return name_map


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    asins = [str(asin) for asin in checkpoint["catalogue_asins"]]
    image_size = int(checkpoint.get("image_size", 224))
    max_quantity = int(
        checkpoint.get("max_quantity", MAX_QUANTITY)
    )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(asins) * (max_quantity + 1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, asins, max_quantity, preprocessing


# --------------------------------------------------
# Predict product quantities from the image only
# --------------------------------------------------



def predict_quantities(image):
    model, asins, max_quantity, preprocessing = load_model()

    image = image.convert("RGB")
    tensor = preprocessing(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        logits = logits.reshape(
            1, len(asins), max_quantity + 1
        )

        probabilities = torch.softmax(logits, dim=2)[0]
        predictions = []

        for product_probs in probabilities:
            nonzero_probability = 1.0 - product_probs[0].item()

            if nonzero_probability >= 0.30:
                quantity = int(
                    torch.argmax(product_probs[1:]).item() + 1
                )
            else:
                quantity = 0

            predictions.append(quantity)

    return {
        asin: int(quantity)
        for asin, quantity in zip(asins, predictions)
    }


# --------------------------------------------------
# Demo image selection
# Uses only the dedicated demo_images folder.
# Does not read any bin metadata or ground-truth labels.
# --------------------------------------------------

@st.cache_data
def get_demo_images():
    demo_images = []

    if not DEMO_IMAGES.exists():
        return demo_images

    for image_path in sorted(DEMO_IMAGES.iterdir()):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            demo_images.append({
                "name": image_path.name,
                "path": str(image_path),
                "split": "demo",
            })

    return demo_images


# --------------------------------------------------
# Write validation and prediction logs
# --------------------------------------------------

def write_log(order, photo_id, predictions, result, elapsed):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "order",
                "photo_id",
                "model_output",
                "result",
                "processing_seconds",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().astimezone().isoformat(),
            "order": json.dumps(order),
            "photo_id": photo_id,
            "model_output": json.dumps(predictions),
            "result": result,
            "processing_seconds": round(elapsed, 6),
        })


# --------------------------------------------------
# Main application
# --------------------------------------------------

# Render the page header before loading the model so the page is not
# completely blank while the model or catalogue is being initialized.
st.title("📦 Smart Bin Classifier")
st.caption(
    "Image-based product quantity estimation and order validation"
)

try:
    with st.spinner("Loading the product catalogue and trained model..."):
        catalogue_asins, catalogue_df = load_catalogue()
        product_names = get_product_name_map(catalogue_df)
        model, model_asins, max_quantity, preprocessing = load_model()

    if catalogue_asins != model_asins:
        st.warning(
            "The catalogue order differs from the model checkpoint. "
            "The model checkpoint's product order will be used."
        )

    if set(catalogue_asins) != set(model_asins):
        st.warning(
            "The catalogue and trained model contain different products. "
            "Predictions are limited to products supported by the model."
        )

except Exception as error:
    st.error(f"Could not load the model or catalogue: {error}")
    st.stop()


tab_validate, tab_blind, tab_about = st.tabs([
    "Validate an order",
    "Blind prediction",
    "About the model",
])


# ==================================================
# TAB 1: ORDER VALIDATION
# ==================================================

with tab_validate:
    st.subheader("Order validation")

    st.write(
        "Enter the quantities requested for each product, "
        "then upload a photograph or choose a demo photograph."
    )

    # Quantity inputs are outside a form so photo-source
    # changes update the interface immediately.
    order = {}

    st.markdown("#### Ordered quantities")

    for asin in model_asins:
        product_name = product_names.get(asin, asin)
        order[asin] = st.number_input(
            f"{product_name} ({asin})",
            min_value=0,
            max_value=max_quantity,
            value=0,
            step=1,
            key=f"order_{asin}",
        )

    st.markdown("#### Bin photograph")

    source = st.radio(
        "Choose photo source",
        ["Upload a photo", "Use a demo photo"],
        horizontal=True,
        key="photo_source",
    )

    uploaded_image = None
    demo_choice = None

    if source == "Upload a photo":
        uploaded_image = st.file_uploader(
            "Upload bin image",
            type=["jpg", "jpeg", "png", "webp"],
            key="validation_upload",
        )

        if uploaded_image is not None:
            try:
                preview_image = Image.open(uploaded_image).convert("RGB")
                st.image(
                    preview_image,
                    caption="Uploaded bin photograph",
                    width=320,
                )
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("The uploaded file is not a readable image.")

    else:
        demo_images = get_demo_images()

        if demo_images:
            options = {
                f"{item['name']} ({item['split']})": item
                for item in demo_images
            }

            selected_label = st.selectbox(
                "Choose a demo photograph",
                list(options.keys()),
                key="demo_photo_choice",
            )

            demo_choice = options[selected_label]

            try:
                preview_image = Image.open(
                    demo_choice["path"]
                ).convert("RGB")

                st.image(
                    preview_image,
                    caption=f"Demo photograph: {demo_choice['name']}",
                    width=320,
                )

            except (UnidentifiedImageError, OSError, ValueError):
                st.error("The selected demo image cannot be opened.")

        else:
            st.warning(
                "No demo images were found. Add JPG, JPEG, PNG, or WEBP "
                "files to data/demo_images and push them to GitHub. "
                "You can still use the Upload a photo option."
            )

    submitted = st.button(
        "Validate order",
        type="primary",
        key="validate_order_button",
    )

    if submitted:
        if source == "Upload a photo" and uploaded_image is None:
            st.error("Please upload a bin photograph.")

        elif source == "Use a demo photo" and demo_choice is None:
            st.error("Please select an available demo photograph.")

        else:
            try:
                if source == "Upload a photo":
                    image = Image.open(uploaded_image).convert("RGB")
                    photo_id = uploaded_image.name
                else:
                    image = Image.open(
                        demo_choice["path"]
                    ).convert("RGB")
                    photo_id = demo_choice["name"]

                # Reject an empty order so it cannot pass by default.
                requested_items = {
                    asin: int(quantity)
                    for asin, quantity in order.items()
                    if int(quantity) > 0
                }

                if not requested_items:
                    st.error("Enter a quantity greater than zero for at least one product.")
                else:
                    start = time.perf_counter()
                    predictions = predict_quantities(image)

                    # An order passes only when every requested product has
                    # a predicted quantity at least as large as requested.
                    # Unrequested extra products do not cause a failure.
                    mismatches = [
                        asin
                        for asin, quantity in requested_items.items()
                        if predictions.get(asin, 0) < quantity
                    ]

                    result = "PASS" if not mismatches else "FAIL"
                    elapsed = time.perf_counter() - start

                    write_log(
                        requested_items,
                        photo_id,
                        predictions,
                        result,
                        elapsed,
                    )

                    st.markdown("### Validation result")

                    if result == "PASS":
                        st.success(
                            "PASS: Every requested product has a sufficient "
                            "predicted quantity. Extra products do not affect the result."
                        )
                    else:
                        st.error(
                            f"FAIL: {len(mismatches)} requested product(s) "
                            "are missing or have insufficient predicted quantities."
                        )

                    results_df = pd.DataFrame([
                        {
                            "Product": product_names.get(asin, asin),
                            "ASIN": asin,
                            "Ordered": int(order[asin]),
                            "Predicted": predictions.get(asin, 0),
                            "Result": (
                                "NOT REQUESTED"
                                if int(order[asin]) == 0
                                else (
                                    "PASS"
                                    if predictions.get(asin, 0) >= int(order[asin])
                                    else "FAIL"
                                )
                            ),
                        }
                        for asin in model_asins
                    ])

                    st.dataframe(
                        results_df,
                        width="stretch",
                        hide_index=True,
                    )

                    st.caption(
                        f"Model inference and comparison: {elapsed:.4f} seconds. "
                        "This excludes image upload and display time."
                    )

                    st.download_button(
                        "Download validation results",
                        data=results_df.to_csv(index=False),
                        file_name="validation_results.csv",
                        mime="text/csv",
                        key="download_validation_results",
                    )

                    with st.expander("View predicted quantities"):
                        named_predictions = {
                            f"{product_names.get(asin, asin)} ({asin})": quantity
                            for asin, quantity in predictions.items()
                        }
                        st.json(named_predictions)

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as error:
                st.error(
                    f"The selected file is not a readable image: {error}"
                )

            except Exception as error:
                st.error(f"Validation failed: {error}")


# ==================================================
# TAB 2: BLIND PREDICTION
# ==================================================

# ==============================
# TAB 2: BLIND PREDICTION
# ==============================


with tab_blind:
    st.subheader("Blind bin prediction")

    st.write(
        "Upload a bin photograph to generate quantity predictions "
        "using the trained model. Ground-truth labels are not used "
        "for inference."
    )

    uploaded_image = st.file_uploader(
        "Upload a bin photograph",
        type=["jpg", "jpeg", "png"],
        key="blind_uploaded_image",
    )

    if uploaded_image is not None:
        try:
            with Image.open(uploaded_image) as source_image:
                image = source_image.convert("RGB")

            blind_bin_id = st.text_input(
                "Bin ID (optional)",
                key="blind_uploaded_bin_id",
            )

            st.image(
                image,
                caption="Uploaded bin photograph",
                width=320,
            )

            if st.button(
                "Predict blind bin",
                type="primary",
                key="predict_blind_upload_button",
            ):
                try:
                    start = time.perf_counter()

                    # Use the same prediction function as order validation.
                    predictions = predict_quantities(image)

                    elapsed = time.perf_counter() - start

                    prediction_df = pd.DataFrame([
                        {
                            "Product": product_names.get(asin, asin),
                            "ASIN": asin,
                            "Predicted Quantity": quantity,
                        }
                        for asin, quantity in predictions.items()
                    ])

                    write_log(
                        {},
                        f"blind:{blind_bin_id.strip() or 'uploaded-image'}",
                        predictions,
                        "BLIND_PREDICTION",
                        elapsed,
                    )

                    st.success("Blind prediction completed.")

                    st.dataframe(
                        prediction_df,
                        width="stretch",
                        hide_index=True,
                    )

                    st.download_button(
                        "Download blind predictions",
                        data=prediction_df.to_csv(index=False),
                        file_name="blind_predictions.csv",
                        mime="text/csv",
                        key="download_blind_upload_predictions",
                    )

                    st.caption(
                        "These are model predictions, not ground-truth labels. "
                        f"Inference time: {elapsed:.4f} seconds."
                    )

                except Exception as error:
                    st.error(f"Blind prediction failed: {error}")

        except Exception as error:
            st.error(f"Could not read the uploaded image: {error}")

# ==================================================
# TAB 3: ABOUT THE MODEL
# ==================================================

with tab_about:
    st.subheader("Model information")

    st.write(f"Products in model catalogue: **{len(model_asins)}**")
    st.write(f"Maximum predicted quantity per product: **{max_quantity}**")
    st.write(f"Compute device: **{DEVICE}**")

    st.info(
        "This prototype only covers the ten selected catalogue products. "
        "A predicted quantity of zero does not prove an item is absent. "
        "The model can make false positives and false negatives, so it "
        "should not be treated as a production-grade guarantee."
    )

    st.markdown("#### Data handling")

    st.write(
        "Order validation uses model predictions from the selected image "
        "and the quantities entered by the user. The app does not load "
        "blind-bin metadata to make predictions. Demo mode reads images "
        "only from the dedicated data/demo_images folder."
    )

    st.markdown("#### Logging")

    st.write(
        "Validation and blind-prediction requests are logged with a "
        "timestamp, photo identifier, entered order where applicable, "
        "model output, result, and measured inference/comparison time. "
        "Logs on free cloud hosting may not persist across restarts."
    )
