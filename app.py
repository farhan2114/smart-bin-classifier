# --------------------------------------------------
# Page header: show immediately
# --------------------------------------------------
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

st.title("📦 Smart Bin Classifier")
st.caption("Image-based product quantity estimation and order validation")
st.success("App startup reached the page header.")
st.title("📦 Smart Bin Classifier")
st.caption(
    "Image-based product quantity estimation and order validation"
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

    asins = catalogue[asin_column].astype(str).str.strip().tolist()
    asins = [asin for asin in asins if asin]

    if not asins:
        raise ValueError("No product ASINs were found.")

    if len(asins) != len(set(asins)):
        raise ValueError("Duplicate ASINs were found in the catalogue.")

    return asins, catalogue


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
    max_quantity = int(checkpoint.get("max_quantity", MAX_QUANTITY))

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
# Load resources with visible error handling
# --------------------------------------------------

with st.spinner("Loading catalogue and trained model..."):
    try:
        catalogue_asins, catalogue_df = load_catalogue()
        model, model_asins, max_quantity, preprocessing = load_model()

        if set(catalogue_asins) != set(model_asins):
            st.warning(
                "The catalogue and model contain different products. "
                "Predictions use the products supported by the model."
            )

    except Exception as error:
        st.error("The application could not load its required files.")
        st.exception(error)
        st.info(
            "Check that the catalogue exists at "
            "`data/processed/catalogue.csv` and the model exists at "
            "`artifacts/resnet18_quantity.pth`."
        )
        st.stop()


# --------------------------------------------------
# Predict product quantities from image only
# --------------------------------------------------

def predict_quantities(image):
    image = image.convert("RGB")
    tensor = preprocessing(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        logits = logits.reshape(
            1,
            len(model_asins),
            max_quantity + 1,
        )
        predictions = logits.argmax(dim=2)[0].cpu().tolist()

    return {
        asin: int(quantity)
        for asin, quantity in zip(model_asins, predictions)
    }


# --------------------------------------------------
# Demo image selection
# --------------------------------------------------

@st.cache_data
def get_demo_images():
    if not DEMO_IMAGES.exists():
        return []

    return [
        {
            "name": path.name,
            "path": str(path),
            "split": "demo",
        }
        for path in sorted(DEMO_IMAGES.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


# --------------------------------------------------
# Logging
# --------------------------------------------------

def write_log(order, photo_id, predictions, result, elapsed):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
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
# Application tabs
# --------------------------------------------------

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
        "Enter the requested quantities, then upload a bin photograph "
        "or choose a demo photograph."
    )

    order = {}
    st.markdown("#### Ordered quantities")

    for asin in model_asins:
        order[asin] = st.number_input(
            asin,
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
                "No demo photos found. Add JPG, JPEG, PNG, or WEBP "
                "images to data/demo_images."
            )

    if st.button("Validate order", type="primary"):
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

                start = time.perf_counter()
                predictions = predict_quantities(image)

                mismatches = [
                    asin for asin in model_asins
                    if int(order[asin]) != predictions[asin]
                ]

                result = "PASS" if not mismatches else "FAIL"
                elapsed = time.perf_counter() - start

                write_log(
                    order,
                    photo_id,
                    predictions,
                    result,
                    elapsed,
                )

                st.markdown("### Validation result")

                if result == "PASS":
                    st.success("PASS: All entered quantities match predictions.")
                else:
                    st.error(
                        f"FAIL: {len(mismatches)} product quantity "
                        "mismatch(es) detected."
                    )

                results_df = pd.DataFrame([
                    {
                        "Product ASIN": asin,
                        "Ordered": int(order[asin]),
                        "Predicted": predictions[asin],
                        "Result": (
                            "PASS"
                            if int(order[asin]) == predictions[asin]
                            else "FAIL"
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
                    f"Inference and comparison time: {elapsed:.4f} seconds."
                )

                st.download_button(
                    "Download validation results",
                    data=results_df.to_csv(index=False),
                    file_name="validation_results.csv",
                    mime="text/csv",
                )

            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"Could not process the selected image: {error}")
            except Exception as error:
                st.error(f"Validation failed: {error}")
                st.exception(error)


# ==================================================
# TAB 2: BLIND PREDICTION
# ==================================================

with tab_blind:
    st.subheader("Blind bin prediction")
    st.write(
        "Enter a bin ID and upload its photograph. Predictions are "
        "generated from the photograph alone."
    )

    blind_bin_id = st.text_input("Bin ID", key="blind_bin_id")

    blind_photo = st.file_uploader(
        "Upload blind bin photograph",
        type=["jpg", "jpeg", "png", "webp"],
        key="blind_upload",
    )

    if st.button("Predict blind bin", type="primary"):
        if not blind_bin_id.strip():
            st.error("Enter a bin ID.")
        elif blind_photo is None:
            st.error("Upload a bin photograph.")
        else:
            try:
                image = Image.open(blind_photo).convert("RGB")

                start = time.perf_counter()
                predictions = predict_quantities(image)
                elapsed = time.perf_counter() - start

                write_log(
                    {},
                    f"blind:{blind_bin_id.strip()}:{blind_photo.name}",
                    predictions,
                    "BLIND_PREDICTION",
                    elapsed,
                )

                st.image(
                    image,
                    caption="Blind bin photograph",
                    width=320,
                )

                prediction_df = pd.DataFrame([
                    {
                        "Product ASIN": asin,
                        "Predicted quantity": predictions[asin],
                    }
                    for asin in model_asins
                ])

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
                )

                st.caption(
                    "These are model estimates, not ground-truth labels. "
                    f"Inference time: {elapsed:.4f} seconds."
                )

            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"Could not process the selected image: {error}")
            except Exception as error:
                st.error(f"Blind prediction failed: {error}")
                st.exception(error)


# ==================================================
# TAB 3: ABOUT THE MODEL
# ==================================================

with tab_about:
    st.subheader("Model information")
    st.write(f"Products in model catalogue: **{len(model_asins)}**")
    st.write(f"Maximum predicted quantity per product: **{max_quantity}**")
    st.write(f"Compute device: **{DEVICE}**")

    st.info(
        "This prototype supports only the selected catalogue products. "
        "Predictions may contain false positives and false negatives "
        "and are not a production-grade guarantee."
    )

    st.markdown("#### Data handling")
    st.write(
        "Predictions are generated from the selected image. The app does "
        "not load blind-bin metadata to make predictions. Demo mode reads "
        "only from the dedicated data/demo_images folder."
    )

    st.markdown("#### Logging")
    st.write(
        "Validation and blind-prediction requests are logged with timestamps, "
        "photo IDs, model output, results, and processing time. Logs on free "
        "cloud hosting may not persist across restarts."
    )