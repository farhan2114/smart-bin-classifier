
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

st.set_page_config(
    page_title="Smart Bin Classifier",
    page_icon="📦",
    layout="wide",
)


# --------------------------------------------------
# Load catalogue and model
# --------------------------------------------------

@st.cache_data
def load_catalogue():
    catalogue = pd.read_csv(CATALOGUE_PATH)

    if "asin" in catalogue.columns:
        asin_column = "asin"
    elif "ASIN" in catalogue.columns:
        asin_column = "ASIN"
    else:
        asin_column = catalogue.columns[0]

    asins = catalogue[asin_column].astype(str).tolist()
    return asins, catalogue


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    asins = checkpoint["catalogue_asins"]
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


def predict_quantities(image):
    model, asins, max_quantity, preprocessing = load_model()

    image = image.convert("RGB")
    tensor = preprocessing(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        logits = logits.reshape(
            1, len(asins), max_quantity + 1
        )
        predictions = logits.argmax(dim=2)[0].cpu().tolist()

    return {
        asin: int(quantity)
        for asin, quantity in zip(asins, predictions)
    }


# --------------------------------------------------
# Demo image selection
# Metadata is used only to offer labelled demo images.
# It is never used to validate an order or predict
# quantities for a selected image.
# --------------------------------------------------

@st.cache_data
@st.cache_data
def get_demo_images():
    demo_images = []

    if DEMO_IMAGES.exists():
        for image_path in sorted(DEMO_IMAGES.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp"
            }:
                demo_images.append({
                    "name": image_path.name,
                    "path": str(image_path),
                    "split": "demo",
                })

    return demo_images


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
# User interface
# --------------------------------------------------

st.title("📦 Smart Bin Classifier")
st.caption(
    "Image-based product quantity estimation and order validation"
)

try:
    catalogue_asins, catalogue_df = load_catalogue()
    model, model_asins, max_quantity, preprocessing = load_model()

    if catalogue_asins != model_asins:
        st.warning(
            "The catalogue order differs from the model checkpoint. "
            "Predictions will use the model checkpoint's product order."
        )

except Exception as error:
    st.error(f"Could not load the model or catalogue: {error}")
    st.stop()


tab_validate, tab_blind, tab_about = st.tabs([
    "Validate an order",
    "Blind prediction",
    "About the model",
])


with tab_validate:
    st.subheader("Order validation")
    st.write(
        "Enter the requested quantity for each catalogue product, "
        "then select or upload a bin photograph."
    )

    order = {}

    with st.form("order_form"):
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
        )

        uploaded_image = None
        demo_choice = None

        if source == "Upload a photo":
            uploaded_image = st.file_uploader(
                "Upload bin image",
                type=["jpg", "jpeg", "png", "webp"],
                key="validation_upload",
            )
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
                )
                demo_choice = options[selected_label]
            else:
                st.warning("No demo images were found.")

        submitted = st.form_submit_button(
            "Validate order",
            type="primary",
        )

    if submitted:
        if source == "Upload a photo" and uploaded_image is None:
            st.error("Please upload a bin photograph.")
        elif source == "Use a demo photo" and demo_choice is None:
            st.error("Please select a demo photograph.")
        elif not any(order.values()):
            st.error("Enter a quantity greater than zero for at least one product.")
        else:
            try:
                if source == "Upload a photo":
                    image = Image.open(uploaded_image).convert("RGB")
                    photo_id = uploaded_image.name
                else:
                    image = Image.open(demo_choice["path"]).convert("RGB")
                    photo_id = demo_choice["name"]

                st.image(image, caption="Selected bin photograph", width=320)

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

                if result == "PASS":
                    st.success(
                        "PASS: Predicted quantities match the order "
                        "for all catalogue products."
                    )
                else:
                    st.error(
                        f"FAIL: {len(mismatches)} product quantity "
                        "or presence mismatch(es) detected."
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

                st.dataframe(results_df, use_container_width=True)
                st.caption(
                    f"Model inference and comparison: {elapsed:.4f} seconds. "
                    "This excludes image upload and display time."
                )

                st.download_button(
                    "Download validation results",
                    data=results_df.to_csv(index=False),
                    file_name="validation_results.csv",
                    mime="text/csv",
                )

                with st.expander("View predicted quantities"):
                    st.json(predictions)

            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"The selected file is not a readable image: {error}")
            except Exception as error:
                st.error(f"Validation failed: {error}")


with tab_blind:
    st.subheader("Blind bin prediction")
    st.write(
        "Enter a bin identifier and upload its photograph. "
        "The app predicts quantities from the photograph alone."
    )

    blind_bin_id = st.text_input("Bin ID")
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

                st.image(image, caption="Blind bin photograph", width=320)
                st.success(f"Predictions generated for bin {blind_bin_id.strip()}")

                prediction_df = pd.DataFrame([
                    {"Product ASIN": asin, "Predicted quantity": predictions[asin]}
                    for asin in model_asins
                ])
                st.dataframe(prediction_df, use_container_width=True)

                st.download_button(
                    "Download blind predictions",
                    data=prediction_df.to_csv(index=False),
                    file_name="blind_predictions.csv",
                    mime="text/csv",
                )

                st.caption(
                    "Predictions are model estimates, not ground-truth labels. "
                    f"Inference time: {elapsed:.4f} seconds."
                )

            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"The selected file is not a readable image: {error}")
            except Exception as error:
                st.error(f"Blind prediction failed: {error}")


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
        "Validation uses image predictions and the entered order. "
        "The app does not load blind-bin metadata to produce predictions. "
        "Training and validation images are offered only as demo photographs."
    )
