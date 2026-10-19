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
