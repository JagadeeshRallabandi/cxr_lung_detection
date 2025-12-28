import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import cv2
import torch
import numpy as np

from models.faster_rcnn import load_model
from inference import predict

st.set_page_config(page_title="Lung X-ray Detection", layout="centered")

st.title("🫁 Lung Abnormality Detection (Faster R-CNN)")
st.write("Upload a Chest X-ray image to detect lung abnormalities.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(ROOT_DIR, "weights", "fasterrcnn_vindr_best.pth")
NUM_CLASSES = 15  # adjust if needed

@st.cache_resource
def load():
    return load_model(
        num_classes=15,   # change if needed
        weight_path="weights/fasterrcnn_vindr_best.pth",
        device=device
    )

model = load()

uploaded = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded:
    file_bytes = uploaded.read()
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    boxes, labels, scores = predict(model, img, device)

    for (x1, y1, x2, y2), s in zip(boxes, scores):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{s:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Prediction",
        use_column_width=True
    )
