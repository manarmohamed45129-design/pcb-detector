from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"

st.set_page_config(page_title="PCB Defect Detection", layout="centered")
st.title("PCB Defect Detection")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error("best.pt not found. Put the trained weights next to app.py.")
    st.stop()

model = load_model()

conf = st.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)
file = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    result = model.predict(img, imgsz=640, conf=conf, device="cpu")[0]

    st.image(
        result.plot()[:, :, ::-1],
        caption=f"{len(result.boxes)} defects found",
        use_container_width=True,
    )

    if len(result.boxes) > 0:
        counts = {}
        for c in result.boxes.cls.tolist():
            name = model.names[int(c)]
            counts[name] = counts.get(name, 0) + 1
        st.write("Defects per class:", counts)
