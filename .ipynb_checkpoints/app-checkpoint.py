import streamlit as st
import torch
import json

from PIL import Image
from torchvision import transforms

from src.config import *
from src.model import build_model
from src.utils import load_checkpoint


# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Rice Leaf Disease Detection System")

st.write(
    "Upload a rice leaf image to detect the disease using EfficientNet-B0."
)


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():

    model = build_model(pretrained=False)

    model = load_checkpoint(
        model,
        MODEL_PATH / "best_model.pth",
        DEVICE
    )

    model.eval()

    return model


model = load_model()
st.write("Class Names:", CLASS_NAMES)
st.write("Model Path:", MODEL_PATH / "best_model.pth")


# ==========================================
# Load Knowledge Base
# ==========================================

@st.cache_data
def load_database():

    with open(
        PROJECT_ROOT /
        "knowledge_base" /
        "rice_disease_info.json",
        "r"
    ) as f:

        return json.load(f)


disease_db = load_database()


# ==========================================
# Upload Image
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Rice Leaf Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# Prediction
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            width=350
        )

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)
        st.write("Probabilities:", probabilities.cpu().numpy())
        st.write("Predicted Index:", predicted.item())

    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence = confidence.item()

    with col2:

        st.success(f"Prediction : {predicted_class}")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.progress(float(confidence))


    # ======================================
    # Disease Information
    # ======================================

    info = disease_db[predicted_class]

    st.divider()

    st.header("Disease Information")

    st.subheader("Description")
    st.write(info["description"])

    st.subheader("Symptoms")

    for symptom in info["symptoms"]:
        st.write("•", symptom)

    st.subheader("Causes")

    for cause in info["causes"]:
        st.write("•", cause)

    st.subheader("Treatment")

    for treatment in info["treatment"]:
        st.write("•", treatment)

    st.subheader("Prevention")

    for prevention in info["prevention"]:
        st.write("•", prevention)