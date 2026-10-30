import os
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model_factory import get_model


# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Domates Hastalık Test",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Swin Transformer Test App")


# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================
# MODEL LOAD
# =========================
@st.cache_resource
def load_model():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(
        base_dir,
        "outputs",
        "models",
        "best_swin.pth"
    )

    if not os.path.exists(model_path):
        st.error(f"Model bulunamadı:\n{model_path}")
        st.stop()

    model = get_model("swin", num_classes=len(CLASS_NAMES))

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()


# =========================
# UI
# =========================
uploaded_file = st.file_uploader(
    "Görsel seç",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred = torch.argmax(outputs, dim=1).item()

    st.success(f"Tahmin: {CLASS_NAMES[pred]}")