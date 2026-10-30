import os
import streamlit as st
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image

from model_factory import get_model
from gradcam_utils import generate_gradcam_for_pil
from gemini_advice import get_advice


st.set_page_config(
    page_title="Domates Hastalık Tespiti",
    page_icon="🍅",
    layout="wide"
)


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

TURKCE_CLASS_NAMES = {
    "Tomato___Bacterial_spot": "Bakteriyel Leke",
    "Tomato___Early_blight": "Erken Yanıklık",
    "Tomato___Late_blight": "Geç Yanıklık",
    "Tomato___Leaf_Mold": "Yaprak Küfü",
    "Tomato___Septoria_leaf_spot": "Septoria Yaprak Lekesi",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Örümcek Akarı",
    "Tomato___Target_Spot": "Hedef Nokta",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Sarı Yaprak Kıvrılma Virüsü",
    "Tomato___Tomato_mosaic_virus": "Domates Mozaik Virüsü",
    "Tomato___healthy": "Sağlıklı"
}

MODEL_CONFIGS = {
    "Swin Transformer": {"factory_name": "swin", "path": "best_swin.pth"},
    "ViT": {"factory_name": "vit", "path": "best_vit.pth"},
    "DeiT": {"factory_name": "deit", "path": "best_deit.pth"},
    "ResNet50": {"factory_name": "resnet", "path": "best_resnet.pth"},
    "DenseNet121": {"factory_name": "densenet", "path": "best_densenet.pth"},
    "MobileNetV2": {"factory_name": "mobilenet", "path": "best_mobilenet.pth"},
    "Basic CNN": {"factory_name": "basic_cnn", "path": "best_basic_cnn.pth"},
    "Improved CNN": {"factory_name": "improved_cnn", "path": "best_improved_cnn.pth"},
    "Deeper CNN": {"factory_name": "deeper_cnn", "path": "best_deeper_cnn.pth"},
}

ALL_MODELS_OPTION = "Tüm Modeller"


st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1b5e20, #43a047);
    padding: 28px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}
.main-header h1, .main-header p {
    color: white !important;
}
.info-card {
    background-color: #102a18;
    color: #f1f8e9 !important;
    padding: 20px;
    border-radius: 15px;
    border-left: 6px solid #43a047;
    margin-bottom: 20px;
}
.result-card {
    background-color: #2b2410;
    color: #fff8e1 !important;
    padding: 22px;
    border-radius: 15px;
    border-left: 6px solid #f9a825;
    margin-top: 15px;
    margin-bottom: 20px;
}
.result-card h2, .result-card h3, .result-card p {
    color: #fff8e1 !important;
}
.ai-card {
    background-color: #10233d;
    color: #e3f2fd !important;
    padding: 22px;
    border-radius: 15px;
    border-left: 6px solid #2196f3;
    margin-top: 20px;
    margin-bottom: 15px;
}
.ai-card h3, .ai-card p {
    color: #e3f2fd !important;
}
.warning-card {
    background-color: #3a1515;
    color: #ffebee !important;
    padding: 16px;
    border-radius: 12px;
    border-left: 6px solid #ef5350;
    margin-top: 25px;
}
.small-muted {
    color: #d6d6d6 !important;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model_by_name(model_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = MODEL_CONFIGS[model_name]

    model_path = os.path.join(base_dir, "outputs", "models", config["path"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    model = get_model(config["factory_name"], num_classes=len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, model_path


def predict_image(model, image: Image.Image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=3)

    results = []

    for idx, prob in zip(
        top_indices.squeeze(0).tolist(),
        top_probs.squeeze(0).tolist()
    ):
        class_name = CLASS_NAMES[idx]
        results.append({
            "english": class_name,
            "turkish": TURKCE_CLASS_NAMES[class_name],
            "probability": prob
        })

    return results


def run_all_models(image):
    rows = []

    for model_name in MODEL_CONFIGS.keys():
        try:
            model, _ = load_model_by_name(model_name)
            results = predict_image(model, image)
            best = results[0]

            rows.append({
                "Model": model_name,
                "Tahmin": best["turkish"],
                "İngilizce Etiket": best["english"],
                "Güven Oranı (%)": round(best["probability"] * 100, 2)
            })

        except Exception as e:
            rows.append({
                "Model": model_name,
                "Tahmin": "Hata",
                "İngilizce Etiket": str(e),
                "Güven Oranı (%)": 0
            })

    return pd.DataFrame(rows)


def generate_deepercnn_gradcam(image):
    deeper_model, _ = load_model_by_name("Deeper CNN")

    target_layer = deeper_model.features[-2]

    return generate_gradcam_for_pil(
        model=deeper_model,
        pil_image=image,
        target_layer=target_layer,
        device="cpu",
        method="gradcam++"
    )

st.markdown("""
<div class="main-header">
    <h1>🍅 Akıllı Domates Hastalık Tespit Sistemi</h1>
    <p>Model seçimi, çoklu model karşılaştırması, Deeper CNN Grad-CAM ve Gemini destekli öneri sistemi</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    Bu sistemde tek model seçildiğinde seçilen model tahmin yapar, Gemini önerisi üretir ve açıklanabilirlik için Deeper CNN Grad-CAM görselleştirmesi gösterilir.
    "Tüm Modeller" seçeneğinde ise tüm modeller sadece tahmin için çalıştırılır; Gemini ve Grad-CAM devre dışı bırakılır.
</div>
""", unsafe_allow_html=True)


selected_model_name = st.selectbox(
    "🤖 Tahmin için model seç",
    list(MODEL_CONFIGS.keys()) + [ALL_MODELS_OPTION]
)

uploaded_file = st.file_uploader(
    "📤 Bir domates yaprağı görseli seç",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Yüklenen Görsel")
        st.image(image, caption="Analiz edilecek yaprak görüntüsü", width=350)

    with col2:
        st.subheader("ℹ️ Sistem Bilgisi")
        st.write(f"Seçilen seçenek: **{selected_model_name}**")
        st.write("Girdi boyutu: **224x224**")
        st.write(f"Sınıf sayısı: **{len(CLASS_NAMES)}**")

        if selected_model_name == ALL_MODELS_OPTION:
            st.warning("Tüm modeller modunda Gemini ve Grad-CAM çalışmaz.")
        else:
            st.success("Tek model modunda Gemini önerisi ve Deeper CNN Grad-CAM çalışır.")

    if st.button("🔍 Tahmin Et"):
        if selected_model_name == ALL_MODELS_OPTION:
            with st.spinner("Tüm modeller sırayla tahmin yapıyor..."):
                df_results = run_all_models(image)

            st.subheader("📊 Tüm Modellerin Tahmin Sonuçları")
            st.dataframe(df_results, use_container_width=True)

            best_row = df_results.sort_values(
                by="Güven Oranı (%)",
                ascending=False
            ).iloc[0]

            st.markdown(f"""
            <div class="result-card">
                <h3>🏆 En Yüksek Güvenli Tahmin</h3>
                <h2>{best_row['Tahmin']}</h2>
                <p><b>Model:</b> {best_row['Model']}</p>
                <p><b>Güven Oranı:</b> %{best_row['Güven Oranı (%)']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

            st.info(
                "Bu modda sadece model karşılaştırması yapılır. "
                "Gemini önerisi ve Grad-CAM görselleştirmesi bilinçli olarak çalıştırılmaz."
            )

        else:
            try:
                model, model_path = load_model_by_name(selected_model_name)
                results = predict_image(model, image)
                best_result = results[0]
                confidence_percent = best_result["probability"] * 100

                st.caption(f"Yüklenen model: {model_path}")

                st.markdown(f"""
                <div class="result-card">
                    <h3>🔍 Tespit Edilen Durum</h3>
                    <h2>{best_result['turkish']}</h2>
                    <p><b>Tahmin Yapan Model:</b> {selected_model_name}</p>
                    <p><b>Model Güven Oranı:</b> %{confidence_percent:.2f}</p>
                    <p class="small-muted">İngilizce sınıf etiketi: {best_result['english']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📊 En Yüksek 3 Tahmin")

                for i, result in enumerate(results, start=1):
                    percent = result["probability"] * 100
                    st.write(
                        f"**{i}. {result['turkish']}** "
                        f"({result['english']}) - %{percent:.2f}"
                    )
                    st.progress(float(result["probability"]))

                st.markdown("""
                <div class="ai-card">
                    <h3>🤖 Yapay Zekâ Destekli Tarımsal Öneri</h3>
                    <p>Gemini, seçilen modelin tahmin sonucuna göre sade ve uygulanabilir öneriler üretir.</p>
                </div>
                """, unsafe_allow_html=True)

                try:
                    with st.spinner("Gemini öneri hazırlıyor..."):
                        advice = get_advice(
                            best_result["turkish"],
                            confidence_percent
                        )

                    st.info(advice)

                except Exception:
                    st.warning("Gemini şu anda cevap veremedi. Model tahmini başarıyla üretildi.")

                st.subheader("🔥 Grad-CAM Görselleştirmesi")
                st.write(
                    "Bu bölümde açıklanabilirlik için her zaman Deeper CNN modeli kullanılır. "
                    "Böylece seçilen modelden bağımsız olarak daha anlaşılır bir Grad-CAM çıktısı elde edilir."
                )

                try:
                    with st.spinner("ResNet50 ile Grad-CAM hazırlanıyor..."):
                        gradcam_result = generate_deepercnn_gradcam(image)

                    predicted_english = gradcam_result["pred_class"]
                    predicted_turkce = TURKCE_CLASS_NAMES[predicted_english]

                    gc1, gc2, gc3 = st.columns(3)

                    with gc1:
                        st.image(
                            gradcam_result["original"],
                            caption="Orijinal Görsel",
                            use_container_width=True
                        )

                    with gc2:
                        st.image(
                            gradcam_result["heatmap"],
                            caption="Deeper CNN Isı Haritası",
                            use_container_width=True
                        )

                    with gc3:
                        st.image(
                            gradcam_result["overlay"],
                            caption=f"Deeper CNN Overlay - {predicted_turkce}",
                            use_container_width=True
                        )

                except Exception as gradcam_error:
                    st.warning("Deeper CNN Grad-CAM üretilemedi.")
                    st.caption(f"Grad-CAM hata detayı: {gradcam_error}")

                st.markdown("""
                <div class="warning-card">
                    ⚠️ Bu sistem akademik/prototip amaçlı geliştirilmiştir.
                    Sonuçlar kesin tanı yerine ön değerlendirme olarak kullanılmalıdır.
                    Tarımsal ilaçlama ve hastalık yönetimi için uzman görüşü alınmalıdır.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"İşlem sırasında hata oluştu: {e}")
