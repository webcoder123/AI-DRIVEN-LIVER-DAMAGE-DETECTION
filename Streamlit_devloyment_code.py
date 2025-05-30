import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("D:\\360DigiTMG Date 28Aug\\Liver Damage Detection Project\\Model Building\\DenseNet121_best_model.h5")

# Class names
class_names = ["CC", "HCC", "NORMAL LIVER"]

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# App config
st.set_page_config(page_title="Liver Damage Detection", page_icon="🧬", layout="centered")

# Custom CSS styles
st.markdown("""
    <style>
        .main-title {
            font-size: 38px;
            font-weight: bold;
            color: #0078D4;
            text-align: center;
        }
        .sub-header {
            font-size: 20px;
            color: #333;
            text-align: center;
        }
        .section-title {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin-top: 30px;
        }
        .result-text {
            font-size: 22px;
            font-weight: bold;
            color: #0E1117;
        }
        .confidence-text {
            font-size: 18px;
            color: #666;
        }
        .footer {
            font-size: 16px;
            text-align: center;
            color: #777;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# Hero image
st.image("D:\\360DigiTMG Date 28Aug\\Liver Damage Detection Project\\Model Building\\Liver_imag.png", use_column_width=True)

# Title
st.markdown('<div class="main-title">🧬 Liver Damage Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a histopathology image (HCC, CC, or Normal Liver) to analyze liver condition with our AI-powered tool.</div>', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="section-title">📁 Upload Your Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing image..."):
        try:
            processed_img = preprocess_image(img)
            predictions = model.predict(processed_img)

            if predictions is not None and len(predictions) > 0:
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = np.max(predictions[0]) * 100

                st.markdown('<div class="section-title">🔎 Prediction Result</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-text">✅ Predicted Class: <span style="color:#0078D4;">{predicted_class}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-text">📊 Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

                st.subheader("📈 Class-wise Confidence")
                for i, cls in enumerate(class_names):
                    st.progress(float(predictions[0][i]))
                    st.markdown(f"**{cls}**: {predictions[0][i] * 100:.2f}%")

            else:
                st.error("🚫 Model returned no predictions. Please check the image or model.")

        except Exception as e:
            st.error(f"🚫 Error during prediction: {str(e)}")

# Footer
st.markdown('<div class="footer">🔗 Developed by <b>Roushan Kumar</b> | Powered by TensorFlow & Streamlit</div>', unsafe_allow_html=True)
