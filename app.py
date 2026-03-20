import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("🚀 ClearLabel AI")
st.subheader("Automated Annotation & Data Quality Audit Pipeline")
st.write("Upload an image to run the AI Auto-Annotation and Quality Audit.")

# Load Model (FIXED)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Layer 1: Quality Audit
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    st.subheader("Layer 1: Quality Audit")
    if variance < 100:
        st.error(f"🚩 Image Rejected: Too Blurry (Variance: {variance:.2f})")
    else:
        st.success(f"✅ Image Passed Quality Check (Variance: {variance:.2f})")

        # Layer 2: AI Audit
        results = model(img_cv)[0]
        
        st.subheader("Layer 2: AI Annotation Audit")
        
        res_plotted = results.plot()
        st.image(res_plotted, caption='AI Predictions', channels="BGR")

        is_low_conf = any(box.conf < 0.85 for box in results.boxes)
        if is_low_conf or len(results.boxes) == 0:
            st.warning("⚠️ Result: Flagged for Human Review (Low Confidence)")
        else:
            st.success("✨ Result: Auto-Accepted (Gold Standard Label)")