import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="ClearLabel AI", layout="centered")

st.title("🚀 ClearLabel AI")
st.subheader("Automated Annotation & Data Quality Audit Pipeline")
st.write("Upload an image to run the AI Auto-Annotation and Quality Audit.")

# -------------------------------
# Safe OpenCV Import
# -------------------------------
@st.cache_resource
def load_cv2():
    try:
        import cv2
        return cv2
    except Exception as e:
        st.error(f"❌ OpenCV failed to load: {e}")
        st.stop()

cv2 = load_cv2()

# -------------------------------
# Load YOLO Model Safely
# -------------------------------
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"❌ YOLO model failed to load: {e}")
        st.stop()

model = load_model()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # -------------------------------
        # Image Processing
        # -------------------------------
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # -------------------------------
        # Layer 1: Quality Audit
        # -------------------------------
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        st.subheader("🔍 Layer 1: Quality Audit")

        if variance < 100:
            st.error(f"🚩 Image Rejected: Too Blurry (Variance: {variance:.2f})")
            st.stop()
        else:
            st.success(f"✅ Image Passed Quality Check (Variance: {variance:.2f})")

        # -------------------------------
        # Layer 2: YOLO Inference
        # -------------------------------
        st.subheader("🤖 Layer 2: AI Annotation Audit")

        results = model(img_cv)[0]

        # Safe plotting
        res_plotted = results.plot()
        st.image(res_plotted, caption="AI Predictions", channels="BGR")

        # -------------------------------
        # Confidence Check (SAFE)
        # -------------------------------
        if results.boxes is None or len(results.boxes) == 0:
            st.warning("⚠️ No objects detected → Needs Human Review")
        else:
            confidences = [float(box.conf) for box in results.boxes]

            low_conf = any(conf < 0.85 for conf in confidences)

            if low_conf:
                st.warning("⚠️ Flagged for Human Review (Low Confidence)")
            else:
                st.success("✨ Auto-Accepted (Gold Standard Label)")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")