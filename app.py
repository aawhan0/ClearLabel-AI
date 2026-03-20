import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import os
import pandas as pd

st.set_page_config(page_title="ClearLabel AI", layout="centered")

st.title("🚀 ClearLabel AI")
st.markdown("### 🧪 AI-Powered Data Validation & Annotation System")
st.write("Upload an image to run automated quality checks and AI annotation.")

st.markdown("---")

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

        if not os.path.exists("yolov8n.pt"):
            st.error("❌ Model file 'yolov8n.pt' not found.")
            st.stop()

        return YOLO("yolov8n.pt")

    except Exception as e:
        st.error(f"❌ YOLO failed to load: {e}")
        st.stop()

model = load_model()

# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.markdown("---")

        # -------------------------------
        # QUALITY CHECK (USER-FRIENDLY)
        # -------------------------------
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        st.subheader("🔍 Image Quality Check")

        col1, col2 = st.columns(2)

        col1.metric("Image Clarity", "Good ✅" if variance >= 100 else "Poor ❌")
        col2.metric("Score", f"{variance:.0f}")

        st.caption("Clarity is based on image sharpness. Blurry images reduce AI accuracy.")

        if variance < 100:
            st.error("🚩 This image is too blurry for reliable AI detection.")
            st.info("💡 Try uploading a sharper image with better focus.")
            st.stop()
        else:
            st.success("✅ This image is clear enough for accurate AI detection.")

        # -------------------------------
        # AI DETECTION
        # -------------------------------
        st.subheader("🤖 AI Detection")

        results = model(img_cv)[0]
        res_plotted = results.plot()

        # -------------------------------
        # IMAGE COMPARISON
        # -------------------------------
        st.subheader("🖼️ Image Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with col2:
            st.image(res_plotted, caption="AI Prediction", channels="BGR", use_column_width=True)

        st.caption("Left: Input Image | Right: AI Annotated Output")

        st.markdown("---")

        # -------------------------------
        # PROCESS RESULTS
        # -------------------------------
        confidences = []
        labels = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                conf = float(box.conf)
                cls = int(box.cls)
                label = model.names[cls]

                confidences.append(conf)
                labels.append(label)

        # -------------------------------
        # SUMMARY
        # -------------------------------
        st.subheader("📊 Summary")

        col1, col2 = st.columns(2)

        col1.metric("Objects Detected", len(confidences))
        avg_conf = sum(confidences)/len(confidences) if confidences else 0
        col2.metric("Avg Confidence", f"{avg_conf:.2f}")

        # -------------------------------
        # CONFIDENCE GRAPH
        # -------------------------------
        if confidences:
            df = pd.DataFrame({
                "Object": labels,
                "Confidence": confidences
            })

            st.bar_chart(df.set_index("Object"), height=250)

        # -------------------------------
        # DECISION
        # -------------------------------
        st.markdown("---")
        st.subheader("📌 Final Decision")

        threshold = 0.85

        if len(confidences) == 0:
            st.warning("⚠️ No detections → Needs Review")
        elif any(conf < threshold for conf in confidences):
            st.warning("⚠️ Needs Human Review (Low Confidence)")
        else:
            st.success("✅ Auto Approved (High Confidence)")

        st.caption(f"Confidence threshold: {threshold}")

        # -------------------------------
        # DOWNLOAD RESULT
        # -------------------------------
        buffer = BytesIO()
        Image.fromarray(res_plotted).save(buffer, format="PNG")

        st.download_button(
            label="📥 Download Annotated Image",
            data=buffer.getvalue(),
            file_name="annotated.png",
            mime="image/png"
        )

        # -------------------------------
        # EXPLANATION
        # -------------------------------
        with st.expander("ℹ️ How it works"):
            st.write("""
            • Checks if the image is clear enough  
            • Runs object detection using AI (YOLOv8)  
            • Measures confidence of predictions  
            • Flags uncertain results for human review  
            """)

    except Exception as e:
        st.error(f"❌ Error: {e}")