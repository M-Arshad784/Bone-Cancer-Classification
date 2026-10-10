import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bone X-ray AI",
    page_icon="🔬",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best.pt")

model = YOLO(model_path)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #080d19;
}

.block-container {
    max-width: 1150px;
    padding-top: 40px;
    padding-bottom: 50px;
}

/* Main title */

.main-title {
    text-align: center;
    color: white;
    font-size: 48px;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 35px;
}

/* Badge */

.badge-container {
    text-align: center;
    margin-bottom: 15px;
}

.badge {
    color: #7dd3fc;
    font-size: 14px;
    font-weight: 600;
}

/* Section */

.section-title {
    color: white;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}

.section-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.6;
}

/* Result */

.result-box {
    background-color: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}

.result-title {
    color: #94a3b8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-name {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin: 10px 0;
}

.result-confidence {
    color: #7dd3fc;
    font-size: 18px;
    font-weight: 700;
}

/* Disclaimer */

.disclaimer {
    background-color: #17130b;
    border: 1px solid #4a3a16;
    border-radius: 15px;
    padding: 20px;
    margin-top: 30px;
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="badge-container">'
    '<div class="badge">🧠 Deep Learning • Computer Vision</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🔬 Bone X-ray AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Bone X-ray Classification using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# TWO COLUMNS
# ============================================================

left_column, right_column = st.columns(
    [1.1, 0.9],
    gap="large"
)

# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:

    st.markdown(
        '<div class="section-title">📤 Upload Bone X-ray</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Upload a bone X-ray image and let the trained AI model '
        'analyze the image and predict the corresponding class.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    uploaded_image = st.file_uploader(
        "Choose Bone X-ray Image",
        type=["jpg", "jpeg", "png"]
    )

    st.caption(
        "Maximum file size: 200 MB • JPG, JPEG, PNG"
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Uploaded Bone X-ray",
            use_container_width=True
        )

# ============================================================
# RIGHT COLUMN
# ============================================================

with right_column:

    st.markdown(
        '<div class="section-title">🧠 AI Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'The trained deep learning model will analyze the '
        'uploaded X-ray and generate a classification prediction '
        'with confidence.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if uploaded_image is None:

        st.info(
            "📤 Upload an X-ray to start AI analysis."
        )

    else:

        if st.button(
            "🔍 Analyze X-ray",
            use_container_width=True
        ):

            with st.spinner("Analyzing X-ray..."):

                # Run prediction
                results = model(image)

                # Get first result
                result = results[0]

                # Get predicted class
                class_id = result.probs.top1

                # Get confidence
                confidence = result.probs.top1conf.item()

                # Get class name
                class_name = model.names[class_id]

            # ====================================================
            # DISPLAY RESULT
            # ====================================================

            st.write("")

            st.subheader("AI Prediction")

            st.markdown(
                f"### ✨ {class_name}"
            )

            st.markdown(
                f"**Confidence: {confidence * 100:.2f}%**"
            )

            st.progress(
                confidence,
                text=f"Model Confidence: {confidence * 100:.2f}%"
            )

# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ Medical Disclaimer\n\n"
    "This application is an AI-based educational and research "
    "project. The prediction should not be considered a medical "
    "diagnosis or a replacement for professional medical advice. "
    "Always consult a qualified healthcare professional for "
    "medical evaluation."
)

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Built with ❤️ using Python • Streamlit • Ultralytics YOLO"
)
