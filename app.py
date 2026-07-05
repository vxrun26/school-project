"""
app.py
------
Bone Fracture Detection AI - Streamlit dashboard.

Run with:
    streamlit run app.py

This file is the FRONTEND + BACKEND in one (that is the Streamlit model).
It stays focused on UI/flow and delegates the real work to:
    - utils.py  : image loading & enhancement
    - model.py  : AI inference (Hugging Face API + offline fallback)
"""

import streamlit as st

from model import HF_MODEL_ID, analyze_image, get_hf_token
from utils import ALLOWED_EXTENSIONS, enhance_xray, load_image


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Bone Fracture Detection AI",
    page_icon="🦴",
    layout="wide",
)


# ----------------------------------------------------------------------
# Sidebar: educational content + status (great for project submission)
# ----------------------------------------------------------------------
def render_sidebar() -> None:
    st.sidebar.title("🦴 About This Project")
    st.sidebar.write(
        "An AI-assisted tool that examines bone X-ray images and predicts "
        "whether a **fracture** is present."
    )

    st.sidebar.subheader("🤖 How AI helps in radiology")
    st.sidebar.markdown(
        "- **Speed:** screens images in seconds, helping prioritize urgent cases.\n"
        "- **Consistency:** does not get tired, so it gives steady second opinions.\n"
        "- **Access:** brings basic screening to areas with few radiologists.\n"
        "- **Support, not replacement:** doctors make the final call; AI just assists."
    )

    st.sidebar.subheader("🔌 AI engine status")
    if get_hf_token():
        st.sidebar.success("Hugging Face API token detected (online AI mode).")
    else:
        st.sidebar.info(
            "No API token set - running in **offline demo mode** "
            "using lightweight OpenCV analysis."
        )
    st.sidebar.caption(f"Model: `{HF_MODEL_ID}`")

    st.sidebar.warning(
        "⚠️ Educational school project only. This is **not** a medical "
        "device and must not be used for real diagnosis."
    )


# ----------------------------------------------------------------------
# Results renderer
# ----------------------------------------------------------------------
def render_result(result: dict) -> None:
    label = result["label"]
    confidence = result["confidence"]

    st.subheader("🔬 Analysis Result")

    if label == "Fracture Detected":
        st.error(f"### 🚨 {label}")
    else:
        st.success(f"### ✅ {label}")

    # Confidence as a metric + visual progress bar.
    st.metric(label="Confidence", value=f"{confidence:.2f}%")
    st.progress(min(int(confidence), 100))

    st.caption(f"Prediction source: {result['source']}")
    if result.get("note"):
        st.info(result["note"])

    with st.expander("See raw model output"):
        st.write(result.get("raw"))


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------
def main() -> None:
    render_sidebar()

    st.title("Bone Fracture Detection AI")
    st.write(
        "Upload a bone X-ray image and click **Scan Image** to check for a "
        "possible fracture. Built with Streamlit + a lightweight AI model."
    )

    uploaded_file = st.file_uploader(
        "Upload an X-ray image",
        type=ALLOWED_EXTENSIONS,
        help="Accepted formats: JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        st.info("👆 Upload an X-ray image to get started.")
        return

    # Load once, then show the original and an enhanced version side by side.
    image = load_image(uploaded_file)
    enhanced = enhance_xray(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original X-ray", use_column_width=True)
    with col2:
        st.image(enhanced, caption="Contrast-enhanced (CLAHE)", use_column_width=True)

    # The action button that triggers inference.
    if st.button("🔍 Scan Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing X-ray..."):
            result = analyze_image(image)
        render_result(result)


if __name__ == "__main__":
    main()
