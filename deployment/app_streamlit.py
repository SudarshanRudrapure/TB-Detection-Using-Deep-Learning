# deployment/app_streamlit.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image
from src.inference import load_model, predict_and_cam, CLASS_NAMES

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Hybrid Deep Learning System for TB Detection",
    layout="wide",
    page_icon="🩻"
)

# -------------------- STYLING --------------------
st.markdown("""
    <style>
        /* Animated background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #dbeafe, #f0f9ff, #e0f2fe);
            background-size: 400% 400%;
            animation: moveBg 10s ease infinite;
        }
        @keyframes moveBg {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* Main Title */
        .hero-title {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            font-size: 2.9rem;
            font-weight: 900;
            color: #1e3a8a;
            letter-spacing: -0.5px;
            padding-top: 0.6em;
        }
        .hero-icon {
            font-size: 2.8rem;
        }
        .hero-sub {
            text-align: center;
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 2.3em;
        }

        /* Prediction Card */
        .result-card {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 30px 20px;
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.15);
            margin-top: 15px;
        }

        /* Prediction Line */
        .prediction-line {
            font-size: 1.7rem;
            font-weight: 800;
            color: #1e40af;
            text-align: center;
            margin-bottom: 1rem;
        }

        /* Probability text */
        .prob-text {
            font-size: 1.15rem;
            font-weight: 600;
            color: #1f2937;
            margin-left: 0.3em;
            line-height: 1.6em;
        }

        /* Images */
        img {
            border-radius: 14px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        img:hover {
            transform: scale(1.02);
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.25);
        }

        /* Captions */
        .img-caption {
            text-align: center;
            font-size: 1.1rem;
            font-weight: 700;
            color: #1e293b;
            margin-top: 0.5em;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e0f7fa 0%, #f0f9ff 100%);
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #64748b;
            margin-top: 40px;
            border-top: 1px solid #cbd5e1;
            padding-top: 10px;
        }

        /* Remove default white background */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def get_model():
    return load_model()

# -------------------- MAIN APP --------------------
def main():
    # Sidebar Inf
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1051/1051262.png", width=80)
        st.title("🧬 About Project")
        st.markdown("""
        **Hybrid Deep Learning System**  
        Combines **CNN + Transformer (DeiT)** for  
        advanced tuberculosis detection.

        - 🩺 3 Classes: Healthy, Non-TB, TB  
        - 🧠 Uses Grad-CAM for explainability  
        - ⚙️ Framework: Grad-CAM, Streamlit  
        """)
        st.markdown("---")
        st.markdown("👨‍💻 *Developed by Sudarshan Rudrapure*")

    # Header with icon beside text
    st.markdown("""
        <div class='hero-title'>
            <span class='hero-icon'>🫁</span> Hybrid Deep Learning System for TB Detection
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Upload a Chest X-ray to classify as Healthy, Non-TB, or TB and visualize model insights using Grad-CAM.</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("📤 Upload Chest X-ray", type=["png", "jpg", "jpeg"])

    if uploaded:
        image_pil = Image.open(uploaded).convert("RGB")

        # Layout: three columns
        col1, col2, col3 = st.columns([1.2, 1.6, 1.2])

        with col1:
            st.image(image_pil, use_container_width=True)
            st.markdown("<div class='img-caption'>🩺 Uploaded X-ray</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            model = get_model()
            pred_name, probs, cam_path = predict_and_cam(image_pil, model=model)

            # Prediction Line
            st.markdown(f"<div class='prediction-line'>🔍 Model Prediction: {pred_name}</div>", unsafe_allow_html=True)

            # Probabilities with bars
            for i, n in enumerate(CLASS_NAMES):
                st.markdown(f"<span class='prob-text'>{n}: {probs[i]*100:.2f}%</span>", unsafe_allow_html=True)
                st.progress(float(probs[i]))

            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            cam_img = Image.open(cam_path)
            st.image(cam_img, use_container_width=True)
            st.markdown("<div class='img-caption'>🫁 Grad-CAM Heatmap</div>", unsafe_allow_html=True)

    else:
        st.info("📁 Please upload a Chest X-ray to begin analysis.")

    # Footer
    # st.markdown("<div class='footer'>💡 Built with using Grad-CAM, Streamlit | © 2025 Hybrid TB Detection Project</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
