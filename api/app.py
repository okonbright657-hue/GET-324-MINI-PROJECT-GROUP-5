"""
Actinic Keratosis vs Seborrheic Keratosis Classifier — Streamlit App
Loads the trained EfficientNetB0 transfer-learning model
(models/efficientnet_transfer_best.keras) and classifies uploaded
skin lesion images.

This is a student research project artifact, not a diagnostic tool.
See the disclaimer in the UI.
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
# Order matches training: tf.keras.utils.image_dataset_from_directory sorts
# class folders alphabetically -> index 0 = Actinic, index 1 = Seborrheic.
# Verify this against your own training run's printed `class_names` before
# trusting it — a flipped list here silently swaps every prediction.
CLASS_NAMES = ["Actinic Keratosis", "Seborrheic Keratosis"]
MODEL_PATH = "models/efficientnet_transfer_best.keras"
CENTROIDS_PATH = "models/class_centroids.npy"
# Cosine distance beyond which an image is rejected as "not a lesion this
# model recognizes." This value is a placeholder — tune it against your own
# held-out sample of known-good lesion images and known-garbage images
# (see notebook OOD cell) before trusting it. Don't ship a guessed number.
OOD_THRESHOLD = 0.35

# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Lesion Classifier",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #f7fafd 0%, #eef4f9 100%);
    }

    .hero {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
        background: linear-gradient(135deg, #1e5b8a 0%, #4a90c4 100%);
        border-radius: 18px;
        margin-bottom: 1.0rem;
        box-shadow: 0 8px 24px rgba(30, 91, 138, 0.25);
    }
    .hero h1 {
        color: white;
        font-weight: 700;
        font-size: 2.0rem;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #eaf3fb;
        font-size: 1.0rem;
        margin: 0;
    }

    .disclaimer {
        background: #fff8e6;
        border: 1px solid #e8c468;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        font-size: 0.88rem;
        color: #6b5117;
        margin-bottom: 1.4rem;
        line-height: 1.4;
    }

    .upload-card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        margin-bottom: 1.4rem;
        border: 1px solid #eef2ea;
    }

    .result-card {
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .result-actinic {
        background: linear-gradient(135deg, #fdeceb 0%, #fbdbd8 100%);
        border: 2px solid #e53935;
    }
    .result-seborrheic {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4e9f2 100%);
        border: 2px solid #3d8fb0;
    }
    .result-label {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-actinic .result-label { color: #c62828; }
    .result-seborrheic .result-label { color: #21617a; }

    .result-note {
        font-size: 0.82rem;
        color: #555;
        margin-top: 0.4rem;
    }

    .confidence-text {
        font-size: 1.0rem;
        color: #444;
        margin-top: 0.3rem;
    }

    .ood-card {
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        margin-top: 1rem;
        background: #f2f2f2;
        border: 2px dashed #999;
        color: #444;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model + centroid loading (cached so it only loads once per session) ────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading reference embeddings...")
def load_centroids():
    return np.load(CENTROIDS_PATH, allow_pickle=True).item()


@st.cache_resource(show_spinner=False)
def load_embedding_model(_model):
    # Look up the pooling layer by type, not by name — the auto-generated
    # Keras layer name (e.g. "global_average_pooling2d_2") depends on how
    # many times the model was rebuilt in the training session and will
    # not match a hardcoded string reliably. Match by type instead.
    gap_layer = next(
        l for l in _model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)
    )
    return tf.keras.Model(_model.input, gap_layer.output)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize + array-ify. No manual normalization here — the model applies
    tf.keras.applications.efficientnet.preprocess_input internally as a
    layer step (see build_transfer_model in the training notebook).
    Normalizing here too would double-preprocess and wreck predictions,
    same failure mode as double-rescaling in the old apple app."""
    img = pil_image.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def predict(model, arr: np.ndarray):
    prob_seborrheic = float(model.predict(arr, verbose=0)[0][0])  # sigmoid output, class index 1
    pred_idx = int(prob_seborrheic >= 0.5)
    label = CLASS_NAMES[pred_idx]
    confidence = prob_seborrheic if pred_idx == 1 else 1 - prob_seborrheic
    return label, confidence, prob_seborrheic


def get_embedding(embedding_model, arr: np.ndarray) -> np.ndarray:
    return embedding_model.predict(arr, verbose=0)[0]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.write(
        "This app uses a transfer-learning model (EfficientNetB0 backbone) "
        "to classify a skin lesion image as **Actinic Keratosis** or "
        "**Seborrheic Keratosis**."
    )
    st.markdown("---")
    st.markdown("### Model details")
    st.write(f"- Input size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    st.write("- Architecture: EfficientNetB0 (frozen) + dense head")
    st.write("- Output: sigmoid (binary)")
    st.write("- Out-of-distribution check: cosine distance to class centroids")
    st.markdown("---")
    st.markdown("### Class reference")
    st.write("- **Actinic Keratosis**: precancerous lesion, warrants clinical follow-up")
    st.write("- **Seborrheic Keratosis**: benign lesion")

# ── Hero header ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>Skin Lesion Classifier</h1>
        <p>Upload a lesion image to classify it as Actinic or Seborrheic Keratosis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Disclaimer ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="disclaimer">
    <strong>Not a diagnostic tool.</strong> This model was trained on a small
    academic dataset for a course project and has not been clinically
    validated. It distinguishes only two lesion types and will produce a
    confident-looking answer even for images of something else entirely —
    an out-of-distribution check below tries to catch that, but is not
    guaranteed. Do not use this to make, delay, or avoid a medical decision.
    If you have a concern about a skin lesion, see a dermatologist.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Upload card ───────────────────────────────────────────────────────────
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a lesion image",
    type=["jpg", "jpeg", "png"],
    help="JPG or PNG. Best results with a clear, well-lit, close-up dermoscopic-style photo.",
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Prediction flow ───────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        model = load_model()
        centroids = load_centroids()
    except Exception as e:
        st.error(
            f"Couldn't load the model or centroid file. Make sure both "
            f"`{MODEL_PATH}` and `{CENTROIDS_PATH}` exist in your deployment.\n\n"
            f"Details: {e}"
        )
        st.stop()

    embedding_model = load_embedding_model(model)

    with st.spinner("Analysing image..."):
        arr = preprocess_image(image)
        label, confidence, prob_seborrheic = predict(model, arr)
        embedding = get_embedding(embedding_model, arr)
        min_dist = min(cosine_distance(embedding, c) for c in centroids.values())

    with col2:
        if min_dist > OOD_THRESHOLD:
            st.markdown(
                f"""
                <div class="ood-card">
                    <strong>No prediction shown</strong>
                    <div class="result-note">
                        This image doesn't resemble the lesion types this
                        model was trained on (distance {min_dist:.2f} &gt;
                        threshold {OOD_THRESHOLD}).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            card_class = "result-actinic" if label == "Actinic Keratosis" else "result-seborrheic"
            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="result-label">{label}</div>
                    <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
                    <div class="result-note">Model output only — not a clinical diagnosis.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.progress(confidence)
            st.caption(f"Raw model output (P[Seborrheic Keratosis]) = {prob_seborrheic:.4f}")

else:
    st.info("Upload a lesion image to get a prediction.")

st.markdown("---")
st.caption("Built with TensorFlow + Streamlit — EfficientNetB0 transfer learning, academic project.")