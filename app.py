"""
Streamlit app for Pneumonia Detection from Chest X-ray images.

It loads the trained model from `cnn_model.pkl` (Keras JSON + weights) and
predicts whether an uploaded image is NORMAL or PNEUMONIA.
"""

import io
import os
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import model_from_json


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load inference artifacts from cnn_model.pkl.

    The pickle file should contain:
      - model_json
      - weights
      - img_size
      - labels
    """
    if not os.path.exists("cnn_model.pkl"):
        raise FileNotFoundError("cnn_model.pkl not found next to app.py")

    with open("cnn_model.pkl", "rb") as f:
        payload = pickle.load(f)

    model = model_from_json(payload["model_json"])
    model.set_weights(payload["weights"])

    # Compile for predict/evaluate
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["accuracy"],
    )

    img_size = payload.get("img_size", 256)
    labels = payload.get("labels", ["NORMAL", "PNEUMONIA"])
    return model, img_size, labels


def preprocess_image(file_bytes: bytes, img_size: int):
    """Convert uploaded image bytes to normalized tensor expected by the model."""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale
    img_resized = img.resize((img_size, img_size))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = img_array.reshape(1, img_size, img_size, 1)
    return tensor, img_resized


def predict(image_tensor: np.ndarray, model):
    """Run inference and return probability of pneumonia."""
    preds = model.predict(image_tensor)
    return float(preds[0][0])


def main():
    st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="centered")
    st.title("Pneumonia Detection from Chest X-ray")
    st.write(
        "Upload a chest X-ray image (JPG/PNG). The model will predict whether it "
        "is NORMAL or PNEUMONIA."
    )

    try:
        model, img_size, labels = load_artifacts()
    except FileNotFoundError:
        st.error("Model file `cnn_model.pkl` not found next to `app.py`.")
        st.info("Train the model in the notebook and ensure `cnn_model.pkl` is copied here.")
        return
    except Exception as e:  # pragma: no cover - displayed to user
        st.error(f"Failed to load model: {e}")
        return

    uploaded = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = uploaded.read()
        try:
            image_tensor, preview_img = preprocess_image(file_bytes, img_size)
        except Exception as e:  # pragma: no cover - displayed to user
            st.error(f"Could not process image: {e}")
            return

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(preview_img, caption="Input image (grayscale)", use_column_width=True, clamp=True)

        if st.button("Predict"):
            with st.spinner("Running inference..."):
                prob_pneumonia = predict(image_tensor, model)
                prob_normal = 1.0 - prob_pneumonia
                pred_idx = int(prob_pneumonia > 0.5)
                pred_label = labels[pred_idx] if pred_idx < len(labels) else "PNEUMONIA"
                result_text = "Positive" if pred_idx == 1 else "Negative"
                result_detail = (
                    "Positive (Pneumonia detected)" if pred_idx == 1 else "Negative (Normal)"
                )

            with col2:
                st.metric("Result", result_detail)
                st.caption(f"Binary result: {result_text}")
                st.write(f"Class label: **{pred_label}**")
                st.write(f"Probability Pneumonia: **{prob_pneumonia:.3f}**")
                st.write(f"Probability Normal: **{prob_normal:.3f}**")

            st.info(
                "This tool is for research/education only and not for clinical use. "
                "Always consult a medical professional for diagnosis."
            )


if __name__ == "__main__":
    main()
