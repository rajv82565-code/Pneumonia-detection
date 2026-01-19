"""
Streamlit app for Pneumonia Detection from Chest X-ray images.

It loads the trained model from a small `cnn_model.tflite` (recommended) or
falls back to `cnn_model.keras`, and predicts whether an uploaded image is
NORMAL or PNEUMONIA.
"""

import io
from typing import Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import os


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load inference artifacts.

    Prefer `cnn_model.tflite` (small, GitHub-friendly). Fallback to `cnn_model.keras`.
    """
    img_size = 256
    labels = ["NORMAL", "PNEUMONIA"]

    if os.path.exists("cnn_model.tflite"):
        interpreter = tf.lite.Interpreter(model_path="cnn_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return ("tflite", interpreter, input_details, output_details, img_size, labels)

    if os.path.exists("cnn_model.keras"):
        model = tf.keras.models.load_model("cnn_model.keras")
        return ("keras", model, None, None, img_size, labels)

    raise FileNotFoundError("Model not found. Expected `cnn_model.tflite` or `cnn_model.keras`.")


def preprocess_image(file_bytes: bytes, img_size: int):
    """Convert uploaded image bytes to normalized tensor expected by the model."""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")  # grayscale
    img_resized = img.resize((img_size, img_size))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = img_array.reshape(1, img_size, img_size, 1)
    return tensor, img_resized


def predict(image_tensor: np.ndarray, artifacts):
    """Run inference and return probability of pneumonia."""
    kind = artifacts[0]
    if kind == "keras":
        model = artifacts[1]
        preds = model.predict(image_tensor)
        return float(preds[0][0])

    # TFLite
    interpreter, input_details, output_details = artifacts[1], artifacts[2], artifacts[3]
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    interpreter.set_tensor(input_index, image_tensor.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    return float(output[0][0])


def main():
    st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="centered")
    st.title("Pneumonia Detection from Chest X-ray")
    st.write(
        "Upload a chest X-ray image (JPG/PNG). The model will predict whether it "
        "is NORMAL or PNEUMONIA."
    )

    try:
        kind, obj, input_details, output_details, img_size, labels = load_artifacts()
    except FileNotFoundError:
        st.error("Model not found. Put `cnn_model.tflite` (recommended) or `cnn_model.keras` next to `app.py`.")
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
                artifacts = (kind, obj, input_details, output_details)
                prob_pneumonia = predict(image_tensor, artifacts)
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
