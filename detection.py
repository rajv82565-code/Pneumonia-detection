"""
Streamlit App for Pneumonia Detection from Chest X-Ray Images
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🏥",
    layout="wide"
)

# Constants
IMG_SIZE = 256
labels = ["NORMAL", "PNEUMONIA"]

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model("cnn_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure 'cnn_model.h5' is in the same directory as this app.")
        return None

def prepare_image(image):
    """
    Preprocess uploaded image for prediction
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Reshape for model input
    img_final = img_normalized.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    return img_final, img_resized

def predict(model, processed_image):
    """Make prediction on processed image"""
    prediction = model.predict(processed_image, verbose=0)
    prediction_class = int(prediction[0] > 0.5)
    confidence = prediction[0][0] if prediction_class == 1 else 1 - prediction[0][0]
    return prediction_class, confidence

# Main app
def main():
    # Title and description
    st.title("🏥 Pneumonia Detection from Chest X-Ray")
    st.markdown("""
    This application uses a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images.
    Upload a chest X-ray image to get a prediction.
    """)
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    **How to use:**
    1. Upload a chest X-ray image (JPG, JPEG, PNG)
    2. Wait for the model to process
    3. View the prediction results
    
    **Model Information:**
    - Architecture: 4-layer CNN
    - Input size: 256x256 pixels
    - Classes: NORMAL, PNEUMONIA
    """)
    
    st.sidebar.header("Model Architecture")
    st.sidebar.code("""
    Conv2D(32) -> MaxPool -> Dropout
    Conv2D(64) -> MaxPool -> Dropout
    Conv2D(128) -> MaxPool -> Dropout
    Conv2D(256) -> MaxPool -> Dropout
    Flatten -> Dense(256) -> Dense(1)
    """, language="text")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    st.header("Upload Chest X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Process and predict
        with st.spinner("Analyzing X-ray image..."):
            processed_image, resized_image = prepare_image(image)
            prediction_class, confidence = predict(model, processed_image)
        
        with col2:
            st.subheader("Processed Image (Grayscale)")
            st.image(resized_image, use_container_width=True, clamp=True, channels="GRAY")
        
        # Display results
        st.header("Prediction Results")
        
        # Create result columns
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("Prediction", labels[prediction_class])
        
        with res_col2:
            st.metric("Confidence", f"{confidence * 100:.2f}%")
        
        with res_col3:
            if prediction_class == 1:
                st.error("⚠️ PNEUMONIA DETECTED")
            else:
                st.success("✅ NORMAL")
        
        # Confidence bar
        st.subheader("Confidence Breakdown")
        conf_data = {
            "NORMAL": (1 - confidence) * 100 if prediction_class == 1 else confidence * 100,
            "PNEUMONIA": confidence * 100 if prediction_class == 1 else (1 - confidence) * 100
        }
        
        # Create a simple bar chart
        fig, ax = plt.subplots(figsize=(10, 2))
        colors = ['green' if k == 'NORMAL' else 'red' for k in conf_data.keys()]
        bars = ax.barh(list(conf_data.keys()), list(conf_data.values()), color=colors, alpha=0.7)
        ax.set_xlabel('Confidence (%)')
        ax.set_xlim(0, 100)
        
        # Add percentage labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}%', 
                   ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Medical disclaimer
        st.warning("""
        **Medical Disclaimer:** This is an AI-assisted diagnostic tool and should NOT be used as the sole basis 
        for medical diagnosis. Always consult with qualified healthcare professionals for proper medical evaluation 
        and diagnosis.
        """)
        
        # Additional information
        with st.expander("ℹ️ Understanding the Results"):
            st.markdown("""
            - **NORMAL**: No signs of pneumonia detected in the X-ray
            - **PNEUMONIA**: Potential signs of pneumonia detected in the X-ray
            - **Confidence**: The model's certainty in its prediction (higher is more certain)
            
            **Important Notes:**
            - This model is trained on a specific dataset and may not generalize to all cases
            - False positives and false negatives are possible
            - Clinical context and additional tests are essential for accurate diagnosis
            """)
    
    else:
        # Show example when no file is uploaded
        st.info("👆 Please upload a chest X-ray image to begin analysis")
        
        # Example section
        st.header("Example Usage")
        st.markdown("""
        1. Click the **'Browse files'** button above
        2. Select a chest X-ray image from your computer
        3. The model will automatically analyze the image
        4. View the prediction and confidence score
        """)

if __name__ == "__main__":
    main()