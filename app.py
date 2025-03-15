import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Set Image Size
IMG_SIZE = 224

# Load the saved model
MODEL_PATH = "malaria_best_model.keras"  # Change this to your actual model filename

@st.cache(allow_output_mutation=True)
def load_malaria_model():
    return load_model(MODEL_PATH)

model = load_malaria_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = np.array(image.convert("L"))  # Convert to grayscale
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0  # Resize and normalize
    image = np.expand_dims(image, axis=-1)  # Convert to (224, 224, 1)
    image = np.expand_dims(image, axis=0)  # Convert to (1, 224, 224, 1) for model input
    return image

# Streamlit UI
st.title("🦠 Malaria Cell Classification")
st.write("Upload an image of a malaria cell, and the model will classify it as **Parasitized** or **Uninfected**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make Prediction
    prediction = model.predict(processed_image)[0][0]
    confidence = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100

    # Classification Result
    if prediction < 0.5:
        result = "🦠 **Parasitized (Infected)**"
    else:
        result = "✅ **Uninfected**"

    # Display Prediction & Confidence Score
    st.subheader("🔍 Prediction Result")
    st.write(f"### {result}")
    st.write(f"**Confidence Score:** {confidence:.2f}%")