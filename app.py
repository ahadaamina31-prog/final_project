# Import required libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained CNN model
model = tf.keras.models.load_model("digit_model.h5", compile=False)

# Title of the application
st.title("Handwritten Digit Recognition System")

# Project description
st.write("Upload an image of a handwritten digit (0–9) to predict the digit using a CNN model.")

# Upload image file
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg"]
)

# Check whether a file is uploaded
if uploaded_file is not None:

    # Open the image and convert it to grayscale
    image = Image.open(uploaded_file).convert("L")

    # Resize image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert image into numpy array
    img_array = np.array(image)

    # Invert image colors
    img_array = 255 - img_array

    # Normalize pixel values
    img_array = img_array / 255.0

    # Reshape image for CNN model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict the digit
    prediction = model.predict(img_array)

    # Get predicted digit
    predicted_digit = np.argmax(prediction)

    # Get prediction confidence
    confidence = np.max(prediction) * 100

    # Display uploaded image
    st.image(image, caption="Uploaded Image", width=150)

    # Display prediction result
    st.success(f"Predicted Digit: {predicted_digit}")

    # Display confidence score
    st.info(f"Prediction Confidence: {confidence:.2f}%")