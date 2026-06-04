# Import required libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model("digit_model.h5", compile=False)

st.title("✍️ Handwritten Digit Recognition System")

st.write("Draw a digit OR upload an image (0-9)")

# =========================
# DRAWING SECTION
# =========================
st.subheader("✍️ Draw Digit")

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# =========================
# UPLOAD SECTION
# =========================
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

# =========================
# PREDICTION FROM DRAWING
# =========================
if st.button("Predict from Drawing"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data[:, :, 0]
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((28, 28))
        img = np.array(img)

        img = 255 - img
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        digit = np.argmax(prediction)

        st.success(f"Predicted Digit (Drawing): {digit}")
        st.write(f"Confidence: {np.max(prediction)*100:.2f}%")

# =========================
# PREDICTION FROM UPLOAD
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    image = image.resize((28, 28))
    image_array = np.array(image)

    image_array = 255 - image_array
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit (Upload): {digit}")
    st.write(f"Confidence: {np.max(prediction)*100:.2f}%")