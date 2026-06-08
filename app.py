import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.h5", compile=False)

model = load_model()
# TITLE

st.title("✍️ AI Digit Recognizer")
st.write("Draw OR upload an image of a digit (0-9)")
# DRAW SECTION
st.subheader("🎨 Draw Digit")

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)
# DRAW PREDICTION BUTTON
if st.button("🎨 Predict Drawing"):

    if canvas_result.image_data is not None:

        # Check if something is drawn
        if np.sum(canvas_result.image_data) > 0:

            img = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            ).convert("L")

            img = img.resize((28, 28))
            img = np.array(img)

            img = 255 - img
            img = img / 255.0
            img = img.reshape(1, 28, 28, 1)

            prediction = model.predict(img)
            digit = np.argmax(prediction)

            st.success(f"🎨 Drawing Prediction: {digit}")
            st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")

        else:
            st.warning("Please draw a digit first.")
# UPLOAD SECTION
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"]
)
# UPLOAD PREDICTION BUTTON
if st.button("📤 show "):

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=200)

        image = image.resize((28, 28))
        image = np.array(image)

        image = 255 - image
        image = image / 255.0
        image = image.reshape(1, 28, 28, 1)

        prediction = model.predict(image)
        digit = np.argmax(prediction)

        st.success(f"📤 Upload Prediction: {digit}")
        st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")

    else:
        st.warning("Please upload an image first.")