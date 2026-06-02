import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

st.title("✍️ Draw a Digit Recognition App (0-9)")
st.write("Draw a digit in the box below")

# Create drawing canvas
canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# If user draws something
if canvas_result.image_data is not None:

    # Convert canvas image
    img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    img = img.convert("L")  # grayscale
    img = img.resize((28, 28))

    # Convert to array
    img_array = np.array(img)

    # Normalize + invert (important for MNIST model)
    img_array = 255 - img_array
    img_array = img_array / 255.0

    # Reshape for model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    # Show results
    st.image(img, caption="Processed Image (28x28)", width=150)
    st.success(f"Predicted Digit: {digit}")