import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognition", page_icon="✍️")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0-9) in the canvas below.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.h5", compile=False)

model = load_model()

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:

    img = Image.fromarray(
        canvas_result.image_data.astype(np.uint8)
    ).convert("L")

    img = img.resize((28, 28))

    img_array = np.array(img)

    img_array = 255 - img_array
    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array, verbose=0)
    digit = np.argmax(prediction)

    st.image(img, caption="Processed Image (28×28)", width=150)
    st.success(f"Predicted Digit: {digit}")