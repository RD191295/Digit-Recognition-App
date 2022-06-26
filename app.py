import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import streamlit_drawable_canvas as sdc

# app title
original_title = '<p style="font-family:Courier; color:Red; font-size: 40px;">DIGIT Recognizer</p>'
st.markdown(original_title, unsafe_allow_html=True)

canvas_result = sdc.st_canvas(
    stroke_width=20, stroke_color='#ffffff',
    background_color='#000000', height=200, width=200)

if st.button("Predict"):
    img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    model = tf.keras.models.load_model("digit_classification.h5")
    op = model.predict(img_gray.reshape(1, 28, 28))
    output = np.argmax(op)
    st.write(output)
