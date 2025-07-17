
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model("xray_model.h5")

st.title("Chest X-Ray Pneumonia Detection")
st.write("Upload a Chest X-ray image to predict if it's Normal or Pneumonia.")

uploaded_file = st.file_uploader("Choose a Chest X-ray image", type=["jpg", "jpeg", "png"])

def predict(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
    return label, confidence

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Chest X-ray", use_column_width=True)
    label, confidence = predict(img)
    st.write(f"### Prediction: {label}")
    st.write(f"### Confidence: {confidence * 100:.2f}%")
