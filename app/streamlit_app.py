"""Interfaz web para clasificar imágenes de perros y gatos."""
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cats_dogs_cnn.keras"
CLASS_NAMES = ["gato", "perro"]
IMG_SIZE = (160, 160)

st.set_page_config(page_title="Clasificador de perros y gatos", page_icon="🐾")
st.title("🐶🐱 Clasificador de perros y gatos")
st.write("Carga una fotografía y la CNN estimará si contiene un gato o un perro.")
st.caption("La confianza es la probabilidad estimada por el modelo; no garantiza que la predicción sea correcta.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
if model is None:
    st.error("No se encontró el modelo entrenado.")
    st.code("python train.py --epochs 12", language="bash")
    st.stop()

uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)
    if st.button("Clasificar imagen", type="primary"):
        resized = image.resize(IMG_SIZE)
        arr = np.asarray(resized, dtype=np.float32)[None, ...]
        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        st.subheader(f"Clase predicha: {CLASS_NAMES[idx].capitalize()}")
        st.metric("Confianza estimada", f"{probs[idx]:.1%}")
        st.write("Probabilidad por clase")
        st.progress(float(probs[0]), text=f"Gato: {probs[0]:.1%}")
        st.progress(float(probs[1]), text=f"Perro: {probs[1]:.1%}")
