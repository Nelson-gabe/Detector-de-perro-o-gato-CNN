"""Predicción de una imagen desde la terminal."""
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from src.model import IMG_SIZE, CLASS_NAMES

def predict_image(image_path, model_path="models/cats_dogs_cnn.keras"):
    model = tf.keras.models.load_model(model_path)
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = model.predict(array, verbose=0)[0]
    idx = int(np.argmax(probabilities))
    return CLASS_NAMES[idx], float(probabilities[idx]), dict(zip(CLASS_NAMES, probabilities.tolist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imagen")
    parser.add_argument("--model", default="models/cats_dogs_cnn.keras")
    args = parser.parse_args()
    label, confidence, all_probs = predict_image(args.imagen, args.model)
    print(f"Predicción: {label} | Confianza estimada: {confidence:.1%}")
    print("Probabilidades:", all_probs)
