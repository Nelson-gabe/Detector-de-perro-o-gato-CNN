"""Definición de la CNN para clasificar perros y gatos."""
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (160, 160)
NUM_CLASSES = 2
CLASS_NAMES = ["gato", "perro"]

def build_model(input_shape=(160, 160, 3)):
    """CNN compacta con aumento de datos integrado."""
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="data_augmentation")

    inputs = layers.Input(shape=input_shape)
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="cnn_gatos_perros")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
