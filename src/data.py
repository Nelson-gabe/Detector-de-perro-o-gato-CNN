"""Carga y preparación reproducible del dataset Cats vs Dogs mediante TFDS."""
import tensorflow as tf
import tensorflow_datasets as tfds
from src.model import IMG_SIZE

SEED = 42
BATCH_SIZE = 32

def prepare_datasets():
    (train_raw, val_raw, test_raw), info = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )
    def prepare(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        return image, tf.cast(label, tf.int32)

    options = tf.data.Options()
    options.experimental_deterministic = True
    train = (train_raw.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
             .shuffle(2000, seed=SEED).batch(BATCH_SIZE)
             .prefetch(tf.data.AUTOTUNE).with_options(options))
    val = (val_raw.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    test = (test_raw.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    return train, val, test, info
