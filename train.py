"""Entrena la CNN, guarda el modelo, gráficas y métricas de evaluación."""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.model import build_model, CLASS_NAMES
from src.data import prepare_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", default="models/cats_dogs_cnn.keras")
    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    train_ds, val_ds, test_ds, info = prepare_datasets()
    print("Dataset:", info.name, "| Clases:", CLASS_NAMES)
    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
            patience=2, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=callbacks)
    model.save(args.output)
    with open("reports/history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # Curvas de aprendizaje
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Entrenamiento")
    axes[0].plot(history.history["val_accuracy"], label="Validación")
    axes[0].set_title("Exactitud"); axes[0].set_xlabel("Época"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="Entrenamiento")
    axes[1].plot(history.history["val_loss"], label="Validación")
    axes[1].set_title("Pérdida"); axes[1].set_xlabel("Época"); axes[1].legend()
    fig.tight_layout(); fig.savefig("reports/figures/curvas_entrenamiento.png", dpi=150); plt.close(fig)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión"); fig.tight_layout()
    fig.savefig("reports/figures/matriz_confusion.png", dpi=150); plt.close(fig)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    with open("reports/metricas.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print(f"Modelo guardado en: {args.output}")

if __name__ == "__main__":
    main()
