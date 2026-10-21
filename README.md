# Clasificación de perros y gatos con una CNN

Proyecto de Deep Learning para clasificar fotografías en dos clases: **gato** y **perro**. Incluye código reproducible, notebook de experimentación para Google Colab, entrenamiento, evaluación, predicción por terminal y aplicación Streamlit.

> **Importante sobre el modelo:** el archivo binario entrenado no se incluye porque depende de ejecutar el entrenamiento sobre Cats vs Dogs. El script y el notebook lo generan en `models/cats_dogs_cnn.keras`. La aplicación requiere ese archivo y muestra instrucciones si aún no existe. No se presenta un modelo sin entrenar como si estuviera entrenado.

## Dataset

Se usa **Cats vs Dogs** a través de TensorFlow Datasets (`cats_vs_dogs`). Contiene fotografías etiquetadas de gatos y perros. La descarga inicial puede tardar y requiere conexión a internet. TFDS puede descartar imágenes corruptas durante su preparación.

## Estructura

```text
clasificacion_perros_gatos_cnn/
├── app/streamlit_app.py
├── notebooks/experimento_cnn_perros_gatos.ipynb
├── src/
│   ├── model.py
│   └── data.py
├── models/                  # aquí se guarda el modelo entrenado
├── reports/
│   ├── figures/             # curvas y matriz de confusión
│   ├── history.json
│   └── metricas.json
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Opción A: ejecutar todo en Google Colab

1. Sube el repositorio a GitHub o carga el ZIP en Google Drive.
2. Abre `notebooks/experimento_cnn_perros_gatos.ipynb` en Colab.
3. En **Entorno de ejecución → Cambiar tipo de entorno de ejecución**, selecciona GPU si está disponible.
4. Ejecuta las celdas en orden. El notebook descarga el dataset, explora imágenes, prepara los conjuntos, entrena y evalúa la CNN, genera gráficas y muestra al menos 10 predicciones.
5. Descarga o copia al repositorio el modelo generado y las salidas que quieras versionar. El modelo puede ser grande; Git LFS o un release de GitHub puede ser más apropiado.

## Opción B: ejecutar localmente

Se recomienda Python 3.10–3.11. Desde la raíz del proyecto:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python train.py --epochs 12
```

El entrenamiento crea:
- `models/cats_dogs_cnn.keras`
- `reports/history.json`
- `reports/metricas.json`
- `reports/figures/curvas_entrenamiento.png`
- `reports/figures/matriz_confusion.png`

Para probar una imagen:

```bash
python predict.py ruta/a/imagen.jpg
```

Para iniciar Streamlit después de entrenar:

```bash
streamlit run app/streamlit_app.py
```

## Metodología

- **Tamaño de entrada:** 160 × 160 píxeles, RGB.
- **Normalización:** conversión a `float32`; la capa `Rescaling(1/255)` escala píxeles al intervalo [0, 1].
- **Partición:** 80% entrenamiento, 10% validación y 10% prueba del split de TFDS.
- **Data augmentation:** volteo horizontal, rotación leve, zoom y contraste, aplicado durante entrenamiento.
- **CNN:** bloques convolucionales con 32, 64 y 128 filtros, Batch Normalization, Max Pooling, Global Average Pooling y capas Dropout.
- **Optimizador y pérdida:** Adam y sparse categorical cross-entropy.
- **Métricas:** accuracy, precisión, recall y F1 por clase; matriz de confusión.
- **Control de entrenamiento:** checkpoint de mejor `val_accuracy`, early stopping y reducción de learning rate.

## Interpretación responsable

La confianza es la salida softmax del modelo y no equivale a una probabilidad calibrada. El rendimiento real debe informarse con las métricas obtenidas en el conjunto de prueba. Evita usar las métricas de entrenamiento como estimación de generalización.

## Git

```bash
git init
git add .
git commit -m "Proyecto CNN para clasificar perros y gatos"
git branch -M main
git remote add origin https://github.com/USUARIO/clasificacion-perros-gatos-cnn.git
git push -u origin main
```

El directorio `models/` se incluye en el repositorio mediante un `.gitkeep`; el modelo se crea al entrenar. Para versionar un modelo grande, considera Git LFS.
