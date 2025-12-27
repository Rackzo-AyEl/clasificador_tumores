# Librerías
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import tensorflow as tf
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tqdm import tqdm


# ============================
# Preprocesamiento de imágenes
# ============================
# Carpetas originales
carpetas_origen = ['glioma', 'healthy', 'meningioma', 'pituitary']
etiquetas = {clase: i for i, clase in enumerate(carpetas_origen)}

# Carpetas destino
base_destino = './Features'
splits = ['Train', 'Validation', 'Test']
for split in splits:
    os.makedirs(os.path.join(base_destino, split), exist_ok=True)

# Dimensiones esperadas por ResNet50
dimensiones = (224, 224)


print("Procesando y separando imágenes para ResNet50...")
for clase in carpetas_origen:
    ruta_clase = f'./{clase}'
    imagenes = [img for img in os.listdir(ruta_clase)
                if img.lower().endswith(('.jpg'))]

    # Separar 70/15/15
    train_imgs, test_imgs = train_test_split(
        imagenes, test_size=0.30, random_state=42, stratify=None)
    val_imgs, test_imgs = train_test_split(
        test_imgs, test_size=0.50, random_state=42)

    separaciones = {
        'Train': train_imgs,
        'Validation': val_imgs,
        'Test': test_imgs
    }

    for split, lista_imgs in separaciones.items():
        X, y = [], []
        for img_nombre in tqdm(lista_imgs):
            ruta_imagen = os.path.join(ruta_clase, img_nombre)
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"No se pudo leer: {ruta_imagen}")
                continue

            # BGR → RGB y redimensionar
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen = cv2.resize(imagen, dimensiones)

            # Convertir a float32 y aplicar preprocesamiento de ResNet50
            imagen = imagen.astype(np.float32)
            # normaliza a [-123.68, 151.06, etc.]
            imagen = preprocess_input(imagen)

            X.append(imagen)
            y.append(etiquetas[clase])

        # Guardar como .npz
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        ruta_salida = os.path.join(base_destino, split, f"{clase}.npz")
        np.savez_compressed(ruta_salida, X=X, y=y)

        print(f"Guardado: {ruta_salida} ({X.shape[0]} imágenes)")

print("\nProcesamiento completado. Archivos listos para ResNet50.")


# ================================
# Vectorización de características
# ================================
# Intentar usar GPU
print("Dispositivos físicos:", tf.config.list_physical_devices())
print("GPUs:", tf.config.list_physical_devices("GPU"))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configurada correctamente")
    except RuntimeError as e:
        print(e)

# Configuración
splits = ['Train', 'Validation', 'Test']
clases = ['glioma', 'healthy', 'meningioma', 'pituitary']

features_dir = "./Features"
vectors_dir = "./Vectors"

# Crear carpetas destino
for split in splits:
    os.makedirs(os.path.join(vectors_dir, split), exist_ok=True)

# Modelo base ResNet50
print("\nCargando modelo base ResNet50...")
modelo_base = ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3))
modelo = Model(inputs=modelo_base.inputs,
               outputs=GlobalAveragePooling2D()(modelo_base.output))

print("Modelo cargado correctamente")


# Función de extracción
def extraer_features(model, X, batch_size=64):
    features = []
    for i in tqdm(range(0, len(X), batch_size), desc="Extrayendo"):
        batch = X[i:i+batch_size]
        feats = model.predict(batch, verbose=0)
        features.append(feats)
    return np.concatenate(features, axis=0)


# Procesar cada conjunto
for split in splits:
    print(f"\nProcesando conjunto: {split}")

    for clase in clases:
        ruta_npz = os.path.join(features_dir, split, f"{clase}.npz")

        if not os.path.exists(ruta_npz):
            print(f"No se encontró {ruta_npz}, se omite.")
            continue

        # Cargar datos
        data = np.load(ruta_npz)
        X, y = data["X"], data["y"]

        print(f"→ {clase}: {X.shape[0]} imágenes")

        # Extraer vectores (sin aplicar preprocess_input)
        vectores = extraer_features(modelo, X)

        # Guardar resultados
        ruta_salida = os.path.join(vectors_dir, split, f"{clase}.npz")
        np.savez_compressed(ruta_salida, X=vectores, y=y)

        print(f"Guardado: {ruta_salida} ({vectores.shape[0]} vectores)")

print("\nVectorización completa. Archivos guardados en ./Vectors/")


# ========================================
# Entrenamiento y clasificación de tumores
# ========================================
# Configuración directorios
base_dir = "./Vectors"
splits = ["Train", "Validation", "Test"]
clases = ["glioma", "healthy", "meningioma", "pituitary"]


# Función para cargar datos
def cargar_datos(directorio):
    X, y = [], []
    for idx, clase in enumerate(clases):
        ruta = os.path.join(base_dir, directorio, f"{clase}.npz")
        if not os.path.exists(ruta):
            print(f"No se encontró {ruta}, se omite.")
            continue
        data = np.load(ruta)
        X.append(data["X"])
        y.append(data["y"])
    return np.concatenate(X), np.concatenate(y)


# Cargar conjuntos
print("Cargando conjuntos...")
X_train, y_train = cargar_datos("Train")
X_val, y_val = cargar_datos("Validation")
X_test, y_test = cargar_datos("Test")

print(f"Conjuntos cargados:")
print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# Búsqueda de hiperparámetros
print("\nIniciando búsqueda de mejores hiperparámetros...")

# Crear instancia de random forest
# rf = RandomForestClassifier(random_state=42, n_jobs=-1)
# Mejores hiperparámetros
# mejor_rf = RandomForestClassifier(
#     max_depth=20,
#     min_samples_leaf=1,
#     min_samples_split=5,
#     n_estimators=200,
#     random_state=42,
#     n_jobs=-1
# )

# Diccionario de hiperparámetros de Random Forest
param_grid = {
    'max_depth': [None, 5, 10, 20],
    'n_estimators': [100, 150, 200],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3]
}

# Búsqueda de hiperparámetros de GridSearch
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    verbose=2,
    n_jobs=-1
)

# Entrenar grid GridSearch
grid.fit(X_train, y_train)

print(f"\nBúsqueda completa")
print(f"Mejores hiperparámetros: {grid.best_params_}")
print(f"Mejor score de accuracy: {grid.best_score_}")

# Obtener mejor modelo con grid search
mejor_rf = grid.best_estimator_

# Entrenar modelo final con los mejores hiperparámetros en paralelo
print("\nEntrenando modelo final con mejores hiperparámetros en paralelo...")
x_train_full = np.concatenate([X_train, X_val], axis=0)
y_train_full = np.concatenate([y_train, y_val], axis=0)

# Entrenar modelo
mejor_rf.fit(x_train_full, y_train_full)
print("Entrenamiento finalizado.")

# Hacer predicciones con conjunto de prueba
y_pred_test = mejor_rf.predict(X_test)

# Calcular métrica de accuracy
acc_test = accuracy_score(y_test, y_pred_test)

# Calcular métrica de f1_score
f1_macro = f1_score(y_test, y_pred_test, average='macro')

print(f"Accuracy test: {acc_test:.4f}")
print(f"F1 macro: {f1_macro:.4f}")

print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    y_pred_test,
    target_names=clases,
    digits=4
))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=clases,
    yticklabels=clases
)

# Visualización de métricas de evaluación
plt.title(
    f'Random forest - test\n'
    f'Acc: {acc_test:.4f} | F1-macro: {f1_macro:.4f}'
)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.show()
