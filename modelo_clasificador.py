# Librerías
from colorama import Fore, Style
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import numpy as np


def evaluar_modelo_estricto(device, modelo, dataloader_test):
    print(Fore.YELLOW + "\n[!] " + Style.RESET_ALL +
          "Generando métricas estrictas (F1-Score y Matriz)...")

    modelo.eval()  # Modo evaluación
    etiquetas_reales = []
    predicciones = []

    with torch.no_grad():
        for caracteristicas, etiquetas in dataloader_test:
            caracteristicas = caracteristicas.to(device)

            # Obtener las salidas de la red
            salidas = modelo(caracteristicas)

            # torch.max devuelve el valor máximo y su índice (que es la clase predicha)
            _, preds = torch.max(salidas, 1)

            # Guardamos resultados pasándolos de vuelta a la CPU y a Numpy
            etiquetas_reales.extend(etiquetas.view(-1).numpy())
            predicciones.extend(preds.cpu().numpy())

    # Calcular F1-Score usando 'macro'
    # (Macro es el más estricto: saca el F1 de cada clase por separado y luego los promedia,
    # penalizando fuertemente si el modelo es malo en una sola clase).
    f1 = f1_score(etiquetas_reales, predicciones, average='macro')
    print(Fore.GREEN + f"[+] F1-Score (Macro): {f1:.4f}" + Style.RESET_ALL)

    # Generar Matriz de Confusión
    cm = confusion_matrix(etiquetas_reales, predicciones)

    # Dibujar la matriz
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3'],
                yticklabels=['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3'])

    plt.title(f'Matriz de Confusión | F1-Score (Macro): {f1:.4f}', pad=20)
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Etiqueta Real (Ground Truth)')

    # Rotar las etiquetas para que se lean bien
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Funcion de entrenamiento
def entrenamiento_modelo(device, modelo, dataloader_train, dataloader_test):
    # Parámetros de aprendizaje de modelo
    optimizador = optim.Adam(modelo.parameters(), lr=0.0001)
    criterio = nn.CrossEntropyLoss()

    # Ciclo de entrenamiento de modelo
    epocas = 70
    historial_loss_train = []
    historial_loss_test = []

    print(Fore.YELLOW + "\n[!] " + Style.RESET_ALL, end='')
    print("Iniciando entrenamiento...")

    for epoca in range(epocas):
        modelo.train()
        loss_train_acumulado = 0.0

        for caracteristicas, etiquetas in dataloader_train:
            caracteristicas, etiquetas = caracteristicas.to(
                device), etiquetas.to(device)

            etiquetas = etiquetas.to(device).view(-1).long()

            optimizador.zero_grad()
            salidas = modelo(caracteristicas)
            loss = criterio(salidas, etiquetas)
            loss.backward()
            optimizador.step()

            loss_train_acumulado += loss.item() * caracteristicas.size(0)

        loss_promedio_train = loss_train_acumulado / len(dataset_train)
        historial_loss_train.append(loss_promedio_train)

        # Fase de Validación
        modelo.eval()
        loss_test_acumulado = 0.0
        with torch.no_grad():
            for caracteristicas, etiquetas in dataloader_test:
                caracteristicas, etiquetas = caracteristicas.to(
                    device), etiquetas.to(device)
                salidas = modelo(caracteristicas)
                loss = criterio(salidas, etiquetas)
                loss_test_acumulado += loss.item() * caracteristicas.size(0)

        loss_promedio_test = loss_test_acumulado / len(dataset_test)
        historial_loss_test.append(loss_promedio_test)

        if (epoca + 1) % 5 == 0:
            print(f"Época {epoca+1:02d}/{epocas} | Loss Train: {
                  loss_promedio_train:.4f} | Loss Test: {loss_promedio_test:.4f}")

    # 7. Graficar resultados
    plt.figure(figsize=(8, 5))
    plt.plot(historial_loss_train, label='Entrenamiento', marker='.')
    plt.plot(historial_loss_test, label='Validación', marker='.')
    plt.title('Curva de Aprendizaje con Caché de Características')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()


# Inicio de código
if __name__ == '__main__':
    # Cargar vectores de características preprocesadas
    path_train = 'features/features_train.pt'
    path_test = 'features/features_test.pt'

    x_train, y_train = torch.load(path_train)
    x_test, y_test = torch.load(path_test)

    print(Fore.CYAN + "\n[*] " + Style.RESET_ALL, end='')
    print("Vectores de características cargados correctamente")

    # Verificar disponibilidad de GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(Fore.CYAN + "[*] " + Style.RESET_ALL, end='')
    print(f"Usando dispositivo {torch.cuda.get_device_name()}")

    # Generar datasets
    dataset_train = TensorDataset(x_train, y_train)
    dataset_test = TensorDataset(x_test, y_test)

    # Generar dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=128)

    # Crear modelo
    modelo = nn.Sequential(
        nn.Linear(2048, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.8),

        nn.Linear(128, 4)
    ).to(device)

    # Llamada a función de entrenamiento
    entrenamiento_modelo(device, modelo, dataloader_train, dataloader_test)

    evaluar_modelo_estricto(device, modelo, dataloader_test)
