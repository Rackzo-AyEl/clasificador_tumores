# Librerías
import albumentations as A
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
import time
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from colorama import Fore, Style
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# Clases
class DatasetTumores(Dataset):
    def __init__(self, ruta_imagenes, ruta_mascaras, transformaciones=None):
        self.ruta_imagenes = ruta_imagenes
        self.ruta_mascaras = ruta_mascaras
        self.transformaciones = transformaciones

        # Leemos los nombres de los archivos (suponiendo que se llaman igual en ambas carpetas)
        self.imagenes = sorted(os.listdir(ruta_imagenes))

    def __len__(self):
        # PyTorch necesita saber cuántas imágenes hay en total
        return len(self.imagenes)

    def __getitem__(self, idx):
        # 1. Armar las rutas exactas
        nombre_archivo = self.imagenes[idx]
        ruta_img = os.path.join(self.ruta_imagenes, nombre_archivo)

        # --- EL PARCHE DE KAGGLE ---
        # Reemplazamos el final del nombre para que coincida con la máscara
        nombre_mascara = nombre_archivo.replace('.jpg', '_m.jpg')
        ruta_mask = os.path.join(self.ruta_mascaras, nombre_mascara)

        # 2. Leer con OpenCV (y convertir colores correctamente)
        imagen = cv2.imread(ruta_img)
        if imagen is None:
            raise FileNotFoundError(
                f"[!] Error: No se encontró la imagen en {ruta_img}")
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # Leer la máscara en escala de grises
        mascara = cv2.imread(ruta_mask, cv2.IMREAD_GRAYSCALE)
        if mascara is None:
            raise FileNotFoundError(
                f"[!] Error: OpenCV no pudo leer la máscara en {ruta_mask}")

        # Convertir la máscara a binaria pura (0.0 y 1.0) para la U-Net
        mascara = (mascara > 127).astype(np.float32)

        # 3. La Magia de Albumentations...
        if self.transformaciones is not None:
            aumentado = self.transformaciones(image=imagen, mask=mascara)
            imagen = aumentado['image']
            mascara = aumentado['mask']

        mascara = mascara.unsqueeze(0)

        return imagen, mascara


# Función para cargar y transformar imágenes
def cargar_imagenes(ruta_img, ruta_mask, training):
    print(Fore.CYAN + "[*] " + Style.RESET_ALL, end='')
    print(f"Preparando Dataset (Imágenes: {ruta_img} | Máscaras: {ruta_mask})")

    # Transformaciones con Albumentations para data augmentation
    if training:
        transformaciones = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            # shift_limit es para que no mueva la imagen fuera del marco, rotate hace la rotación
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            # Normalización estándar de ImageNet
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()  # Convierte de NumPy a Tensor de PyTorch
        ])

    # Carga limpia para Validación/Test (Sin rotaciones ni locuras)
    else:
        transformaciones = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    # En lugar de ImageFolder, devolvemos nuestra propia clase
    return DatasetTumores(ruta_imagenes=ruta_img,
                          ruta_mascaras=ruta_mask,
                          transformaciones=transformaciones)


def calcular_iou(predicciones, mascaras_reales, umbral=0.5):
    """
    Calcula el Intersection over Union (IoU) entre el dibujo de la red y la máscara real.
    """
    with torch.no_grad():  # Apagamos gradientes porque esto es solo una métrica, no aprendemos de aquí
        # 1. Convertir los números crudos de la U-Net (logits) a probabilidades (0 a 1) y luego a 0s o 1s puros
        predicciones_binarias = (torch.sigmoid(predicciones) > umbral).float()

        # 2. Aplanar los tensores (Dejar de verlos como imágenes 2D y verlos como una tira larga de píxeles)
        predicciones_planas = predicciones_binarias.view(-1)
        mascaras_planas = mascaras_reales.view(-1)

        # 3. Matemática de conjuntos
        interseccion = (predicciones_planas * mascaras_planas).sum()
        total = predicciones_planas.sum() + mascaras_planas.sum()
        union = total - interseccion

        # 4. Seguro contra imágenes completamente negras (sin tumor en ambas)
        if union == 0:
            return 1.0 if total == 0 else 0.0

        iou = interseccion / union
        return iou.item()


# Función para entrenar modelo
def entrenar_modelo(modelo):
    # Imágenes para train
    print(Fore.CYAN + "[*] " + Style.RESET_ALL, end='')
    print(f"Cargando y transformando máscaras e imágenes para segmentación")
    # Llamada para modelo de glioma
    dataset_entrenamiento = cargar_imagenes('Masks/image/pituitary/',
                                            'Masks/mask/pituitary/',
                                            training=True)

    loader_entrenamiento = DataLoader(
        dataset_entrenamiento, batch_size=32, shuffle=True, num_workers=2)
    # BCEWithLogitsLoss es perfecta para máscaras de 0s y 1s
    criterio = nn.BCEWithLogitsLoss()

    # Adam es el albañil estándar que ajustará los pesos
    optimizador = optim.Adam(modelo.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode='min', factor=0.5, patience=3)
    # --- 2. CONFIGURACIÓN DEL BUCLE ---
    epocas = 30
    mejor_loss = float('inf')

    print(Fore.YELLOW +
          "\n[!] INICIANDO ENTRENAMIENTO DE U-NET" + Style.RESET_ALL)

    # Inicio de toma de tiempo
    inicio = time.time()

    for epoca in range(epocas):
        modelo.train()
        loss_acumulado = 0.0
        iou_acumulado = 0.0  # <--- NUEVO: Variable para guardar el IoU

        barra_progreso = tqdm(loader_entrenamiento, desc=f"Época {
                              epoca+1}/{epocas}", unit="batch")

        for batch_idx, (imagenes, mascaras_reales) in enumerate(barra_progreso):
            imagenes = imagenes.to(device)
            mascaras_reales = mascaras_reales.to(device)

            predicciones = modelo(imagenes)
            loss = criterio(predicciones, mascaras_reales)

            optimizador.zero_grad()
            loss.backward()
            optimizador.step()

            # --- NUEVO: Calcular IoU de este batch ---
            iou_batch = calcular_iou(predicciones, mascaras_reales)

            loss_acumulado += loss.item()
            iou_acumulado += iou_batch  # <--- NUEVO: Sumar al acumulado

            # --- NUEVO: Mostrar ambos en la barra de Neovim ---
            barra_progreso.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'IoU': f"{iou_batch:.4f}"
            })

        # --- RESULTADOS DE LA ÉPOCA ---
        loss_promedio = loss_acumulado / len(loader_entrenamiento)
        # <--- NUEVO: Promedio del IoU
        iou_promedio = iou_acumulado / len(loader_entrenamiento)

        print(Fore.GREEN + f"[-] Fin Época {epoca+1} | Loss: {
              loss_promedio:.4f} | IoU: {iou_promedio:.4f}" + Style.RESET_ALL)

        # Le damos el Loss promedio. Él internamente decide si baja el LR o lo deja igual.
        scheduler.step(loss_promedio)

        # Guardar el mejor modelo (tu código original)
        if loss_promedio < mejor_loss:
            mejor_loss = loss_promedio
            torch.save(modelo.state_dict(), "mejor_modelo.pth")
            print(Fore.MAGENTA +
                  "[+] ¡Nuevo récord! Modelo guardado." + Style.RESET_ALL)

    tiempo_total = time.time() - inicio
    print(Fore.YELLOW + "[!] " + Style.RESET_ALL, end='')
    print(f"Tiempo total de ejecución: {round(tiempo_total, 2)}")

    print(Fore.CYAN + "\n[*] ENTRENAMIENTO FINALIZADO." + Style.RESET_ALL)


# Inicio de código
if __name__ == '__main__':
    # Verificar disponibilidad de GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(Fore.CYAN + "[*] " + Style.RESET_ALL, end='')
    print(f"Usando dispositivo {torch.cuda.get_device_name()}")

    # Instancia de modelo U-Net
    modelo = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    ).to(device)

    # Función para cargar imágenes
    entrenar_modelo(modelo)
