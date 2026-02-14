# Librerías
from colorama import Fore, Style
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Función para cargar imágenes
def cargar_imagenes(ruta, training):
    # Imágenes para train
    print(Fore.CYAN + "[*] " + Style.RESET_ALL, end='')
    print(f"Cargando y transformando imágenes de carpeta {ruta}")

    # Transformaciones matriciales para data augmentation
    if training:
        transformaciones = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    # Carga normal de datos para ResNet50
    else:
        transformaciones = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    # Devolver dataset con transformaciones de imágenes aplicadas
    return datasets.ImageFolder(root=ruta, transform=transformaciones)


# Función de entrenamiento
def entrenar_modelo(dataset_train, dataset_test, device, modelo):
    # Crear variables de optimización
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=0.0001)

    historial_loss_train = []
    historial_loss_test = []

    print(Fore.YELLOW + "\n[!] " + Style.RESET_ALL, end='')
    print("Iniciando entrenamiento de modelo...")

    # Ciclo de entrenamiento de red
    epocas = 30

    for epoca in range(epocas):
        modelo.train()
        loss_train_acumulado = 0.0

        for imagenes, etiquetas in dataloader_train:
            # Mover datos a GPU
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)

            # Establecer gradiente en cero
            optimizador.zero_grad()

            # Pasar imágenes por toda resnet50
            salidas = modelo(imagenes)

            loss = criterio(salidas, etiquetas)
            loss.backward()
            optimizador.step()

            loss_train_acumulado += loss.item() * imagenes.size(0)

        tamanno_dataloader_train = len(dataloader_train.dataset)
        loss_promedio_train = loss_train_acumulado / tamanno_dataloader_train
        historial_loss_train.append(loss_promedio_train)

        modelo.eval()
        loss_test_acumulado = 0.0

        with torch.no_grad():
            for imagenes, etiquetas in dataloader_test:
                imagenes = imagenes.to(device)
                etiquetas = etiquetas.to(device).view(-1).long()

                salidas = modelo(imagenes)
                loss = criterio(salidas, etiquetas)
                loss_test_acumulado += loss.item() * imagenes.size(0)

        loss_promedio_test = loss_test_acumulado / len(dataloader_test.dataset)
        historial_loss_test.append(loss_promedio_test)

        print(f"Época {epoca+1:02d}/{epocas} | Loss Train: {
              loss_promedio_train:.4f} | Loss Test: {loss_promedio_test:.4f}")

    # Graficar (Opcional para pruebas cortas)
    plt.plot(historial_loss_train, label='Train')
    plt.plot(historial_loss_test, label='Test')
    plt.legend()
    plt.show()


# Inicio de código
if __name__ == '__main__':
    # Cargar imágenes de entrenamiento con sus respectivas transformaciones
    dataset_train = cargar_imagenes('train/', True)
    dataset_test = cargar_imagenes('test/', False)

    # Crear dataloaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=256,
        num_workers=2
    )

    # Verificar disponibilidad de GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(Fore.CYAN + "[*] " + Style.RESET_ALL, end='')
    print(f"Usando dispositivo {torch.cuda.get_device_name()}")

    # Cargar resnet50 con sus pesos
    modelo = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Congelar gradientes en modelo
    for param in modelo.parameters():
        param.requires_grad = False

    # Reemplazar capa fc
    modelo.fc = nn.Sequential(
        # Capa de entrada
        nn.Linear(2048, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.6),

        # Primera capa oculta
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(p=0.6),

        # Segunda capa oculta
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(p=0.6),

        # Capa de salida
        nn.Linear(64, 4)
    )

    # Mover modelo a device
    modelo.to(device)

    # Llamar a función para entrenar modelo
    entrenar_modelo(dataset_train, dataset_test, device, modelo)
