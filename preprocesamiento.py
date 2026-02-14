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
def cargar_imagenes(ruta):
    print(Fore.CYAN + "[*] " + Style.RESET_ALL, end='')
    print(f"Cargando y transformando imágenes de carpeta {ruta}")
    # Transformaciones de imágenes para redes neuronales
    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Devolcer dataset con transformaciones de imágenes aplicadas
    return datasets.ImageFolder(root=ruta, transform=transformaciones)


# Función para generar vectores de características de imágenes
def generar_vectores(dataloader, modelo, device, nombre):
    # Configurar modelo en modo de evaluacion
    modelo.eval()

    lista_caracteristicas = []
    lista_etiquetas = []

    print(Fore.YELLOW + "\n[!] " + Style.RESET_ALL, end='')
    print(f"Extratendo características de imágenes para {nombre}")

    with torch.no_grad():
        for imagenes, etiquetas in dataloader:
            # Mover datos a device
            imagenes = imagenes.to(device)

            # Pasar todas las imágenes por resnet50
            caracteristicas = modelo(imagenes)

            # Guardar resultados en RAM
            lista_caracteristicas.append(caracteristicas.cpu())
            lista_etiquetas.append(etiquetas.cpu())

    # Concatenar lotes en un solo tensor
    tensor_caracteristicas = torch.cat(lista_caracteristicas)
    tensor_etiquetas = torch.cat(lista_etiquetas)

    # Almacenar archivo
    torch.save((tensor_caracteristicas, tensor_etiquetas), f'features/{nombre}')
    print(Fore.GREEN + "[+] " + Style.RESET_ALL, end='')
    print("Vectores almacenados correctamente")
    

# Inicio de código
if __name__ == '__main__':
    # Cargar imágenes de entrenamiento
    dataset_train = cargar_imagenes('train/')
    conjunto_test = cargar_imagenes('test/')
    
    # Crear dataloader
    dataloader_train  = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=6)
    dataloader_test  = DataLoader(conjunto_test, batch_size=32, num_workers=6)
    
    # Verificar disponibilidad de GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(Fore.CYAN + "[*] " + Style.RESET_ALL, end='')
    print(f"Usando dispositivo {torch.cuda.get_device_name()}")

    # Cargar resnet50 con sus pesos
    pesos = ResNet50_Weights.IMAGENET1K_V2
    modelo = resnet50(weights=pesos)
    
    # Congelar parámetros de aprendizaje
    for parametros in modelo.parameters():
        parametros.requires_grad = False

    # Cambiar salida de capa fc
    modelo.fc = nn.Identity()

    # Mover modelo a device
    modelo.to(device)

    # Guardar vectores de características de imágenes
    generar_vectores(dataloader_train, modelo, device, 'features_train.pt')
    generar_vectores(dataloader_test, modelo, device, 'features_test.pt')
