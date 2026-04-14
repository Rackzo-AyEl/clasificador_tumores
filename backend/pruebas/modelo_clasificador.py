import torch
import torch.nn as nn
from colorama import Fore, Style
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torchvision.models.densenet import _load_state_dict

# Preprocesar imagen
def cargar_imagenes(ruta_imagen, device):
    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    imagen = Image.open(ruta_imagen).convert('RGB')
    imagen = transformaciones(imagen)
    imagen = imagen.unsqueeze(0)
    return imagen.to(device)


# Dispotivo GPU
device = torch.device('cuda')

modelo = resnet50()

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

# Cargar pesos
ruta_pesos = '../Modelos/clasificador_resnet50.pth'
modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))

# Mover modelo a device
modelo.to(device)
# Desactivar gradientes
modelo.eval()

# Cargar imagen
ruta_imagen = '../test/glioma/gl-0045.jpg'
input_tensor = cargar_imagenes(ruta_imagen, device)

# Ejecutar prueba
with torch.no_grad():
    salida = modelo(input_tensor)
    probabilidades = torch.nn.functional.softmax(salida, dim=1)
    confianza, clase_predicha = torch.max(probabilidades, 1)

print(Fore.YELLOW, end='')
print(f"\n--- Prueba PTH ---")
print(f"Predicción: Clase {clase_predicha.item()} con {confianza.item()*100:.2f}% de confianza")
