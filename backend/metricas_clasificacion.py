import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# 1. Configuración y Carga de Datos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = './test/' # Rumeningiomata a tu carpeta de validación
batch_size = 16

# Transformaciones (Deben ser las mismas que usaste en el entrenamiento)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_data = datasets.ImageFolder(data_dir, transform=val_transforms)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
class_names = val_data.classes # ['glioma', 'healthy', 'meningioma', 'pituitary']

# 2. Cargar el Modelo Entrenado
# Asegúrate de que coincida con tu arquitectura (ResNet50 + MLP)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
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
ruta_pesos = "./Modelos/clasificador_resnet50.pth" # Cambia por el nombre de tu archivo
model.load_state_dict(torch.load(ruta_pesos, map_location=device))
model = model.to(device)
model.eval()

# 3. Inferencia
all_preds = []
all_labels = []

print("Iniciando evaluación en el conjunto de validación...")

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 4. Cálculo de Métricas Finales
print("\n" + "="*30)
print("REPORTE DE CLASIFICACIÓN")
print("="*30)
print(classification_report(all_labels, all_preds, target_names=class_names))

# 5. Matriz de Confusión (Visualización Pro)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción del modelo')
plt.ylabel('Clase real')
plt.title('Matriz de confusión')
plt.show()
