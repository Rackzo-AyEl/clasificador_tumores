import torch
import torch.nn as nn
from torchvision.models import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Reconstruir tu modelo
modelo = resnet50()
modelo.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(p=0.6),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(p=0.6),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(p=0.6),
    nn.Linear(64, 4)
)

# 2. Cargar los pesos
ruta_pesos = '../Modelos/clasificador_resnet50.pth'
modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
modelo.to(device)

# --- EL PASO CRÍTICO QUE FALTA ---
modelo.eval() 
# ---------------------------------

# 3. Crear tensor de ejemplo (Nota: usando la dimensión 256 que te pidió ONNX)
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# 4. Exportar el modelo
ruta_onnx = '../Modelos/clasificador_resnet50_corregido.onnx'
torch.onnx.export(
    modelo,                      # El modelo en memoria
    dummy_input,                 # Input de ejemplo
    ruta_onnx,                   # Dónde guardarlo
    export_params=True,          # Guardar los pesos entrenados
    opset_version=14,            # Versión de operaciones ONNX (14 es muy estable)
    do_constant_folding=True,    # Optimización clave para inferencia
    input_names=['entrada_mri'], # El nombre que configuraste
    output_names=['salida']
)

print("Exportación exitosa en modo evaluación.")
