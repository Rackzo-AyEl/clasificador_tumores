import numpy as np
import onnxruntime as ort
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms

def preprocesar_onnx(ruta_imagen):
    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagen = Image.open(ruta_imagen).convert('RGB')
    imagen = transformaciones(imagen).unsqueeze(0)
    
    # ONNX espera arreglos de NumPy
    return imagen.numpy()

# Configurar el Proveedor (GPU o CPU)
providers = ['CPUExecutionProvider']
ruta_onnx = '../Modelos/clasificador_resnet50.onnx'

session = ort.InferenceSession(ruta_onnx, providers=providers)

# Preparar la entrada
ruta_imagen = '../test/glioma/gl-0045.jpg'
input_data = preprocesar_onnx(ruta_imagen)

# Ejecutar la inferencia
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

raw_result = session.run([output_name], {input_name: input_data})[0]

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

probabilidades = softmax(raw_result)

clase_predicha = np.argmax(probabilidades, axis=1).item()
confianza = probabilidades[0, clase_predicha].item()

print(Fore.GREEN, end='')
print(f"--- Prueba ONNX ---")
print(f"Predicción: Clase {clase_predicha} con {confianza*100:.2f}% de confianza")
