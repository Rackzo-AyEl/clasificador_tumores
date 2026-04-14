import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from colorama import Fore, Style

# ==========================================
# 1. CONFIGURACIÓN (Lo único que modificas)
# ==========================================
RUTA_IMAGEN = '../test/pituitary/pi-0325.jpg'  # <-- Pon aquí la imagen que quieras probar
CLASE_A_PROBAR = 'glioma'                   # <-- Opciones: 'pituitaria', 'glioma', 'meningioma'

# Tus rutas exactas
modelos = {
    "pituitaria": {
        "pth": "../Modelos/segmentador_pituitary.pth",
        "onnx": "../Modelos/segmentador_pituitary.onnx"
    },
    "glioma": {
        "pth": "../Modelos/segmentador_glioma.pth",
        "onnx": "../Modelos/segmentador_glioma.onnx"
    },
    "meningioma": {
        "pth": "../Modelos/segmentador_meningioma.pth",
        "onnx": "../Modelos/segmentador_meningioma.onnx"
    }
}
# ==========================================

def procesar_imagen_para_inferencia(ruta):
    """Aplica exactamente el mismo preprocesamiento que en validación"""
    imagen = cv2.imread(ruta)
    if imagen is None:
        raise FileNotFoundError(f"[!] No se encontró la imagen: {ruta}")
    
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_original = imagen.copy() # Guardamos copia para graficar después
    
    # Transformaciones idénticas a tu DatasetTumores
    transformaciones = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    tensor = transformaciones(image=imagen)['image'].unsqueeze(0)
    return tensor, cv2.resize(imagen_original, (224, 224))

def main():
    print(Fore.CYAN + f"[*] Iniciando prueba comparativa para: {CLASE_A_PROBAR.upper()}" + Style.RESET_ALL)
    rutas = modelos[CLASE_A_PROBAR]
    
    # Preparar imagen
    tensor_input, img_visualizacion = procesar_imagen_para_inferencia(RUTA_IMAGEN)
    
    # ---------------------------------------------------------
    # 2. INFERENCIA CON PYTORCH (.pth)
    # ---------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[-] Cargando modelo PyTorch en {device}...")
    
    modelo_pth = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights=None, 
        in_channels=3,
        classes=1
    ).to(device)
    
    modelo_pth.load_state_dict(torch.load(rutas['pth'], map_location=device, weights_only=True))
    modelo_pth.eval()
    
    with torch.no_grad():
        salida_pth = modelo_pth(tensor_input.to(device))
        # Aplicamos sigmoide y umbral para convertir a máscara binaria
        prob_pth = torch.sigmoid(salida_pth).squeeze().cpu().numpy()
        mascara_pth = (prob_pth > 0.5).astype(np.uint8)

    # ---------------------------------------------------------
    # 3. INFERENCIA CON ONNX (.onnx)
    # ---------------------------------------------------------
    print("[-] Cargando modelo ONNX...")
    proveedores = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sesion_onnx = ort.InferenceSession(rutas['onnx'], providers=proveedores)
    
    input_name = sesion_onnx.get_inputs()[0].name
    # ONNX necesita un array de numpy puro
    input_data = tensor_input.numpy() 
    
    salida_onnx = sesion_onnx.run(None, {input_name: input_data})[0]
    
    def sigmoide(x): 
        return 1 / (1 + np.exp(-x))
    
    prob_onnx = sigmoide(salida_onnx).squeeze()
    mascara_onnx = (prob_onnx > 0.5).astype(np.uint8)

    # ---------------------------------------------------------
    # 4. VISUALIZACIÓN "A BUEN OJO"
    # ---------------------------------------------------------
    print(Fore.GREEN + "[+] Generando comparativa visual..." + Style.RESET_ALL)
    
    # Crear una figura con 3 subgráficos
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Comparativa de Segmentación: {CLASE_A_PROBAR.upper()}', fontsize=16)

    # Imagen Original
    axes[0].imshow(img_visualizacion)
    axes[0].set_title('Resonancia Original')
    axes[0].axis('off')
    
    axes[1].imshow(img_visualizacion)
    axes[1].imshow(mascara_pth, cmap='jet', alpha=0.4)
    axes[1].set_title('Predicción PyTorch (.pth)')
    axes[1].axis('off')
    
    axes[2].imshow(img_visualizacion)
    axes[2].imshow(mascara_onnx, cmap='jet', alpha=0.4)
    axes[2].set_title('Predicción ONNX (.onnx)')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
