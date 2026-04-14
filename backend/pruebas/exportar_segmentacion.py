
import torch
import segmentation_models_pytorch as smp
import os
from colorama import Fore, Style

def exportar_modelos_unet():
    # 1. Configuración general
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Usando dispositivo para exportación: {device}")
    
    # Tamaño confirmado por tu script de entrenamiento
    TAMANO_IMG = 224 
    
    # Diccionario con las rutas de tus modelos .pth y dónde guardar los .onnx
    # Ajusta las rutas según cómo tengas estructurado tu proyecto
    modelos_a_exportar = {
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

    # 2. Instanciar la arquitectura base (Debe ser idéntica al script original)
    # Como los 3 tumores usan la misma arquitectura base, solo la declaramos una vez
    modelo_base = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights=None, # Ya no necesitamos descargar pesos de ImageNet aquí
        in_channels=3,
        classes=1
    ).to(device)

    # Tensor explorador (El "Efecto Espejo" a 224x224)
    dummy_input = torch.randn(1, 3, TAMANO_IMG, TAMANO_IMG, device=device)

    # 3. Bucle para exportar cada modelo
    for clase, rutas in modelos_a_exportar.items():
        if not os.path.exists(rutas["pth"]):
            print(Fore.RED + f"[!] No se encontró el modelo {rutas['pth']}" + Style.RESET_ALL)
            continue
            
        print(Fore.YELLOW + f"\n[*] Procesando segmentador de {clase}..." + Style.RESET_ALL)
        
        # Cargar pesos específicos de la clase
        modelo_base.load_state_dict(torch.load(rutas["pth"], map_location=device, weights_only=True))
        
        # EL PASO CRÍTICO: Desactivar Dropout y BatchNorm
        modelo_base.eval()
        
        # Exportar a ONNX
        torch.onnx.export(
            modelo_base,                 # Modelo con los pesos cargados
            dummy_input,                 # Imagen falsa de 224x224
            rutas["onnx"],               # Ruta de salida
            export_params=True,          # Guardar los pesos dentro del archivo
            opset_version=14,            # Versión estable para U-Net y EfficientNet
            do_constant_folding=True,    # Optimizar el grafo
            input_names=['imagen_mri'],  # Nombre de entrada
            output_names=['mascara']     # Nombre de salida
        )
        
        print(Fore.GREEN + f"[+] {clase.capitalize()} exportado exitosamente a {rutas['onnx']}" + Style.RESET_ALL)

if __name__ == '__main__':
    exportar_modelos_unet()
