
import torch
import segmentation_models_pytorch as smp
import os
from colorama import Fore, Style

def exportar_modelos_unet():
    # Configuración general
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Usando dispositivo para exportación: {device}")
    
    TAMANO_IMG = 224 
    
    # Diccionario con las rutas de tus modelos .pth y dónde guardar los .onnx
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

    # Instanciar la arquitectura base
    modelo_base = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)

    # Tensor explorador
    dummy_input = torch.randn(1, 3, TAMANO_IMG, TAMANO_IMG, device=device)

    # Bucle para exportar cada modelo
    for clase, rutas in modelos_a_exportar.items():
        if not os.path.exists(rutas["pth"]):
            print(Fore.RED + f"[!] No se encontró el modelo {rutas['pth']}" + Style.RESET_ALL)
            continue
            
        print(Fore.YELLOW + f"\n[*] Procesando segmentador de {clase}..." + Style.RESET_ALL)
        
        # Cargar pesos específicos de la clase
        modelo_base.load_state_dict(torch.load(rutas["pth"], map_location=device, weights_only=True))
        
        # Desactivar Dropout y BatchNorm
        modelo_base.eval()
        
        # Exportar a ONNX
        torch.onnx.export(
            modelo_base,
            dummy_input,
            rutas["onnx"],
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['imagen_mri'],
            output_names=['mascara']
        )
        
        print(Fore.GREEN + f"[+] {clase.capitalize()} exportado exitosamente a {rutas['onnx']}" + Style.RESET_ALL)

if __name__ == '__main__':
    exportar_modelos_unet()
