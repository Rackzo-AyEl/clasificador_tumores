import torch
import numpy as np
import cv2
import os
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm
from colorama import Fore, Style

# ==========================================
# 1. CONFIGURACIÓN (CAMBIA ESTO)
# ==========================================
RUTA_MODELO = "./Modelos/segmentador_pituitary.pth"
DIR_IMAGENES = "./Masks/image/test/pituitary/"
DIR_MASCARAS = "./Masks/mask/test/pituitary/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. FUNCIONES DE MÉTRICAS
# ==========================================
def calcular_metricas(pred, target, threshold=0.5, smooth=1e-6):
    with torch.no_grad():
        # Convertir logit a binario
        pred = (torch.sigmoid(pred) > threshold).float()
        
        # Aplanar tensores
        pred = pred.view(-1)
        target = target.view(-1)

        interseccion = (pred * target).sum()
        total = pred.sum() + target.sum()
        union = total - interseccion

        # Dice Coefficient
        dice = (2. * interseccion + smooth) / (total + smooth)
        # IoU (Jaccard)
        iou = (interseccion + smooth) / (union + smooth)

        return dice.item(), iou.item()

# ==========================================
# 3. DATASET PERSONALIZADO
# ==========================================
class DatasetEval(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imagenes = sorted(os.listdir(img_dir))
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, idx):
        nombre_img = self.imagenes[idx]
        # Lógica para encontrar la máscara _m.jpg
        nombre_mask = nombre_img.replace('.jpg', '_m.jpg')
        
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, nombre_img)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, nombre_mask), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=img, mask=mask)
        return augmented['image'], augmented['mask'].unsqueeze(0)

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
def evaluar():
    print(Fore.CYAN + f"[*] Cargando modelo: {RUTA_MODELO}" + Style.RESET_ALL)
    
    # Reconstruir arquitectura
    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights=None, # No necesitamos pesos de imagenet para evaluar
        in_channels=3,
        classes=1
    ).to(DEVICE)

    # Cargar pesos entrenados
    model.load_state_dict(torch.load(RUTA_MODELO, map_location=DEVICE))
    model.eval()

    # Preparar Datos
    dataset = DatasetEval(DIR_IMAGENES, DIR_MASCARAS)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    dice_scores = []
    iou_scores = []

    print(Fore.YELLOW + f"[*] Evaluando {len(dataset)} imágenes en {DEVICE}..." + Style.RESET_ALL)

    with torch.no_grad():
        for img, mask in tqdm(loader, desc="Progreso"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            output = model(img)
            
            dice, iou = calcular_metricas(output, mask)
            dice_scores.append(dice)
            iou_scores.append(iou)

    # Mostrar Resultados
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)

    print("\n" + "="*40)
    print(Fore.GREEN + f"RESULTADOS FINALES - {RUTA_MODELO.upper()}" + Style.RESET_ALL)
    print("="*40)
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Mean IoU (Jaccard):    {mean_iou:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluar()
