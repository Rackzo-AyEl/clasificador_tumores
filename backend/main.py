import io
import base64
import cv2
import numpy as np
import onnxruntime as ort
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

print("Cargando motores ONNX en memoria...")
clasificador = ort.InferenceSession("./Modelos/clasificador_resnet50.onnx")

segmentadores = {
    "Glioma": ort.InferenceSession("./Modelos/segmentador_glioma.onnx"),
    "Meningioma": ort.InferenceSession("./Modelos/segmentador_meningioma.onnx"),
    "Pituitaria": ort.InferenceSession("./Modelos/segmentador_pituitary.onnx")
}

CLASES = ["Glioma", "Meningioma", "Pituitaria", "Sano"]

# =========================================================
# 2. FUNCIONES DE PREPROCESAMIENTO
# =========================================================
def preprocesar_imagen(img_pil):
    img = img_pil.convert("RGB").resize((224, 224))
    img_arr = np.array(img).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std
    
    img_arr = np.transpose(img_arr, (2, 0, 1)) 
    img_arr = np.expand_dims(img_arr, axis=0)  
    img_arr = img_arr.astype(np.float32)
    return img_arr

def preprocesar_segmentacion(img_pil):  # ← NUEVA
    img = np.array(img_pil.convert("RGB"))
    img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# =========================================================
# 3. EL ENDPOINT PRINCIPAL
# =========================================================
@app.post('/procesar-mri/')
async def procesar_imagen(imagen: UploadFile = File(...)):
    try:
        bytes_imagen = await imagen.read()
        imagen_abierta = Image.open(io.BytesIO(bytes_imagen))
        
        tensor_entrada = preprocesar_imagen(imagen_abierta)
        
        input_name_cls = clasificador.get_inputs()[0].name
        logits_cls = clasificador.run(None, {input_name_cls: tensor_entrada})[0]
        probabilidades = softmax(logits_cls)
        indice_ganador = np.argmax(probabilidades, axis=1)[0]
        clase_predicha = CLASES[indice_ganador]
        confianza = round(float(probabilidades[0][indice_ganador] * 100), 2)
        
        print(f"🧠 Detección: {clase_predicha} al {confianza}%")

        mascara_b64 = ""
        
        if clase_predicha in segmentadores:
            modelo_unet = segmentadores[clase_predicha]
            
            input_name_seg = modelo_unet.get_inputs()[0].name
            
            tensor_seg = preprocesar_segmentacion(imagen_abierta)

            logits_seg = modelo_unet.run(None, {input_name_seg: tensor_seg})[0]
            
            prob_píxeles = 1 / (1 + np.exp(-logits_seg))
            
            mascara_binaria = (prob_píxeles > 0.5).astype(np.uint8) 
            
            mascara = mascara_binaria[0][0]  # (256,256)
            mascara_img = Image.fromarray(mascara * 255, mode='L')
            buffer = io.BytesIO()
            mascara_img.save(buffer, format="PNG")
            mascara_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
            print("Input shape esperado:", modelo_unet.get_inputs()[0].shape)

        else:
            print("🟢 Paciente Sano. Se omite la segmentación.")

        return {
            "clase": clase_predicha,
            "confianza": confianza,
            "mascara_base64": mascara_b64
        }

    except Exception as e:
        print(f"❌ Error interno procesando la imagen: {e}")
        traceback.print_exc() 
        return {
            "clase": "Error",
            "confianza": 0,
            "mascara_base64": ""
        }
