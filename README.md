# Clasificador y segmentador de tumores cerebrales
En este proyecto se buscan métodos confiables para trabajar sobre visión artificial con imágenes médicas que requieren una alta precisión y confiabilidad para la clasificación y segmentación de tumores cerebrales usando MRI (Imágenes de resonancia magnética) entre 4 clases diferentes (healthy, glioma, meningioma, potuitary).
Las redes propuestas sobre este trabajo son MLP (perceptrón multicapa) junto a ResNet50 para clasificación y U-Net con EfficientNet para la segmentación. Aquí se aprovecha el uso de transferencia de aprendizaje y fine-tuning para mejorar los resultados de precisión y reducir tiempos de entrenamiento.

## Conjunto de datos
Se usaron dos conjuntos de datos diferentes, uno para clasificación y otro para segmentación, esto se hizo para tener más variabilidad de los datos en los entrenamientos.
El dataset para clasificación se separó en 70% para entrenamiento, 15% para validación y 15% para prueba. Este se extrajo de la siguiente página:
https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data
<br>
El dataset usado para segmentación es perfecto para este caso debido a que tiene muestras de las mismas clases para clasificación y además incluye las máscaras de tumores para que la red neuronal aprenda a reconocer los tumores cerebrales. Este se extrajo de la siguiente página:
https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset
<br>

<div align="center">
  <img src="https://www.kenhub.com/thumbor/zoz_XVCq44UFroH2ds6eoOUvdtA=/fit-in/800x1600/filters:watermark(/images/logo_url.png,-10,-10,0):background_color(FFFFFF):format(jpeg)/images/library/13517/ff.jpg" width="300" alt="Muestra de una imagen de resonancia magnética cerebral">
  <br>
  <em>Figura 1: Muestra de una resonancia magnética cerebral (MRI).</em>
</div>

## Clasificación


## Despliegue con Docker
Este proyecto está contenido utilizando Docker, lo que permite levantar el frontend en React y la API de FastAPI en servicios aislados con un solo comando.

Los modelos de inteligencia artificial se descargan de forma dinámica desde Hugging Face Hub durante el arranque del contenedor, eliminando la necesidad de configurar pesos locales.

### Requisitos
* Tener instalado [Docker](https://docs.docker.com/get-docker/)

### Instrucciones de ejecución

1. Clonar este repositorio en la máquina local:
   ```bash
   git clone [https://github.com/tu-usuario/Clasificacion_tumores.git](https://github.com/tu-usuario/Clasificacion_tumores.git)
   cd Clasificacion_tumores

2. Generar imagen en el sistema, este proceso puede demorar debido a la descarga de los modelos y pesos desde Hugging Face:
   ``` bash
   docker compose up --build

Con esto hecho, el sistema estará operativo en las siguientes rutas locales:

Interfaz web (React): http://localhost:3000

Documentación Interactiva de la API (Swagger UI): http://localhost:8000/docs

Para detener los contenedores, se debe correr el comando:
```bash
docker compose down
