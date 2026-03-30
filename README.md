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
