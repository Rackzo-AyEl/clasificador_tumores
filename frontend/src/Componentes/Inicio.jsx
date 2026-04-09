import '../Hojas_estilo/inicio.css'

function Inicio() {
  return (
    <div className="contenedor_contenido">
      <h2>Introducción</h2>
      <p>Los tumores cerebrales tienen una tasa de incidencia de alrededor de 10 por cada 100,000 personas al año. Es importante hacer un correcto reconocimiento de estas afectaciones para dar un tratamiento adecuado y mejorar la calidad de vida de las personas. Hacer monitoreo de tumores es un trabajo que requiere años de experiencia para los médicos, por ello se propone el uso de inteligencia artificial como herramienta para ayudar con esta tarea usando técnicas que permite analizar detalles que el ojo humano es incapaz de identificar.</p>
      <p>Este sistema integra arquitecturas de aprendizaje profundo con imágenes médicas para ayudar en la clasificación y segmentación de tumores cerebrales con una alta precisión y confiabilidad usando MRI (Imágenes de resonancia magnética) entre 4 clases diferentes (sano, glioma, meningioma, pituitaria). Las redes propuestas sobre este trabajo son MLP (perceptrón multicapa) junto a ResNet50 para clasificación y U-Net con EfficientNet para la segmentación. Aquí se aprovecha el uso de transferencia de aprendizaje y fine-tuning para mejorar los resultados y reducir tiempos de entrenamiento.</p>
      <div className='contenedor_imagen'>
        <img
          src={require('../Imagenes/pipeline_proyecto.png')}
          alt='Pipeline de proyecto'
        />
      </div>


      <h2>Conjunto de datos</h2>
      <p>Los datos de entrada para alimentar los modelos propuestos son extraídos de repositorios en línea, los cuales se seleccionaron por su estructura de carpetas y calidad.</p>
      <p>Para entrenar el modelo clasificador, se empleó un dataset de 7,153 imágenes de resonancia magnética de <a href='https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data' target='_blank' rel="noreferrer">kaggle</a>, el cual se separa en las clases: 'glioma', 'meningioma', 'pituitary', 'healthy'.<br />Para la tarea de segmentación de tumores, se usó el conjunto de 5,237 muestras, también de <a href='https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset' target='_blank' rel="noreferrer">kaggle</a>. Este dataset es perfecto para el modelo segmentador, pues cuenta con una separacion de imágenes MRI así como con las máscaras binarias de tumores para el entrenamiento.</p>

      <div className='contenedor_imagen'>
        <img
          src={require('../Imagenes/conjunto-datos.png')}
          alt='Tabla de conjunto de datos'
        />
      </div>

      <h1>Muestras de ejecución</h1>
      <p>*APLICAR IMÁGENES DE MUESTRA*</p>
    </div>
  )
}

export default Inicio;
