function Inicio() {
  return (
    <div className="contenedor_contenido">
      <h2>Introducción</h2>
      <p>Los tumores cerebrales tienen una tasa de incidencia de alrededor de 10 por cada 100,000 personas al año. Es importante hacer un correcto reconocimiento de estas afectaciones para dar un tratamiento y mejorar la calidad de vida de las personas. Hacer monitoreo de tumores es un trabajo que requiere años de experiencia para los médicos, por ello se propone el uso de inteligencia artificial para ayudar con esta tarea usando técnicas que permite analizar detalles que el ojo humano es incapaz de identificar.</p>
      <p>Este sistema integra arquitecturas de aprendizaje profundo con imágenes médicas para ayudar en la clasificación y reconocimiento de tumores cerebrales con una alta precisión y confiabilidad para la clasificación y segmentación de tumores cerebrales usando MRI (Imágenes de resonancia magnética) entre 4 clases diferentes (healthy, glioma, meningioma, pituitary). Las redes propuestas sobre este trabajo son MLP (perceptrón multicapa) junto a ResNet50 para clasificación y U-Net con EfficientNet para la segmentación. Aquí se aprovecha el uso de transferencia de aprendizaje y fine-tuning para mejorar los resultados de precisión y reducir tiempos de entrenamiento.</p>
      <img
        className='imagen-pipeline'
        src={require('../Imagenes/pipeline_proyecto.png')}
        alt='Pipeline de proyecto'
      />


      <h2>Conjunto de datos</h2>
      <p>Los datos de entrada para alimentar los modelos propuestos son extraídos de repositorios en línea, los cuales se seleccionaron por la correcta estructuración de carpetas y limpieza para omitir tareas extensas de organización.</p>
      <p>Para entrenar el modelo clasificador, se empleó un dataset de 7,153 imágenes de resonancia magnética de <a href='https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data' target='_blank' rel="noreferrer">kaggle</a>, el cual se separa en las clases: 'glioma', 'meningioma', 'pituitary', 'healthy'.<br />Para la tarea de segmentación de tumores, se usó el conjunto de 5,237 muestras, también de <a href='https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset' target='_blank' rel="noreferrer">kaggle</a>. Este dataset es perfecto para el modelo segmentador, pues cuenta con una separacion de imágenes MRI así como con las máscaras binarias de tumores para el entrenamiento.</p>

      <img
        className='imagen-conjunto'
        src={require('../Imagenes/conjunto-datos.png')}
        alt='Tabla de conjunto de datos'
      />

      <h2>Evaluación y resultados</h2>
      <p>Cada modelo aplicado requiere de una evaluación para medir la fiabilidad de los resultados. Para tener una mayor seguridad de que los números obtenidos no son realistas, se ha hecho la evaluación sobre el conjunto de datos de prueba que se separó en la fase de preprocesamiento, así se evita tener fuga de datos que provoquen que a los modelos se les mide usando datos que hayan estudiadio durante el entrenamiento.</p>
      <img
        src={require('../Imagenes/metricas-evaluacion.png')}
        alt='Descripción de métricas de evaluación'
      />
      <h3>Evaluación clasificación</h3>
      <div className='contenedor_metricas_clasificacion'>
        <img
          src={require('../Imagenes/matriz-confusion.png')}
          alt='Descripción de métricas de evaluación'
        />
        <img
          src={require('../Imagenes/metricas-clasificacion.png')}
          alt='Resultados de evaluación de clasificación'
        />
      </div>

      <h3>Evaluación segmentación</h3>
      <div className='contenedor_metricas_clasificacion'>
        <img
          src={require('../Imagenes/metricas-segmentacion.png')}
          alt='Resultados de evaluación de segmentación'
        />
      </div>
    </div>
  )
}

export default Inicio;
