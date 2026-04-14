function Teoria() {
  return (
    <div className='contenedor_contenido'>
      <h1 id='tumores-teoria'>¿Qué es un tumor cerebral?</h1>
      <p>
        De acuerdo con el NIH (National Institute of Neurological Disorders and Stroke), un tumor
        es una masa sólida que se forma cuando las células crecen y se dividen más de lo debido, o no mueren cuando
        deberían hacerlo. Estos crecimientos anormales son causados por mutaciones en el ADN. Existen tumores
        benignos (no cancerosos) y malignos (cancerosos), sin embargo, ambos están compuestos por células anormales
        y requieren atención médica.
      </p>
      <p>
        Algunos tumores son asintomáticos en sus primeras etapas, mientras que otros pueden causar síntomas
        sistémicos y neurológicos, como dolores de cabeza constantes, convulsiones, náuseas, o alteraciones
        en los sentidos y el habla. Los tumores en el área del cerebro pueden presionar contra la masa encefálica
        o crecer infiltrándose en el tejido sano. Los tipos de tumores cerebrales que se abordan en este
        sistema de clasificación son: glioma, meningioma y tumores de la glándula pituitaria.
      </p>

      <h2 id='glioma'>Glioma</h2>
      <p>
        Este tipo de tumor se origina en las células gliales, que son las células de soporte del cerebro.
        El mayor desafío clínico del glioma es su naturaleza infiltrante, pues sus células tienden a mezclarse y diseminarse entre el tejido cerebral sano.
        Al no poseer bordes definidos o una cápsula que lo contenga, delimitar visualmente dónde termina el tumor y dónde comienza
        el cerebro es una tarea compleja, lo que dificulta tanto su extracción quirúrgica como su segmentación digital.
      </p>
      <div className='contenedor_imagen'>
        <img
          className='imagen-tumor'
          src={require('../Imagenes/gl-0001.jpg')}
          alt='Imagen MRI glioma'
          style={{
            width: '50%',
            maxWidth: 300,
            borderRadius: 15
          }}
        />
      </div>

      <h2 id='meningioma'>Meningioma</h2>
      <p>Un meningioma es un tumor que inicia en las meninges, las cuales son las capas de membranas de tejido que envuelven y protegen el sistema nervioso central, incluyendo el cerebro. La mayoría de meningiomas crecen lentamente, pueden no causar síntomas en muchos años, pero si crecen demasiado o se sitúan cerca de áreas importantes pueden causar grandes afectaciones a la salud.</p>
      <p>Los meningiomas se presentan con más frecuencia en mujeres y en personas de edad avanzada, pero generalmente pueden presentarse a cualquier edad.</p>
      <p>Alrededor del 80% de meningiomas son benignos, por lo que no suelen representar cáncer, sin embargo es importante mantener un buen diagnóstico y monitoreo con escaneos regulares en caso de presencia de tumor.</p>
      <div className='contenedor_imagen'>
        <img
          className='imagen-tumor'
          src={require('../Imagenes/me-0305.jpg')}
          alt='Imagen MRI meningioma'
          style={{
            width: '50%',
            maxWidth: 300,
            borderRadius: 15
          }}
        />
      </div>

      <h2 id='pituitaria'>Pituitaria</h2>
      <p>La glándula pituitaria (también llamada hipófisis) es una glándula endocrina ubicada justo debajo del hipotálamo, esta regula los procesos como el crecimiento y el metabolismo.
        El tumor de hipófisis en la mayoría de los casos (como el tumor de meningioma) es de crecimiento lento y suelen ser benignos, en baja frecuencia de casos se expande hacia estructuras vecinas. La aparición de estos tumores puede llegar a afectar a la producción de hormonas y glándulas.</p>
      <p> Algunos síntomas comúnes de este tumor son dolores de cabeza, alteración de visión, pérdida de vello corporal, disminución del impulso sexual. En las mujeres pueden presentarse períodos menstruales menos frecuentes, ausencia de menstruación o falta de leche en las mamas. En hombres puede notarse pérdida de vello facial, crecimiento del tejido de la mama o esterilidad.</p>
      <div className='contenedor_imagen'>
        <img
          className='imagen-tumor'
          src={require('../Imagenes/pi-0272.jpg')}
          alt='Imagen MRI pituitaria'
          style={{
            width: '50%',
            maxWidth: 300,
            borderRadius: 15
          }}
        />
      </div>

      <h1 id='redes-neuronales'>Redes neuronales</h1>
      <p>
        El aprendizaje profundo permite a las máquinas reconocer patrones complejos mediante redes neuronales multicapa,
        facilitando el análisis de grandes volúmenes de datos. Sin embargo, diseñar y entrenar una
        arquitectura de este tipo desde cero es una enorme inversión de tiempo y poder computacional.
      </p>
      <p>
        Para optimizar este proceso se utiliza la transferencia de aprendizaje (Transfer learning),
        que consiste en aprovechar el conocimiento de una red previamente entrenada y aplicarlo a un problema distinto,
        ahorrando un esfuerzo significativo. Además, mediante el ajuste fino (Fine-Tuning),
        se ajustan ligeramente los parámetros de esta red para que aprenda las características específicas de nuestro
        nuevo conjunto de datos, mejorando así su precisión.
      </p>
      <div className='contenedor_imagen'>
        <img
          src={require('../Imagenes/redes-neuronales.png')}
          alt='Imagen de redes neuronales IBM'
          style={{
            width: '100%',
            maxWidth: 600
          }}
        />
      </div>

      <div className="bloque-arquitecturas">
        <h1 id='arquitecturas'>Arquitecturas de aprendizaje profundo aplicadas</h1>
        <p>
          Para abordar la complejidad de las imágenes médicas, este sistema divide el problema en dos tareas
          asignadas a redes neuronales especializadas: descubrir qué tipo de tumor está presente (clasificación) y
          delimitar exactamente dónde se encuentra (segmentación).
        </p>

        <h3>Clasificación: ResNet-50 y perceptrón multicapa (MLP)</h3>
        <p>
          Para la tarea de clasificación se implementó ResNet-50 (Red residual de 50 capas).
          El mayor problema de las redes neuronales muy profundas es el "desvanecimiento del gradiente", donde la red
          olvida lo aprendido en las primeras capas. ResNet soluciona esto mediante conexiones residuales,
          permitiendo que la información salte capas y fluya directamente hacia el final de la red.
        </p>
        <p>
          En este proyecto, ResNet-50 actúa como el extractor de características visuales.
          Aprovechando el transfer learning, la red ya sabe identificar texturas y bordes complejos. Posteriormente,
          sus capas finales fueron reemplazadas por un perceptrón multicapa (MLP) adaptado específicamente
          para tomar esas características visuales y emitir un diagnóstico final entre las 4 clases médicas (Sano, Glioma, Meningioma o Pituitaria).
        </p>
        <div className='contenedor_imagen'>
          <img
            src={require('../Imagenes/resnet50.png')}
            alt='Arquitectura de ResNet-50'
          />
        </div>

        <h3>Segmentación: U-Net con encoder EfficientNet</h3>
        <p>
          Una vez clasificado el tumor, el sistema debe mapearlo píxel por píxel. Para esto se utilizó la arquitectura
          U-Net, una arquitectura especializada en la segmentación de imágenes médicas. Su diseño en forma de "U"
          consiste en una ruta de contracción (que captura el contexto general) y una ruta de expansión (que recupera la precisión espacial).
        </p>
        <p>
          Para mejorar el rendimiento, el diseño original de U-Net fue modificado reemplazando su ruta de contracción estándar
          por un encoder EfficientNet. EfficientNet es una arquitectura que escala
          matemáticamente la profundidad, anchura y resolución de la red de forma equilibrada. Al fusionar el poder de
          extracción de EfficientNet con la capacidad de localización precisa de U-Net, el modelo logra generar
          máscaras binarias detalladas, incluso en los bordes difusos de patologías complejas como los gliomas.
        </p>
      </div>
      <div className='contenedor_imagen'>
        <img
          src={require('../Imagenes/u-net.png')}
          alt='Arquitectura de U-Net'
        />
      </div>

      <div className="bloque-metodologia">
        <h1 id='metodologia'>Metodología</h1>
        <p>
          Para garantizar que las redes neuronales aprendan las características biológicas de los tumores y no memoricen
          imágenes específicas, el flujo de trabajo se divide en fases de preparación de datos y procesamiento de optimizado para un alto rendimiento.
        </p>

        <h3>1. Preprocesamiento de datos</h3>
        <p>
          Antes de alimentar los modelos, las imágenes crudas de resonancia magnética (MRI) y sus respectivas máscaras
          binarias deben ser estandarizadas. En esta fase, todas las imágenes son redimensionadas a una escala uniforme
          para cumplir con los requisitos de entrada de las arquitecturas ResNet-50 y U-Net. Además, se aplica una
          normalización de píxeles, ajustando los valores de intensidad para que el modelo no se vea sesgado
          por diferencias en la iluminación o el contraste de los distintos escáneres médicos. También, se hace una
          partición de los datos en conjuntos de entrenamiento, validación y prueba, asegurando que no exista
          fuga de información.
        </p>

        <h3>2. Aumento de datos</h3>
        <p>
          En el campo médico, la cantidad de imágenes etiquetadas suele ser limitada. Para prevenir el sobreajuste (overfitting)
          y hacer que el modelo sea más robusto ante casos del mundo real, se implementó una técnica de aumento de datos.
          Durante el entrenamiento, las imágenes originales sufren transformaciones matriciales aleatorias en tiempo real,
          tales como rotaciones, espejado horizontal y ligeros recortes. Esto obliga a las redes convolucionales
          a aprender la forma fundamental del tumor (glioma, meningioma o pituitaria) independientemente de su orientación
          o posición exacta dentro del cráneo.
        </p>

        <h3>3. Entrenamiento y aceleración por hardware (GPU)</h3>
        <p>
          Entrenar modelos profundos de clasificación y segmentación requiere realizar millones de operaciones matemáticas
          (multiplicación de matrices) simultáneamente. Para hacer esto viable, el entrenamiento de ambos modelos se aceleró
          mediante el uso de una tarjeta de video (unidad de procesamiento gráfico o GPU) de Nvidia. Gracias a los
          núcleos CUDA y al procesamiento en paralelo, los tiempos de cálculo se reducen de semanas a tan solo unas horas.
          Este poder de cómputo permitió iterar múltiples veces sobre la arquitectura, ajustar los hiperparámetros y aplicar
          el fine-tuning necesario para alcanzar métricas de evaluación superiores al 90%.
        </p>
      </div>

      <h1 id='evaluacion'>Evaluación y resultados</h1>
      <p>Cada modelo aplicado requiere de una evaluación para medir la fiabilidad de los resultados. Para tener una mayor seguridad de que los números obtenidos son realistas, por ello se ha hecho la evaluación sobre el conjunto de datos de prueba que se separó en la fase de preprocesamiento, así se evita tener fuga de datos que provoquen que a los modelos se les mida usando datos que hayan estudiado durante el entrenamiento.</p>
      <div className='contenedor_imagen'>
        <img
          src={require('../Imagenes/metricas-evaluacion.png')}
          alt='Descripción de métricas de evaluación'
          style={{
            width: '80%',
            maxWidth: 600
          }}
        />
      </div>

      <h2>Evaluación de clasificación</h2>

      <p>Los resultados de la clasificación son altos gracias al proceso de incremento de datos y un ajuste fine de la red convolucional aplicada *AGREGAR REFERENCIA DE OTRA SECCIÓN*, por lo que la computadora fue capaz de detectar los detalles que describen a cada clase. En promedio, se obtuvo un accuracy de 0.97 y un F1-Score de 0.97.</p>
      <div className='contenedor_imagen'>
        <img
          src={require('../Imagenes/metricas-clasificacion.png')}
          alt='Resultados de evaluación de clasificación'
          style={{
            width: '100%',
            maxWidth: 500
          }}
        />
      </div>

      <p>La matriz de confusión ayuda a ver mejor el comportamiento entre las predicciones del modelo y las clases reales de los datos de prueba. La clase que es más complicada de diferenciar es 'glioma' debido a sus características biológicas, esto se puede ver en la siguiente imagen donde hay más confusiones entre esta clase y 'meningioma'. Sin embargo, el comportamiento del modelo es bastante bueno sobre los datos de prueba.</p>
      <div className='contenedor_imagen'>
        <img
          src={require('../Imagenes/matriz-confusion.png')}
          alt='Descripción de métricas de evaluación'
          style={{
            width: '100%',
            maxWidth: 650
          }}
        />
      </div>

      <h2>Evaluación de segmentación</h2>
      <p>El modelo encargado para la segmentación de tumores ha tenido resultados variados dependiendo de la clase. Con números superiores al 81% en DC e IoU para las clases 'meningioma' y 'pituitary', se puede confirmar que las máscaras binarias son buenas y tienen un gran acierto por los píxeles de las imágenes aún considerando que para la segunda clase los conjuntos de valores positivos son menores. Por otro lado, la clase 'glioma' no ha tenido tan buenos resultados, pues como se ha mencionado anteriormente, las características de este tipo de tumores no son tan definidas y los bordes pueden tener desvanecimientos en MRI que confunden al modelo con tejido cerebral.</p>
      <div className='contenedor_imagen'>
        <img
          src={require('../Imagenes/metricas-segmentacion.png')}
          alt='Resultados de evaluación de segmentación'
          style={{
            width: '100%',
            maxWidth: 550
          }}
        />
      </div>

      <p>Aunque los resultados obtenidos son satisfactorios, es claro que se requiere mayor esfuerzo por mejorar los resultados de la clase glioma,
        por lo cual, se trabajará a futuro para reentrenar el modelo de segmentación y números e imágenes con la calidad que comparten las clases restantes.
      </p>
    </div>
  )
}

export default Teoria;
