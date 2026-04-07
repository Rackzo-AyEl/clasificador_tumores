function Teoria() {
  return (
    <div className='contenedor_contenido'>
      <h1>¿Qué es un tumor cerebral?</h1>
      <p>De acuerdo a NIH (National Institute of Neurological Disorders and Stroke), Un tumor es una masa sólida que se forma cuando las células anormales se agrupan, estos crecimientos anormales son causados por una mutación sobre el ADN que hace que las células se dupliquen rápidamente y que vivan más tiempo de lo normal. Existen muchos tipos de tumores, estos pueden ser benignos o otros malignos, lo que significa que el tumor puede estar compuesto por células anormales.
        Algunos tumores no causan sintomas, mientras que otros pueden causar afectaciones en todo el cuerpo, pues pueden generar síntomas como dolores de cabeza, dolores de espalda, convulsiones, náuseas, problemas en los sentidos (olfato, vista, oído) e incluso en el habla.
      </p>
      <p>Los tumores que se generan en el área del cerebro pueden simplemente presionar contra la masa encefáilca, o crecer dentro de la misma generando afectaciones en el tejido. Los tipos de tumores cerebrales que se analizan para este caso son glioma, meningioma y pituitaria.</p>

      <h2>Glioma</h2>
      <p>Este tipo de tumor se conforma por celulas que son muy parecidas a células sanas. Esto es un problema en la detección de tumores, pues al no tener una forma bien definida característica de este tipo de tumor, es complicado clasificar la masa para su tratamiento.</p>
      <img
        className='imagen-tumor'
        src={require('../Imagenes/gl-0001.jpg')}
        alt='Imagen MRI glioma'
      />

      <h2>Meningioma</h2>
      <p>Un meningioma es un tumor que inicia en las meninges, las cuales son las capas de membranas de tejido que envuelven y protegen el sistema nervioso central, incluyendo el cerebro. La mayoría de meningiomas crecen lentamente, pueden no causar síntomas en muchos años, pero si crecen demasiado o se sitúan cerca de áreas importantes pueden causar grandes afectaciones a la salud.</p>
      <p>Los meningiomas se presentan con más frecuencia en mujeres y en personas de edad avanzada, pero generalmente pueden presentarse a cualquier edad.</p>
      <p>Alrededor del 80% de meningiomas son benignos, por lo que no suelen representar cáncer, sin embargo es importante mantener un buen diagnóstico y monitoreo con escaneos regulares en caso de presencia de tumor.</p>
      <img
        className='imagen-tumor'
        src={require('../Imagenes/me-0305.jpg')}
        alt='Imagen MRI meningioma'
      />

      <h2>Pituitaria</h2>
      <p>La glándula pituitaria (también llamada hipófisis) es una glándula endocrina ubicada justo debajo del hipotálamo, esta regula los procesos como el crecimiento y el metabolismo.</p>
      <p>El tumor de hipófisis en la mayoría de los casos (como el tumor de meningioma) es de crecimiento lento y suelen ser benignos, en baja frecuencia de casos se expande hacia estructuras vecinas. La aparición de estos tumores puede llegar a afectar a la producción de hormonas y glándulas.</p>
      <p> Algunos síntomas comúnes de este tumor son dolores de cabeza, alteración de visión, pérdida de vello corporal, disminución del impulso sexual. En las mujeres pueden presentarse períodos menstruales menos frecuentes, ausencia de menstruación o falta de leche en las mamas. En hombres puede notarse pérdida de vello facial, crecimiento del tejido de la mama o esterilidad.</p>
      <img
        className='imagen-tumor'
        src={require('../Imagenes/pi-0272.jpg')}
        alt='Imagen MRI pituitaria'
      />
    </div>
  )
}

export default Teoria;
