import '../Hojas_estilo/modelos.css';

function Modelos() {
  return (
    <div className="contenedor-contenido">
      <h1>Demo</h1>
      <p>
        <strong>Instrucciones:</strong> Para ejecutar el programa de clasificación y segmentación,
        es necesario seleccionar una imagen del catálogo de muestras desde la opción 'Seleccionar imagen' o hacer click en 'Cargar imagen'
        para usar una imagen local.
      </p>

      <div className='contenedor-demo'>

        {/* PANEL IZQUIERDO: Entrada de datos */}
        <div className='catalogo-imagenes'>
          <h3 className="titulo-panel">1. Entrada de Imagen</h3>

          <div className="grupo-botones-entrada">
            <button className="btn-secundario">Seleccionar del catálogo</button>
            <button className="btn-secundario">Cargar imagen local</button>
          </div>

          {/* Cuadro donde aparecerá la imagen seleccionada */}
          <div className="vista-previa-caja">
            <p>Aún no se ha seleccionado ninguna imagen.</p>
          </div>
        </div>

        {/* PANEL DERECHO: Acciones y Resultados */}
        <div className='contenedor-resultados'>
          <h3 className="titulo-panel">2. Resultados del Modelo</h3>

          {/* Cuadro donde aparecerán la segmentación y clasificación */}
          <div className="vista-previa-caja">
            <p>Esperando imagen para analizar...</p>
          </div>

          <div className="grupo-botones-accion">
            <button className="btn-limpiar">Limpiar resultados</button>
            <button className="btn-primario">Comenzar análisis</button>
          </div>
        </div>

      </div>
    </div>
  )
}

export default Modelos;
