import { useState, useRef } from 'react';
import '../Hojas_estilo/modelos.css';

function Modelos() {
  const [imagenArchivo, setImagenArchivo] = useState(null);
  const [imagenPreview, setImagenPreview] = useState(null);
  const [resultados, setResultados] = useState(null);
  const [cargando, setCargando] = useState(false);
  const [mostrarCatalogo, setMostrarCatalogo] = useState(false);

  const inputFileRef = useRef(null);

  const imagenesCatalogo = [
    { id: 1, clase: '1. Glioma', src: require('../Imagenes/Demo/gl-0001.jpg') },
    { id: 2, clase: '2. Glioma', src: require('../Imagenes/Demo/gl-0002.jpg') },
    { id: 3, clase: '3. Glioma', src: require('../Imagenes/Demo/gl-0003.jpg') },
    { id: 4, clase: '4. Glioma', src: require('../Imagenes/Demo/gl-0004.jpg') },
    { id: 5, clase: '5. Meningioma', src: require('../Imagenes/Demo/me-0005.jpg') },
    { id: 6, clase: '6. Meningioma', src: require('../Imagenes/Demo/me-0006.jpg') },
    { id: 7, clase: '7. Meningioma', src: require('../Imagenes/Demo/me-0007.jpg') },
    { id: 8, clase: '8. Meningioma', src: require('../Imagenes/Demo/me-0008.jpg') },
    { id: 9, clase: '9. Pituitaria', src: require('../Imagenes/Demo/pi-0005.jpg') },
    { id: 10, clase: '10. Pituitaria', src: require('../Imagenes/Demo/pi-0006.jpg') },
    { id: 11, clase: '11. Pituitaria', src: require('../Imagenes/Demo/pi-0007.jpg') },
    { id: 12, clase: '12. Pituitaria', src: require('../Imagenes/Demo/pi-0008.jpg') },
    { id: 13, clase: '13. Sano', src: require('../Imagenes/Demo/no-0005.jpg') },
    { id: 14, clase: '14. Sano', src: require('../Imagenes/Demo/no-0006.jpg') },
    { id: 15, clase: '15. Sano', src: require('../Imagenes/Demo/no-0007.jpg') },
    { id: 16, clase: '16. Sano', src: require('../Imagenes/Demo/no-0008.jpg') }
  ];

  const seleccionarDelCatalogo = async (rutaImagen) => {
    try {
      setImagenPreview(rutaImagen);
      setResultados(null);
      setMostrarCatalogo(false);

      const respuesta = await fetch(rutaImagen);
      const blob = await respuesta.blob();
      const archivoSimulado = new File([blob], "imagen_catalogo.jpg", { type: blob.type });
      setImagenArchivo(archivoSimulado);
    } catch (error) {
      console.error("Error al cargar imagen del catálogo:", error);
      alert("Hubo un problema al cargar la muestra.");
    }
  };

  const manejarSeleccion = (evento) => {
    const archivo = evento.target.files[0];
    if (archivo) {
      setImagenArchivo(archivo);
      setImagenPreview(URL.createObjectURL(archivo));
      setResultados(null);
    }
  };

  const ejecutarAnalisis = async () => {
    if (!imagenArchivo) {
      alert("Por favor, selecciona una imagen primero.");
      return;
    }

    setCargando(true);
    const formData = new FormData();
    formData.append("imagen", imagenArchivo);

    try {
      const respuesta = await fetch("http://localhost:8000/procesar-mri/", {
        method: "POST",
        body: formData,
      });

      if (!respuesta.ok) throw new Error("Error en el servidor");

      const datos = await respuesta.json();
      setResultados(datos);
    } catch (error) {
      console.error("Error al procesar:", error);
      alert("No se pudo conectar con el motor de IA. Revisa tu terminal de FastAPI.");
    } finally {
      setCargando(false);
    }
  };

  const limpiarResultados = () => {
    setImagenArchivo(null);
    setImagenPreview(null);
    setResultados(null);
    if (inputFileRef.current) inputFileRef.current.value = "";
  };

  return (
    <div className="contenedor-contenido">
      <h1>Demo</h1>
      <p>
        <strong>Instrucciones:</strong> Para ejecutar el programa de clasificación y segmentación,
        selecciona una imagen del catálogo o carga una local.
      </p>

      <div className='contenedor-demo'>

        <div className='area-visual'>
          <div className="panel-imagen">
            <h3 className="titulo-panel">Imagen de entrada (MRI)</h3>
            <div className="vista-previa-caja">
              {imagenPreview ? (
                <img src={imagenPreview} alt="MRI Entrada" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <p>Aún no se ha seleccionado imagen.</p>
              )}
            </div>
          </div>

          <div className="panel-imagen">
            <h3 className="titulo-panel">Resultado de segmentación</h3>
            <div className="vista-previa-caja">
              {cargando ? (
                <p>Analizando con IA...</p>
              ) : resultados && resultados.mascara_base64 ? (
                <img src={`data:image/png;base64,${resultados.mascara_base64}`} alt="Máscara" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : resultados ? (
                <p>Segmentación completada (Sin máscara)</p>
              ) : (
                <p>Esperando análisis...</p>
              )}
            </div>
          </div>
        </div>

        <div className='area-controles'>

          <div className="barra-probabilidad-contenedor" style={{ opacity: resultados ? 1 : 0.3 }}>
            <div className="etiquetas-probabilidad">
              <span>Clase detectada: <strong>{resultados ? resultados.clase : '---'}</strong></span>
              <span>{resultados ? `${resultados.confianza}%` : '0%'}</span>
            </div>
            <div className="probabilidad-fondo">
              <div
                className="probabilidad-lleno"
                style={{ width: resultados ? `${resultados.confianza}%` : '0%' }}
              ></div>
            </div>
          </div>

          <div className="grupo-botones-general">
            <input
              type="file"
              accept="image/*"
              ref={inputFileRef}
              style={{ display: 'none' }}
              onChange={manejarSeleccion}
            />

            <button className="btn-secundario" onClick={() => setMostrarCatalogo(true)}>Catálogo</button>
            <button className="btn-secundario" onClick={() => inputFileRef.current.click()}>Cargar local</button>
            <button className="btn-limpiar" onClick={limpiarResultados}>Limpiar</button>
            <button className="btn-primario" onClick={ejecutarAnalisis} disabled={cargando}>
              {cargando ? 'Procesando...' : 'Ejecutar análisis'}
            </button>
          </div>

        </div>
      </div>

      {mostrarCatalogo && (
        <div className="modal-overlay" onClick={() => setMostrarCatalogo(false)}>
          <div className="modal-contenido" onClick={(e) => e.stopPropagation()}>
            <div className="modal-cabecera">
              <h3>Muestras clínicas</h3>
              <button className="btn-cerrar-modal" onClick={() => setMostrarCatalogo(false)}>X</button>
            </div>

            <div className="grid-catalogo">
              {imagenesCatalogo.map((imagen) => (
                <div key={imagen.id} className="tarjeta-catalogo" onClick={() => seleccionarDelCatalogo(imagen.src)}>
                  <div className="contenedor-img-catalogo">
                    <img src={imagen.src} alt={imagen.clase} />
                  </div>
                  <span className="etiqueta-clase">{imagen.clase}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

export default Modelos;
