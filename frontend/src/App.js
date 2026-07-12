import { useState } from 'react';
import './App.css';
import Inicio from './Componentes/Inicio.jsx';
import Modelos from './Componentes/Modelos.jsx';
import Teoria from './Componentes/Teoria.jsx';
import Enlaces from './Componentes/Enlaces.jsx';
import Footer from './Componentes/Footer.jsx';

function App() {
  // Hook de pestaña
  const [seccion, setSeccion] = useState('inicio');

  // Función para renderizar cada sección
  const renderizarContenido = () => {
    switch (seccion) {
      case 'inicio': return <Inicio />
      case 'teoria': return <Teoria />
      case 'modelos': return <Modelos />
      case 'enlaces': return <Enlaces />
      default: return <Inicio />
    }
  }


  // Función para renderizar índice
  const renderizarIndice = () => {
    return (
      <div className="indice-global">

        <div className="grupo-indice">
          <h4
            onClick={() => setSeccion('inicio')}
            className={seccion === 'inicio' ? 'seccion-activa' : ''}
          >
            Inicio
          </h4>
          <ul>
            <li><a href="#introduccion" onClick={() => setSeccion('inicio')}>Introducción</a></li>
            <li><a href="#datos" onClick={() => setSeccion('inicio')}>Conjunto de datos</a></li>
            <li><a href="#muestra" onClick={() => setSeccion('inicio')}>Muestra de ejecución</a></li>
          </ul>
        </div>

        <div className="grupo-indice">
          <h4
            onClick={() => setSeccion('teoria')}
            className={seccion === 'teoria' ? 'seccion-activa' : ''}
          >
            Teoría y metodología
          </h4>
          <ul>
            <li><a href="#tumores-teoria" onClick={() => setSeccion('teoria')}>¿Qué es un tumor?</a></li>
            <li><a href="#glioma" onClick={() => setSeccion('teoria')}>Glioma</a></li>
            <li><a href="#meningioma" onClick={() => setSeccion('teoria')}>Meningioma</a></li>
            <li><a href="#pituitaria" onClick={() => setSeccion('teoria')}>Pituitaria</a></li>
            <li><a href="#redes-neuronales" onClick={() => setSeccion('teoria')}>Redes neuronales</a></li>
            <li><a href="#arquitecturas" onClick={() => setSeccion('teoria')}>Arquitecturas</a></li>
            <li><a href="#metodologia" onClick={() => setSeccion('teoria')}>Metodología</a></li>
            <li><a href="#evaluacion" onClick={() => setSeccion('teoria')}>Evaluación y resultados</a></li>
          </ul>
        </div>

        <div className="grupo-indice">
          <h4
            onClick={() => setSeccion('modelos')}
            className={seccion === 'modelos' ? 'seccion-activa' : ''}
          >
            Demo interactiva
          </h4>
        </div>

        <div className="grupo-indice">
          <h4
            onClick={() => setSeccion('enlaces')}
            className={seccion === 'enlaces' ? 'seccion-activa' : ''}
          >
            Enlaces
          </h4>
        </div>

      </div>
    );
  }

  return (
    <div className='App'>
      <div className='contenedor-titulo'>
        <h1 className='titulo-principal' style={{ fontWeight: 'bold' }}>Clasificación y segmentación de tumores cerebrales</h1>
      </div>

      <div className='contenedor-navegacion'>
        <h2 onClick={() => setSeccion('inicio')}>Inicio</h2>
        <h2 onClick={() => setSeccion('teoria')}>Teoría y metodología</h2>
        <h2 onClick={() => setSeccion('modelos')}>Demo</h2>
        <h2 onClick={() => setSeccion('enlaces')}>Enlaces</h2>
      </div>

      <div className='layout-cuerpo'>

        <aside className='contenedor-indice'>
          <h2 className='header-indice'>Índice</h2>
          {renderizarIndice()}
        </aside>

        <main className='contenedor-principal'>
          {renderizarContenido()}
        </main>

      </div>

      <Footer />
    </div>
  );
}

export default App;
