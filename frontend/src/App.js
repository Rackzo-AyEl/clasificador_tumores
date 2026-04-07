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


  return (
    <div className='App'>
      <div className='contenedor-titulo'>
        <h1 className='titulo-principal'>Clasificación y segmentación de tumores cerebrales</h1>
      </div>
      <div className='contenedor-navegacion'>
        <h2 onClick={() => setSeccion('inicio')}>Inicio</h2>
        <h2 onClick={() => setSeccion('teoria')}>Tumores cerebrales</h2>
        <h2 onClick={() => setSeccion('modelos')}>Modelos y metodología</h2>
        <h2 onClick={() => setSeccion('enlaces')}>Repositorio</h2>
      </div>

      <main className='contenedor-principal'>
        {renderizarContenido()}
      </main>

      <Footer />
    </div>
  );
}

export default App;
