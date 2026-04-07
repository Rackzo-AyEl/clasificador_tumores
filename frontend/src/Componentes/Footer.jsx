import '../Hojas_estilo/footer.css';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer-container">
      <div className="footer-content">
        <p className="disclaimer">
          <strong>Aviso:</strong> Esta aplicación es un prototipo desarrollado con fines
          estrictamente académicos y de investigación en el área de inteligencia artificial.
          Los resultados generados por los modelos son estimaciones
          computacionales y <strong>no representan un diagnóstico médico real</strong>.
          Consulte siempre a un neuro-radiólogo o especialista calificado para cualquier
          evaluación de salud.
        </p>
        <p className="copyright">
          &copy; {currentYear} Proyecto clasificador de tumores cerebrales
        </p>
      </div>
    </footer>
  );
};

export default Footer;
