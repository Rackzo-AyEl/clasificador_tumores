# Imagen de contenedor
FROM python:3.12-slim

# Directorio interno de contenedor
WORKDIR /app

# Definir ruta de caché de HuggingFace
ENV HF_HOME=/app/hf_cache

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
