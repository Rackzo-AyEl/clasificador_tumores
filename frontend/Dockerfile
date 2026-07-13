FROM node:18-alpine

WORKDIR /app

# Copiar los archivos de configuración de paquetes
COPY package*.json ./

# Instalar las dependencias
RUN npm install

# Copiar el código fuente del frontend
COPY . .

EXPOSE 3000

# Arrancar el servidor de desarrollo de React
CMD ["npm", "start"]
