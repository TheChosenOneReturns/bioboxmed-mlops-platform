# --- Etapa 1: Construcción (Build Stage) ---
# Usamos una imagen de Node.js para construir nuestro proyecto
FROM node:18 AS build

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los archivos de configuración de dependencias
COPY package.json ./
COPY package-lock.json ./

# Instalamos todas las dependencias
RUN npm install

# Copiamos todo el código fuente de nuestra app
COPY . .

# Creamos la versión de producción optimizada de nuestra app
RUN npm run build

# --- Etapa 2: Producción (Production Stage) ---
# Usamos una imagen de Nginx, un servidor web ligero y muy rápido
FROM nginx:stable-alpine

# Copiamos la build optimizada de la etapa anterior a la carpeta que Nginx usa para servir archivos
COPY --from=build /app/build /usr/share/nginx/html

# Le decimos al contenedor que exponga el puerto 80 (el puerto estándar para web)
EXPOSE 80

# El comando por defecto de Nginx se encargará de iniciar el servidor
CMD ["nginx", "-g", "daemon off;"]