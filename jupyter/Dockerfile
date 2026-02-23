# jupyter/Dockerfile

#Imagen base oficial de Python 3.11 en su versión slim (sin herramientas extra), reduce el tamaño final de la imagen.
FROM python:3.11-slim

#Copia el binario de UV directamente desde su imagen oficial en GitHub Container Registry.
#Lo instala en /usr/local/bin para que esté disponible globalmente sin instalación adicional.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

#Establece /workspace como directorio de trabajo. Todos los comandos siguientes se ejecutan desde aquí.
WORKDIR /workspace

#Copia el archivo de dependencias al contenedor antes de instalar, aprovechando el cache de capas de Docker.
COPY pyproject.toml uv.lock ./

#UV lee pyproject.toml y crea un entorno virtual con todas las dependencias declaradas. Equivalente a pip install, pero más rápido.
RUN uv sync

#Documenta que el contenedor usa el puerto 8888 (JupyterLab). No abre el puerto — eso lo hace docker-compose.yml.
EXPOSE 8888

#Comando por defecto al iniciar el contenedor. uv run ejecuta dentro del entorno virtual.
CMD ["uv", "run", "jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]

     # escucha todas las interfaces de red.
     # puerto 8888 en el que JupyterLab escucha dentro del container
     # container no tiene interfaz gráfica, por lo tanto, no intenta abrir browser
     # ejecuta JupyterLab como usuario root
     # se desactiva el token de autenticación (ambientes de desarrollo)
     # se desactiva la contraseña (ambientes de desarrollo)