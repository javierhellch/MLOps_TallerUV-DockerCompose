# Penguins Species Classification API – MLOps Taller 1 - UV_DockerCompose_Jupyter

# Presentado Por
- Jacobo Orozco Ardila
- Javier Chaparro

## Descripción

Este proyecto implementa un flujo completo de Machine Learning utilizando el dataset Palmer Penguins.  
Se desarrolla el entrenamiento de múltiples modelos y posteriormente se expone un servicio de inferencia mediante FastAPI, el cual es contenerizado usando Docker.

El objetivo principal de este taller es construir entorno de desarrollo con Docker Compose y JupyterLab e integrarlo con UV.

---

## Estructura del Proyecto

```
Repo/
├── docker-compose.yml
├── jupyter/
│   ├── Dockerfile
│   └── pyproject.toml
├── models/              # Tu carpeta existente con .joblib
├── notebooks/
│   └── tu_notebook.ipynb
└── .gitignore
```

---

## Dataset

Se utiliza la librería `palmerpenguins` para descargar los datos.

El problema consiste en predecir la especie del pingüino:

- Adelie
- Gentoo
- Chinstrap

---

## Proceso de Entrenamiento

El entrenamiento se realiza en el notebook `entrenamiento_pinguinos.ipynb`.

Etapas principales:

1. Carga de datos
2. Separación en variables X y Y
3. División en entrenamiento y validación
4. Definición de pipelines:
   - Imputación para variables numéricas
   - Imputación + OneHotEncoding para variables categóricas
5. Entrenamiento de múltiples modelos
6. Evaluación básica con accuracy
7. Serialización de modelos usando `joblib`
8. Creación de `registry.json` con:
   - Modelo por defecto
   - Modelos disponibles

---

## API – FastAPI

La API permite:

- Consultar el estado del servicio
- Listar modelos disponibles
- Seleccionar el modelo activo
- Realizar predicciones

### Endpoints disponibles

GET `/`  
Retorna estado del servicio y modelo activo.

GET `/models`  
Lista modelos disponibles y modelo activo.

POST `/select_model`  
Permite cambiar el modelo activo.

POST `/predict`  
Recibe las características de un pingüino y retorna la predicción.

---

## Archivos de Docker Compose y JupyterLab + UV
1. Dockerfile de JupyterLab: /jupyter/Dockerfile
2. Definir dependencias con UV: /jupyter/pyproject.toml
3. Docker Compose: ./docker-compose.yaml  

Líneas de ejecución

# Construir la imagen
docker compose build

# Levantar el servicio
docker compose up -d

# Ver logs
docker compose logs -f jupyter

# Acceder a JupyterLab a través de la URL: http://127.0.0.1:8888/
Dentro del JupyterLab ingresar al directorio "notebooks" y ejecutar el notebook "entrenamiento_pinguinos.ipynb".
Los modelos quedaran almacenados en el directorio compartido "models". 

