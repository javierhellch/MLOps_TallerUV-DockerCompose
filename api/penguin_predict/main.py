"""
API de inferencia para el dataset Palmer Penguins - Taller 2 MLOps.

Cambios clave vs Taller 1:
- Los modelos y el registry.json se leen desde la carpeta 'models/', que en Docker será un volumen compartido con el contenedor de Jupyter.
- Se registran logs de cada request de predicción en 'logs/predictions.log', que será un volumen solo accesible para el equipo de pruebas.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict
from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd


# -------------------------------------------------------------------
# Configuración de rutas dentro del contenedor
# -------------------------------------------------------------------

# Carpeta donde están los modelos serializados (.joblib) y registry.json
MODELS_DIR = Path("models")

# Archivo que contiene el registro de modelos disponibles
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Carpeta para logs de predicciones
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # por si no existe
PREDICTIONS_LOG_PATH = LOGS_DIR / "predictions.log"


# -------------------------------------------------------------------
# Variables globales de estado
# -------------------------------------------------------------------

# Nombre del modelo actualmente activo
ACTIVE_MODEL_NAME: Optional[str] = None

# Pipeline completo cargado en memoria (preprocesamiento + clasificador)
ACTIVE_MODEL_PIPE = None  # tipo Any

# Diccionario con la información del registry.json
REGISTRY: Optional[Dict[str, Any]] = None


# -------------------------------------------------------------------
# Modelos de datos (Pydantic)
# -------------------------------------------------------------------

class PenguinFeatures(BaseModel):
    """
    Modelo de entrada para predicción.

    Representa las características físicas de un pingüino.
    Algunos campos son opcionales porque el pipeline incluye imputación.
    """
    island: str
    bill_length_mm: Optional[float] = None
    bill_depth_mm: Optional[float] = None
    flipper_length_mm: Optional[float] = None
    body_mass_g: Optional[float] = None
    sex: Optional[str] = None
    year: int


class SelectModelRequest(BaseModel):
    """
    Modelo de entrada para cambiar el modelo activo.
    """
    model_name: str


# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------

def load_registry() -> Dict[str, Any]:
    """
    Carga el archivo registry.json.

    Salida:
        dict con:
            - default_model
            - available_models
    """
    if not REGISTRY_PATH.exists():
        raise RuntimeError(f"registry.json no existe en {REGISTRY_PATH}")

    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def load_model(model_name: str):
    """
    Carga un modelo específico desde la carpeta models.

    Entrada:
        model_name (str): nombre del modelo a cargar

    Salida:
        Pipeline entrenado

    Lanza error HTTP 404 si el modelo no existe.
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no encontrado: {model_name}"
        )

    return joblib.load(model_path)


def set_active_model(model_name: str) -> None:
    """
    Cambia el modelo activo en memoria.

    Entrada:
        model_name (str)

    Efecto:
        Actualiza ACTIVE_MODEL_PIPE y ACTIVE_MODEL_NAME
    """
    global ACTIVE_MODEL_NAME, ACTIVE_MODEL_PIPE

    ACTIVE_MODEL_PIPE = load_model(model_name)
    ACTIVE_MODEL_NAME = model_name


def log_prediction(payload: Dict[str, Any]) -> None:
    """
    Registra un log de una solicitud de predicción en formato JSON lines.

    Cada línea del archivo es un JSON con:
        - timestamp
        - model_used
        - input_features
        - prediction
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        **payload,
    }

    # Guardamos como JSON line
    with open(PREDICTIONS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# -------------------------------------------------------------------
# Inicialización de la aplicación FastAPI
# -------------------------------------------------------------------

app = FastAPI(title="Penguins Prediction API - Taller 2")


@app.on_event("startup")
def startup_event() -> None:
    """
    Se ejecuta automáticamente cuando la API arranca.

    Proceso:
        1. Carga el registry.json desde MODELS_DIR
        2. Identifica el modelo por defecto
        3. Carga ese modelo en memoria
    """
    global REGISTRY

    # Cargar registry
    REGISTRY = load_registry()

    default_model = REGISTRY.get("default_model")

    if default_model is None:
        raise RuntimeError("registry.json no tiene 'default_model' definido")

    # Cargar modelo por defecto
    set_active_model(default_model)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@app.get("/")
def home():
    """
    Endpoint de verificación.

    Salida:
        Mensaje indicando que la API está activa
        Nombre del modelo activo
    """
    return {
        "message": "API Penguins funcionando (Taller 2)",
        "active_model": ACTIVE_MODEL_NAME,
    }


@app.get("/models")
def list_models():
    """
    Devuelve información sobre los modelos disponibles.

    Salida:
        - Modelo por defecto
        - Lista de modelos disponibles
        - Modelo actualmente activo
    """
    if REGISTRY is None:
        raise HTTPException(
            status_code=500,
            detail="Registry de modelos no cargado",
        )

    return {
        "default_model": REGISTRY.get("default_model"),
        "available_models": REGISTRY.get("available_models", []),
        "active_model": ACTIVE_MODEL_NAME,
    }


@app.post("/select_model")
def select_model(req: SelectModelRequest):
    """
    Permite cambiar el modelo activo.

    Entrada:
        JSON con:
            {
                "model_name": "rf"
            }

    Proceso:
        Verifica que el modelo exista en available_models
        Si existe, lo carga en memoria
        Si no, devuelve error 404
    """
    if REGISTRY is None:
        raise HTTPException(
            status_code=500,
            detail="Registry de modelos no cargado",
        )

    available = REGISTRY.get("available_models", [])

    if req.model_name not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no disponible: {req.model_name}",
        )

    set_active_model(req.model_name)

    return {
        "message": "Modelo activo actualizado",
        "active_model": ACTIVE_MODEL_NAME,
    }


@app.post("/predict")
def predict(features: PenguinFeatures):
    """
    Realiza una predicción de especie.

    Entrada:
        JSON con las características del pingüino.

    Proceso:
        1. Convierte el input en DataFrame
        2. El pipeline aplica automáticamente:
           - imputación
           - encoding
           - transformación
           - predicción
        3. Devuelve la especie predicha
        4. Registra un log en 'logs/predictions.log'

    Salida:
        {
            "prediction": "Adelie",
            "model_used": "rf"
        }
    """

    if ACTIVE_MODEL_PIPE is None or ACTIVE_MODEL_NAME is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo no cargado",
        )

    # Convertir el JSON en un DataFrame de una sola fila
    df = pd.DataFrame([features.dict()])

    # El pipeline hace todo: preprocesamiento + predicción
    pred = ACTIVE_MODEL_PIPE.predict(df)[0]

    response = {
        "prediction": str(pred),
        "model_used": ACTIVE_MODEL_NAME,
    }

    # Registrar log (input + salida)
    log_prediction(
        payload={
            "model_used": ACTIVE_MODEL_NAME,
            "input_features": features.dict(),
            "prediction": str(pred),
        }
    )

    return response