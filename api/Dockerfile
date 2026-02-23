# api/Dockerfile

FROM python:3.11-slim

# Instalamos uv
RUN pip install --no-cache-dir uv

# Directorio de trabajo
WORKDIR /app

# Copiamos pyproject global y lockfile
COPY pyproject.toml uv.lock ./

# Instalamos dependencias sin dev
RUN uv sync --frozen --no-dev

# Copiamos el código de la API
COPY api/penguin_predict ./penguin_predict

# Creamos directorios que usará la API (montados como volúmenes)
RUN mkdir -p /app/models /app/logs

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "penguin_predict.main:app", "--host", "0.0.0.0", "--port", "8000"]