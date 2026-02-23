FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copiamos metadata primero (mejor cache)
COPY pyproject.toml uv.lock ./

# Instalamos dependencias sin dev
RUN uv sync --frozen --no-dev

# Copiamos el código de la API
COPY api/penguin_predict ./penguin_predict

# Creamos directorios que usará la API (montados como volúmenes)
RUN mkdir -p /app/models /app/logs

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "penguin_predict.main:app", "--host", "0.0.0.0", "--port", "8000"]