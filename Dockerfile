# ============================================================================
# DTF Predictive Platform — Dockerfile
# ============================================================================
# Imagen base con Python 3.11 + todas las dependencias ML
# Usada por los servicios 'api' y 'dashboard' en docker-compose
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para Prophet, psycopg2, statsmodels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Puerto por defecto (API)
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1