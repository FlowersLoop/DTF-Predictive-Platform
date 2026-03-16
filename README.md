# Plataforma de Análisis Predictivo con IA para Gestión de Demanda en Marcas de Moda DTF

> Proyecto de titulación — Enero 2026 – Mayo 2026

Plataforma de análisis predictivo basada en inteligencia artificial para pronosticar la demanda de diseños y productos impresos bajo tecnología **Direct-to-Film (DTF)**. El sistema integra datos históricos de ventas y señales externas (Google Trends) para generar pronósticos accionables que optimicen las decisiones de producción e inventario.

---

## Problema

Las marcas de moda que operan con producción DTF de bajo volumen enfrentan:

- **Sobreproducción** — se imprimen diseños que no se venden, generando pérdidas.
- **Desabasto (stockouts)** — diseños populares se agotan por falta de anticipación.
- **Decisiones intuitivas** — la producción se basa en experiencia, no en datos.
- **Oportunidades perdidas** — tendencias emergentes se detectan demasiado tarde.

La gestión reactiva basada en análisis histórico descriptivo resulta insuficiente frente al dinamismo del mercado de moda digital.

## Solución

Un pipeline end-to-end que ingiere datos de ventas, los transforma en features predictivos, entrena y compara tres modelos de forecasting, y expone los pronósticos mediante una API REST y un dashboard interactivo.

### Resultados obtenidos

| Modelo | MAE | MAPE | R² | Mejora vs Baseline |
|--------|-----|------|----|--------------------|
| Baseline Naive | 0.7812 | 81.2% | -0.2363 | — |
| Seasonal Forecast | 0.9903 | 100.0% | -0.6081 | -26.8% |
| Random Forest | 0.6878 | 63.6% | 0.3117 | +12.0% |
| **Gradient Boosting** | **0.4165** | **32.8%** | **0.6418** | **+46.7%** |

El modelo Gradient Boosting supera el objetivo de mejora del 20–25% con un **+46.7%** de reducción en MAE respecto al baseline.

---

## Arquitectura

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐     ┌──────────────┐
│  Fuentes Datos  │────▶│  Módulo ETL       │────▶│  Entrenamiento ML  │────▶│  API FastAPI  │
│                 │     │  Python/Pandas    │     │                    │     │              │
│ • CSV Ventas    │     │ • Ingesta/merge   │     │ • SARIMA           │     │ POST /train  │
│ • Google Trends │     │ • Limpieza        │     │ • Random Forest    │     │ GET  /predict│
│                 │     │ • Feature Eng.    │     │ • XGBoost          │     │ GET  /trends │
└─────────────────┘     └──────────────────┘     └────────────────────┘     └──────┬───────┘
                                                                                   │
                                                                          ┌────────▼───────┐
                                                                          │   Dashboard    │
                                                                          │   Streamlit    │
                                                                          │                │
                                                                          │ • Pronósticos  │
                                                                          │ • Rankings     │
                                                                          │ • Alertas DTF  │
                                                                          └────────────────┘
```

---

## Estructura del repositorio

```
dtf-predictive-platform/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                          # Datos originales (Excel de ventas)
│   │   └── DTF_s_DATA_CORRECT.xlsx
│   └── processed/                    # Datos limpios generados por ETL
│       ├── ventas_limpias.csv
│       ├── serie_semanal.csv
│       ├── features_modelo.csv
│       ├── baseline_naive.csv
│       ├── resumen_eda.csv
│       └── pipeline_metadata.json
│
├── etl/
│   └── etl_pipeline.py              # Pipeline ETL completo
│
├── models/
│   ├── train_models.py              # Entrenamiento SARIMA, RF, XGBoost
│   └── saved/                        # Modelos serializados (.pkl)
│       ├── random_forest_model.pkl
│       ├── gradient_boosting_model.pkl
│       ├── comparacion_modelos.csv
│       ├── feature_importance.csv
│       ├── predicciones_completas.csv
│       └── training_report.json
│
├── api/
│   └── main.py                       # API REST FastAPI
│
├── dashboard/
│   └── app.py                        # Dashboard Streamlit (en desarrollo)
│
├── docs/
│   └── arquitectura.md               # Documentación técnica
│
└── notebooks/
    └── eda_exploratorio.ipynb        # Análisis exploratorio
```

---

## Instalación

### Prerrequisitos

- Python 3.10+
- pip

### Setup

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/dtf-predictive-platform.git
cd dtf-predictive-platform

# Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución rápida

```bash
# 1. Ejecutar pipeline ETL (limpieza y transformación de datos)
python etl/etl_pipeline.py

# 2. Entrenar modelos
python models/train_models.py

# 3. Levantar API
cd api && uvicorn main:app --reload --port 8000

# 4. Abrir documentación interactiva de la API
# Navegar a http://localhost:8000/docs
```

---

## API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Health check y estado del sistema |
| `GET` | `/predict/{categoria}` | Pronóstico por categoría (ej: `/predict/Gym`) |
| `GET` | `/predict` | Pronóstico de todas las categorías |
| `GET` | `/trends` | Ranking de categorías por demanda predicha |
| `GET` | `/metrics` | Métricas de evaluación de modelos |
| `GET` | `/features` | Feature importance (qué variables predicen más) |
| `GET` | `/recommendations` | Recomendaciones de producción DTF |
| `GET` | `/history/{categoria}` | Historial de ventas semanales |
| `POST` | `/train` | Re-entrenar modelos con datos actualizados |

### Ejemplo de uso

```bash
# Obtener pronóstico de la categoría Gym
curl http://localhost:8000/predict/Gym

# Obtener ranking de tendencias
curl http://localhost:8000/trends

# Obtener recomendaciones de producción
curl http://localhost:8000/recommendations
```

---

## Modelos de Machine Learning

### 1. SARIMA (Seasonal ARIMA)
Captura patrones de estacionalidad y tendencia en series temporales. Ideal para detectar ciclos semanales y mensuales en la demanda.

### 2. Random Forest
Ensemble de árboles de decisión que expone la importancia relativa de cada feature. Robusto frente a outliers y datos faltantes.

### 3. XGBoost / Gradient Boosting
Modelo de gradient boosting para predicción tabular de alta precisión. Mejor rendimiento general con +46.7% de mejora sobre el baseline.

### Ensemble
Promedio ponderado de los tres modelos (RF 30%, GB 50%, SARIMA 20%) para generar predicciones más estables.

### Features predictivos principales

| Feature | Importancia | Interpretación |
|---------|-------------|----------------|
| `cambio_semanal` | 0.55 | Aceleración de demanda — el mejor predictor |
| `lag_1w` | 0.11 | Ventas de la semana pasada |
| `ventas_acumuladas` | 0.07 | Popularidad acumulada de la categoría |
| `rolling_mean_2w` | 0.04 | Promedio móvil de 2 semanas |
| `semana_iso` | 0.03 | Semana del año (estacionalidad) |

---

## Datos

El dataset actual contiene **54 transacciones** de una marca DTF operando desde octubre 2025, con las siguientes dimensiones:

- **13 categorías**: Sports, Gym, Fútbol, Basketball, Tenis, Casual, Movies, Música, etc.
- **4 tipos de prenda**: T-Shirt (69%), Jacket (13%), Hoodie (9%), Long Sleeve (9%)
- **14 estados** de México + 1 internacional (Canadá)
- **37 diseños únicos**

### Hallazgos clave en los datos

- **Rotación mensual de categorías dominantes**: Fútbol en diciembre, Gym en enero-febrero, Sports en marzo.
- **Bursts de demanda**: Track & Field representó 14 de 16 ventas en marzo, todas desde CDMX.
- **Concentración geográfica**: CDMX concentra el 47% de las ventas totales.

---

## Metodología

El desarrollo sigue la metodología **APQP** (Advanced Product Quality Planning) adaptada a software:

| Fase | Periodo | Descripción |
|------|---------|-------------|
| 1. Planeación | Ene 12 – Mar 14 | Definición del problema, requerimientos, stack |
| 2. Diseño | Mar 15 – Mar 29 | Arquitectura, pipeline ETL, diseño de API |
| 3. Implementación | Mar 28 – Abr 21 | ETL, modelos ML, API, dashboard |
| 4. Validación | Abr 20 – Abr 27 | Métricas, pruebas end-to-end |
| 5. Cierre | Abr 28 – May 4 | Documentación, presentación oral |

---

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.10+ |
| ETL | pandas, numpy |
| ML | scikit-learn, XGBoost, statsmodels |
| API | FastAPI, uvicorn |
| Dashboard | Streamlit |
| Señales externas | pytrends (Google Trends API) |
| Serialización | pickle, JSON |
| Versionado | Git / GitHub |

---

## Métricas de evaluación

- **MAE** (Mean Absolute Error): error promedio en unidades. Menor = mejor.
- **MAPE** (Mean Absolute Percentage Error): error porcentual. Menor = mejor.
- **R²** (Coeficiente de determinación): varianza explicada. Mayor = mejor (máximo 1.0).

El objetivo del proyecto es una **mejora mínima del 20–25%** en MAE respecto al baseline naive. El resultado actual es **+46.7%**.

---

## Trabajo futuro

- Integración completa de **Google Trends API** como feature externo para mejorar detección de bursts.
- **Sistema de recomendación** tipo collaborative filtering ("clientes que compraron X también compraron Y").
- **Alertas automáticas** cuando el modelo detecta un burst inminente de demanda.
- Migración a **PostgreSQL** cuando el volumen de datos lo justifique.
- **MLflow** para versionado y tracking de experimentos de modelos.
- Containerización con **Docker** para despliegue en nube.

---

## Autor
Santiago Tapia & Fernando Flores

## Licencia

Uso académico. Todos los derechos reservados.
