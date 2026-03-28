# Plataforma de Análisis Predictivo con IA para Gestión de Demanda en Marcas de Moda DTF

> Proyecto de titulación — Enero 2026 – Mayo 2026  
> Santiago Tapia & Fernando Flores

Plataforma de análisis predictivo basada en inteligencia artificial para pronosticar la demanda de diseños y productos impresos bajo tecnología **Direct-to-Film (DTF)**. El sistema integra datos históricos de ventas, transferencia de patrones estacionales desde el dataset H&M (31.7M transacciones), y señales externas (Google Trends) para generar pronósticos accionables que optimicen las decisiones de producción e inventario.

---

## Problema

Las marcas de moda que operan con producción DTF de bajo volumen enfrentan:

- **Sobreproducción** — se imprimen diseños que no se venden, generando pérdidas.
- **Desabasto (stockouts)** — diseños populares se agotan por falta de anticipación.
- **Decisiones intuitivas** — la producción se basa en experiencia, no en datos.
- **Oportunidades perdidas** — tendencias emergentes se detectan demasiado tarde.

La gestión reactiva basada en análisis histórico descriptivo resulta insuficiente frente al dinamismo del mercado de moda digital.

---

## Solución

Un pipeline end-to-end que combina dos enfoques complementarios:

1. **Modelos ML independientes** (Random Forest, XGBoost, SARIMA) entrenados sobre features de la serie temporal de ventas DTF.
2. **Modelo de transferencia estadística** que extrae patrones estacionales macro del dataset H&M (31.7M transacciones) y los calibra al volumen real de DTF Fashion.

Ambos enfoques se combinan en un **ensemble con pesos optimizados** y se exponen mediante una API REST y un dashboard interactivo.

### Resultados obtenidos

| Modelo | MAE | MAPE | R² | Mejora vs Baseline |
|--------|-----|------|----|--------------------|
| Baseline Naive | 0.7812 | 81.2% | -0.2363 | — |
| SARIMA(1,1,1)(1,1,1)[7] | — | — | — | — |
| Random Forest | 0.6878 | 63.6% | 0.3117 | +12.0% |
| **XGBoost** | **0.4165** | **32.8%** | **0.6418** | **+46.7%** |
| Transferencia H&M | — | — | — | — |
| Ensemble Optimizado | — | — | — | — |

> **Nota:** Las métricas de SARIMA, Transferencia H&M y Ensemble se calculan al ejecutar `train_models.py` v2.0 con los datos reales. El modelo XGBoost supera el objetivo de mejora del 20–25% con un **+46.7%** de reducción en MAE.

### Modelo de Transferencia H&M

| Indicador | Valor |
|-----------|-------|
| Dataset de referencia | H&M Personalized Fashion (Kaggle) |
| Transacciones procesadas | 31,788,324 |
| Modelo SARIMA aplicado | SARIMA(1,1,1)(1,1,1)[7] sobre serie limpia |
| MAPE del modelo H&M | 15.48% |
| Outliers corregidos en H&M | 26 días (IQR ×2.5) |
| Pronóstico DTF a 30 días | 61 unidades (rango: 43–80) |

> El modelo híbrido SARIMA+XGBoost fue probado y **descartado** porque empeoró el MAPE de 16.27% a 23.43%. XGBoost amplificó outliers al sobreajustar al patrón de fin de semana. Los modelos ML se usan de forma independiente, NO como correctores de residuos SARIMA. Ver Reporte Técnico Final, Sección 2.5.

---

## Arquitectura

```
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌───────────────┐
│  Fuentes de Datos│────▶│  Módulo ETL v2.0  │────▶│  Entrenamiento ML   │────▶│  API FastAPI   │
│                  │     │  Python/Pandas    │     │                     │     │  v2.0          │
│ • Excel Ventas   │     │ • Ingesta/merge   │     │ • SARIMA real       │     │ /predict       │
│ • Índices H&M    │     │ • Limpieza        │     │ • Random Forest     │     │ /transfer      │
│ • Google Trends  │     │ • Feature Eng.    │     │ • XGBoost real      │     │ /trends/live   │
│                  │     │ • Outlier detect  │     │ • Transfer H&M      │     │ /seasonality   │
│                  │     │ • H&M calibration │     │ • Ensemble óptimo   │     │ /recommendations│
└──────────────────┘     └──────────────────┘     └─────────────────────┘     └───────┬───────┘
                                                                                      │
                                                                             ┌────────▼────────┐
                                                                             │   Dashboard     │
                                                                             │   Streamlit v2.0│
                                                                             │                 │
                                                                             │ • Pronósticos   │
                                                                             │ • Rankings      │
                                                                             │ • H&M Transfer  │
                                                                             │ • Google Trends │
                                                                             │ • Producción    │
                                                                             └─────────────────┘
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
│   ├── raw/                              # Datos originales (no en git)
│   │   └── DTF_s_DATA_CORRECT.xlsx
│   └── processed/                        # Datos limpios generados por ETL
│       ├── ventas_limpias.csv
│       ├── serie_semanal.csv
│       ├── serie_semanal_limpia.csv      # (v2.0) Outliers suavizados
│       ├── features_modelo.csv
│       ├── baseline_naive.csv
│       ├── resumen_eda.csv
│       ├── factores_correccion_hm.csv    # (v2.0) Factores H&M → DTF
│       ├── indices_estacionales_hm.json  # (v2.0) Índices H&M hardcodeados
│       └── pipeline_metadata.json
│
├── etl/
│   └── etl_pipeline.py                   # Pipeline ETL v2.0
│
├── models/
│   ├── train_models.py                   # Entrenamiento v2.0 (SARIMA + RF + XGBoost + H&M)
│   └── saved/                            # Modelos serializados (.pkl)
│       ├── random_forest_model.pkl
│       ├── xgboost_model.pkl             # (v2.0) Antes: gradient_boosting_model.pkl
│       ├── comparacion_modelos.csv
│       ├── feature_importance.csv
│       ├── predicciones_completas.csv
│       ├── sarima_diagnostico.json       # (v2.0) Diagnóstico SARIMA por categoría
│       └── training_report.json
│
├── api/
│   └── main.py                           # API REST FastAPI v2.0
│
├── dashboard/
│   └── app.py                            # Dashboard Streamlit v2.0
│
├── research/                             # Experimentos de investigación H&M
│   ├── README.md                         # Documentación de los experimentos
│   ├── hm_limpieza_agregacion.py         # Limpieza dataset H&M (3.5GB)
│   ├── hm_exploracion.py                 # Análisis exploratorio H&M
│   ├── hm_sarima.py                      # SARIMA base — iteración 1
│   ├── hm_xgboost.py                     # Híbrido SARIMA+XGBoost — descartado
│   ├── hm_sarima_mejorado.py             # SARIMA con limpieza outliers — final
│   ├── dtf_finetuning.py                 # Calibración de patrones H&M → DTF
│   └── demo_trends.py                    # Demo de integración pytrends
│
├── docs/
│   ├── DTF_Fashion_Reporte_Tecnico_Final.pdf
│   └── Predictive_Analysis_Platform_for_Fashion_Brands.pdf
│
└── notebooks/
    └── eda_exploratorio.ipynb
```

---

## Instalación

### Prerrequisitos

- Python 3.10+
- pip

### Setup

```bash
# Clonar repositorio
git clone https://github.com/FlowersLoop/dtf-predictive-platform.git
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
# 1. Ejecutar pipeline ETL (limpieza, transformación, calibración H&M)
python etl/etl_pipeline.py

# 2. Entrenar modelos (SARIMA, RF, XGBoost, Transfer H&M, Ensemble)
python models/train_models.py

# 3. Levantar API
cd api && uvicorn main:app --reload --port 8000

# 4. Levantar Dashboard
streamlit run dashboard/app.py

# 5. Abrir documentación interactiva de la API
# Navegar a http://localhost:8000/docs
```

---

## API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Health check y estado del sistema |
| `GET` | `/predict/{categoria}` | Pronóstico por categoría con bandas ±30% |
| `GET` | `/predict` | Pronóstico de todas las categorías |
| `GET` | `/transfer/{categoria}` | Pronóstico modelo transferencia H&M **(v2.0)** |
| `GET` | `/trends` | Ranking de categorías por demanda predicha |
| `GET` | `/trends/live` | Google Trends en tiempo real **(v2.0)** |
| `GET` | `/metrics` | Métricas de evaluación (5 modelos + ensemble) |
| `GET` | `/features` | Feature importance (incluye features H&M) |
| `GET` | `/recommendations` | Recomendaciones de producción con escenarios |
| `GET` | `/seasonality` | Índices estacionales H&M vs DTF **(v2.0)** |
| `GET` | `/history/{categoria}` | Historial de ventas semanales |
| `POST` | `/train` | Re-entrenar modelos con datos actualizados |

### Ejemplo de uso

```bash
# Obtener pronóstico de la categoría Gym (incluye bandas de incertidumbre)
curl http://localhost:8000/predict/Gym

# Obtener pronóstico con modelo de transferencia H&M
curl http://localhost:8000/transfer/Gym

# Consultar Google Trends para una categoría
curl "http://localhost:8000/trends/live?categoria=Futbol&timeframe=today%203-m"

# Comparar estacionalidad H&M vs DTF
curl http://localhost:8000/seasonality

# Obtener recomendaciones de producción con escenarios
curl http://localhost:8000/recommendations
```

---

## Modelos de Machine Learning

### 1. SARIMA(1,1,1)(1,1,1)[7]
Modelo de series temporales con componente estacional semanal (s=7). Parámetros validados por análisis ACF/PACF sobre el dataset H&M. Usa `statsmodels.SARIMAX` con fallback a Holt-Winters para categorías con pocas observaciones (<14 semanas).

### 2. Random Forest
Ensemble de árboles de decisión que expone la importancia relativa de cada feature. Robusto frente a outliers y datos faltantes.

### 3. XGBoost
Modelo de gradient boosting real (`xgboost.XGBRegressor`) para predicción tabular de alta precisión. Mejor rendimiento individual con +46.7% de mejora sobre el baseline.

### 4. Transferencia H&M (v2.0)
Modelo puramente estadístico que no usa ML. Extrae índices estacionales normalizados del dataset H&M (31.7M transacciones), los escala al volumen DTF, y los corrige mes a mes con datos reales de la tienda. Su valor está en capturar patrones macro de la industria de moda que los modelos ML no pueden aprender con solo 54 transacciones locales.

### 5. Ensemble Optimizado (v2.0)
Combinación ponderada de los 4 modelos con pesos optimizados por grid search sobre MAE en validación. Los pesos ya no son fijos — se recalculan en cada entrenamiento.

### Iteración descartada: SARIMA + XGBoost híbrido
Se probó usar XGBoost como corrector de residuos del SARIMA (13 features de calendario y rezagos). El MAPE empeoró de 16.27% a 23.43% porque XGBoost sobreajustó al patrón de fin de semana (`es_finde` importancia: 0.20) y amplificó un día atípico. Se descartó el enfoque híbrido. Ver `research/hm_xgboost.py`.

### Features predictivos principales

| Feature | Tipo | Interpretación |
|---------|------|----------------|
| `cambio_semanal` | Lag | Aceleración de demanda — el mejor predictor |
| `lag_1w` | Lag | Ventas de la semana pasada |
| `ventas_acumuladas` | Acumulado | Popularidad total de la categoría |
| `rolling_mean_2w` | Rolling | Promedio móvil de 2 semanas |
| `hm_idx_mensual` | H&M Transfer | Índice estacional mensual H&M |
| `hm_pred_escalada` | H&M Transfer | Predicción H&M calibrada a escala DTF |
| `ratio_vs_hm_lag1` | H&M Transfer | Desviación DTF vs H&M la semana pasada |
| `trends_score` | Google Trends | Interés de búsqueda (placeholder) |

---

## Datos

El dataset actual contiene **54 transacciones** de una marca DTF operando desde octubre 2025, con las siguientes dimensiones:

- **13 categorías**: Sports, Gym, Fútbol, Basketball, Tenis, Casual, Movies, Música, etc.
- **4 tipos de prenda**: T-Shirt (69%), Jacket (13%), Hoodie (9%), Long Sleeve (9%)
- **14 estados** de México + 1 internacional (Canadá)
- **37 diseños únicos**

### Métricas financieras del período

| Indicador | Valor |
|-----------|-------|
| Período de análisis | Octubre 2025 – Marzo 2026 |
| Ingresos brutos totales | $24,092.00 MXN |
| Ingreso neto total | $19,623.48 MXN |
| Margen neto promedio | 81.5% |
| Ticket promedio | $446.15 MXN |
| Categoría líder por volumen | Sports — 14 unidades (25.9%) |
| Categoría líder por ingreso | Gym — $5,846 MXN |
| Concentración geográfica | CDMX — 47% de las ventas |

### Hallazgos clave

- **Rotación mensual de categorías dominantes**: Fútbol en diciembre, Gym en enero-febrero, Sports en marzo.
- **Bursts de demanda**: Track & Field representó 14 de 16 ventas en marzo, todas desde CDMX.
- **Divergencia estacional H&M vs DTF**: DTF muestra pico los domingos (compradores online), H&M los sábados (tienda física). Febrero y marzo en DTF superan la predicción H&M en 53% y 75% respectivamente.

---

## Transferencia de patrones H&M

### ¿Por qué se usa el dataset H&M?

Con solo 54 transacciones, entrenar SARIMA directamente sobre los datos DTF genera overfitting severo. Se adoptó una estrategia de **transferencia de conocimiento estadístico** usando el dataset público H&M (Kaggle, 2022) como prior:

1. Se procesaron 31.7M de transacciones para extraer índices estacionales normalizados (semanal y mensual).
2. Se limpió la serie con IQR ×2.5 (26 outliers corregidos), mejorando el MAPE de 16.27% a 15.48%.
3. Los índices se escalaron al volumen DTF con un factor de escala de 0.0000346.
4. Se calcularon factores de corrección mensual para ajustar divergencias culturales (mercado europeo vs mexicano).

Los scripts de este proceso están en la carpeta `research/` y el reporte técnico detallado en `docs/`.

---

## Deploy

### Streamlit Cloud (gratis)

1. Sube el repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio → selecciona `dashboard/app.py`
4. Se publica automáticamente como `tu-app.streamlit.app`

### Otros servicios

- **Railway / Render**: Agregar `Procfile` con `web: streamlit run dashboard/app.py --server.port $PORT`
- **Docker**: Containerización pendiente como trabajo futuro
- **Dominio propio**: Apuntar un dominio a cualquiera de estos servicios

---

## Metodología

El desarrollo sigue la metodología **APQP** (Advanced Product Quality Planning) adaptada a software:

| Fase | Periodo | Descripción |
|------|---------|-------------|
| 1. Planeación | Ene 12 – Mar 14 | Definición del problema, requerimientos, stack |
| 2. Diseño | Mar 15 – Mar 29 | Arquitectura, pipeline ETL, diseño de API |
| 3. Implementación | Mar 28 – Abr 21 | ETL, modelos ML, API, dashboard, transfer H&M |
| 4. Validación | Abr 20 – Abr 27 | Métricas, pruebas end-to-end, comparación modelos |
| 5. Cierre | Abr 28 – May 4 | Documentación, presentación oral |

---

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.10+ |
| ETL | pandas, numpy |
| ML | scikit-learn, XGBoost, statsmodels (SARIMAX) |
| API | FastAPI, uvicorn, pydantic |
| Dashboard | Streamlit, Plotly |
| Señales externas | pytrends (Google Trends) |
| Dataset de referencia | H&M Kaggle (31.7M transacciones) |
| Serialización | pickle, JSON |
| Versionado | Git / GitHub |

---

## Métricas de evaluación

- **MAE** (Mean Absolute Error): error promedio en unidades. Menor = mejor.
- **MAPE** (Mean Absolute Percentage Error): error porcentual. Menor = mejor.
- **R²** (Coeficiente de determinación): varianza explicada. Mayor = mejor (máximo 1.0).

El objetivo del proyecto es una **mejora mínima del 20–25%** en MAE respecto al baseline naive. El resultado actual del mejor modelo individual es **+46.7%**.

---

## Trabajo futuro

- Integración completa de **Google Trends API** como feature activo en el pipeline ETL (actualmente es placeholder).
- **Sistema de recomendación** tipo collaborative filtering ("clientes que compraron X también compraron Y").
- **Alertas automáticas** cuando el modelo detecta un burst inminente de demanda.
- Migración a **PostgreSQL** cuando el volumen de datos lo justifique.
- **MLflow** para versionado y tracking de experimentos de modelos.
- Containerización con **Docker** para despliegue en nube.
- **A/B testing** de pesos del ensemble con datos de producción real.

---

## Autores

Santiago Tapia & Fernando Flores

## Licencia

Uso académico. Todos los derechos reservados.