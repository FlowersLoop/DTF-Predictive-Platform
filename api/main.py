"""
============================================================================
API Backend — Plataforma Predictiva DTF
============================================================================
FastAPI REST API que expone los modelos de forecasting para consumo
del dashboard y sistemas externos.

Endpoints:
  GET  /                        → Health check
  GET  /predict/{categoria}     → Pronóstico por categoría
  GET  /predict                 → Pronóstico de todas las categorías
  GET  /trends                  → Ranking de categorías por demanda predicha
  GET  /metrics                 → Métricas de los modelos
  GET  /features                → Feature importance
  GET  /recommendations         → Recomendaciones de producción DTF
  GET  /history/{categoria}     → Historial de ventas de una categoría
  POST /train                   → Re-entrenar modelos con datos actualizados

Ejecución:
  uvicorn main:app --reload --port 8000

============================================================================
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import json
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "saved"
ETL_SCRIPT = BASE_DIR / "etl" / "etl_pipeline.py"
TRAIN_SCRIPT = BASE_DIR / "models" / "train_models.py"

# ============================================================================
# APP FASTAPI
# ============================================================================
app = FastAPI(
    title="DTF Predictive Platform API",
    description=(
        "API de análisis predictivo para marcas de moda con producción DTF. "
        "Expone pronósticos de demanda por categoría, rankings de tendencias, "
        "recomendaciones de producción y métricas de modelos ML."
    ),
    version="1.0.0",
    contact={"name": "DTF Predictive Team"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# CARGA DE DATOS Y MODELOS
# ============================================================================
def load_model(name: str):
    """Carga un modelo serializado desde disco."""
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(name: str) -> Optional[pd.DataFrame]:
    """Carga un CSV de datos procesados."""
    path = DATA_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_json(name: str) -> Optional[dict]:
    """Carga un JSON de metadata o reportes."""
    for directory in [MODEL_DIR, DATA_DIR]:
        path = directory / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


# ============================================================================
# SCHEMAS PYDANTIC
# ============================================================================
class PredictionResponse(BaseModel):
    categoria: str
    semana: str
    prediccion_unidades: float
    confianza: float
    modelo_principal: str
    pred_random_forest: float
    pred_gradient_boosting: float
    pred_seasonal: float
    recomendacion: str


class TrendItem(BaseModel):
    categoria: str
    demanda_predicha: float
    tendencia: str  # "subiendo", "estable", "bajando"
    confianza_media: float
    recomendacion: str


class MetricsResponse(BaseModel):
    modelo: str
    MAE: float
    MAPE: float
    R2: float
    mejora_vs_baseline: str


class TrainResponse(BaseModel):
    status: str
    mensaje: str
    timestamp: str
    metricas: Optional[dict] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["health"])
def health_check():
    """Health check — verifica que la API está corriendo y los datos están cargados."""
    predictions = load_csv("predicciones_completas.csv")
    report = load_json("training_report.json")

    return {
        "status": "online",
        "proyecto": "Plataforma Predictiva DTF",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "datos_cargados": predictions is not None,
        "modelos_disponibles": report is not None,
        "endpoints": [
            "GET /predict/{categoria}",
            "GET /predict",
            "GET /trends",
            "GET /metrics",
            "GET /features",
            "GET /recommendations",
            "GET /history/{categoria}",
            "POST /train",
        ],
    }


@app.get("/predict/{categoria}", response_model=list[PredictionResponse], tags=["predicciones"])
def predict_categoria(categoria: str):
    """
    Pronóstico de demanda para una categoría específica.
    Devuelve predicciones semanales con intervalos de confianza.
    """
    df = load_csv("predicciones_completas.csv")
    if df is None:
        raise HTTPException(status_code=503, detail="Predicciones no disponibles. Ejecutar POST /train primero.")

    categorias_disponibles = df["categoria"].unique().tolist()
    cat_match = None
    for c in categorias_disponibles:
        if c.lower() == categoria.lower():
            cat_match = c
            break

    if cat_match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Categoría '{categoria}' no encontrada. Disponibles: {categorias_disponibles}"
        )

    cat_df = df[df["categoria"] == cat_match].sort_values("semana_inicio")

    results = []
    for _, row in cat_df.iterrows():
        results.append(PredictionResponse(
            categoria=cat_match,
            semana=str(row["semana_inicio"]),
            prediccion_unidades=round(float(row["pred_ensemble"]), 2),
            confianza=round(float(row.get("confianza", 50)), 1),
            modelo_principal="Gradient Boosting (XGBoost)",
            pred_random_forest=round(float(row["pred_rf"]), 2),
            pred_gradient_boosting=round(float(row["pred_gb"]), 2),
            pred_seasonal=round(float(row["sarima_pred"]), 2),
            recomendacion=str(row.get("recomendacion", "SIN_DATO")),
        ))

    return results


@app.get("/predict", response_model=list[PredictionResponse], tags=["predicciones"])
def predict_all(
    top: int = Query(default=50, ge=1, le=200, description="Número máximo de resultados"),
):
    """
    Pronóstico de demanda para todas las categorías.
    Ordenado por predicción de demanda descendente.
    """
    df = load_csv("predicciones_completas.csv")
    if df is None:
        raise HTTPException(status_code=503, detail="Predicciones no disponibles.")

    df = df.sort_values("pred_ensemble", ascending=False).head(top)

    results = []
    for _, row in df.iterrows():
        results.append(PredictionResponse(
            categoria=str(row["categoria"]),
            semana=str(row["semana_inicio"]),
            prediccion_unidades=round(float(row["pred_ensemble"]), 2),
            confianza=round(float(row.get("confianza", 50)), 1),
            modelo_principal="Gradient Boosting (XGBoost)",
            pred_random_forest=round(float(row["pred_rf"]), 2),
            pred_gradient_boosting=round(float(row["pred_gb"]), 2),
            pred_seasonal=round(float(row["sarima_pred"]), 2),
            recomendacion=str(row.get("recomendacion", "SIN_DATO")),
        ))

    return results


@app.get("/trends", response_model=list[TrendItem], tags=["tendencias"])
def get_trends():
    """
    Ranking de categorías por demanda predicha.
    Incluye dirección de tendencia (subiendo/bajando/estable).
    """
    pred_df = load_csv("predicciones_completas.csv")
    serie_df = load_csv("serie_semanal.csv")

    if pred_df is None:
        raise HTTPException(status_code=503, detail="Predicciones no disponibles.")

    ranking = pred_df.groupby("categoria").agg(
        demanda_total=("pred_ensemble", "sum"),
        confianza_media=("confianza", "mean"),
    ).sort_values("demanda_total", ascending=False)

    results = []
    for cat, row in ranking.iterrows():
        # Calcular tendencia comparando últimas semanas del historial
        tendencia = "estable"
        if serie_df is not None:
            cat_hist = serie_df[serie_df["categoria"] == cat].sort_values("semana_inicio")
            if len(cat_hist) >= 4:
                recent = cat_hist["unidades"].tail(4).values
                first_half = recent[:2].mean()
                second_half = recent[2:].mean()
                if second_half > first_half * 1.2:
                    tendencia = "subiendo"
                elif second_half < first_half * 0.8:
                    tendencia = "bajando"

        # Recomendación basada en demanda predicha
        d = row["demanda_total"]
        if d >= 4:
            reco = "PRODUCIR_ALTO — Preparar stock inmediato"
        elif d >= 2:
            reco = "PRODUCIR_MEDIO — Lote moderado"
        elif d >= 0.5:
            reco = "PRODUCIR_BAJO — Impresión bajo demanda"
        else:
            reco = "MONITOREAR — Sin señal fuerte"

        results.append(TrendItem(
            categoria=str(cat),
            demanda_predicha=round(float(d), 2),
            tendencia=tendencia,
            confianza_media=round(float(row["confianza_media"]), 1),
            recomendacion=reco,
        ))

    return results


@app.get("/metrics", response_model=list[MetricsResponse], tags=["modelos"])
def get_metrics():
    """
    Métricas de evaluación de todos los modelos entrenados.
    Incluye MAE, MAPE, R² y mejora porcentual vs baseline.
    """
    df = load_csv("comparacion_modelos.csv")
    if df is None:
        raise HTTPException(status_code=503, detail="Métricas no disponibles.")

    results = []
    for _, row in df.iterrows():
        mejora = row.get("mejora_mae_pct", 0)
        results.append(MetricsResponse(
            modelo=str(row["modelo"]),
            MAE=round(float(row["MAE"]), 4),
            MAPE=round(float(row["MAPE"]), 2),
            R2=round(float(row["R2"]), 4),
            mejora_vs_baseline=f"{mejora:+.1f}%" if mejora != 0 else "baseline",
        ))

    return results


@app.get("/features", tags=["modelos"])
def get_feature_importance(top: int = Query(default=15, ge=1, le=30)):
    """
    Feature importance combinada de Random Forest y Gradient Boosting.
    Muestra qué variables son más predictivas para la demanda.
    """
    df = load_csv("feature_importance.csv")
    if df is None:
        raise HTTPException(status_code=503, detail="Feature importance no disponible.")

    df = df.sort_values("importance_avg", ascending=False).head(top)

    return {
        "top_features": [
            {
                "feature": str(row["feature"]),
                "importance_random_forest": round(float(row["importance_rf"]), 4),
                "importance_gradient_boosting": round(float(row["importance_gb"]), 4),
                "importance_promedio": round(float(row["importance_avg"]), 4),
            }
            for _, row in df.iterrows()
        ],
        "interpretacion": {
            "cambio_semanal": "Aceleración de demanda — el predictor más fuerte",
            "lag_1w": "Ventas de la semana pasada — comportamiento reciente",
            "ventas_acumuladas": "Popularidad acumulada de la categoría",
            "rolling_mean_2w": "Promedio móvil de 2 semanas — tendencia a corto plazo",
        },
    }


@app.get("/recommendations", tags=["producción"])
def get_recommendations():
    """
    Recomendaciones de producción DTF basadas en las predicciones.
    Responde: ¿qué imprimir, cuánto y cuándo?
    """
    pred_df = load_csv("predicciones_completas.csv")
    if pred_df is None:
        raise HTTPException(status_code=503, detail="Predicciones no disponibles.")

    # Agrupar por categoría
    summary = pred_df.groupby("categoria").agg(
        demanda_total=("pred_ensemble", "sum"),
        demanda_maxima_semana=("pred_ensemble", "max"),
        confianza_media=("confianza", "mean"),
        semana_pico=("pred_ensemble", "idxmax"),
    ).sort_values("demanda_total", ascending=False)

    recommendations = []
    for cat, row in summary.iterrows():
        d = row["demanda_total"]
        pico_idx = int(row["semana_pico"])
        semana_pico = pred_df.loc[pico_idx, "semana_inicio"] if pico_idx in pred_df.index else "N/A"

        if d >= 4:
            prioridad = "ALTA"
            accion = f"Producir {int(np.ceil(d))}+ unidades. Imprimir antes de {semana_pico}."
            urgencia = "INMEDIATA"
        elif d >= 2:
            prioridad = "MEDIA"
            accion = f"Preparar {int(np.ceil(d))} unidades. Monitorear tendencia."
            urgencia = "ESTA SEMANA"
        elif d >= 0.5:
            prioridad = "BAJA"
            accion = "Impresión bajo demanda. No acumular stock."
            urgencia = "CUANDO SE PIDA"
        else:
            prioridad = "NINGUNA"
            accion = "No producir. Sin señal de demanda."
            urgencia = "N/A"

        recommendations.append({
            "categoria": str(cat),
            "prioridad": prioridad,
            "urgencia": urgencia,
            "accion": accion,
            "demanda_predicha_unidades": round(float(d), 1),
            "confianza": round(float(row["confianza_media"]), 1),
            "semana_pico_demanda": str(semana_pico),
        })

    return {
        "fecha_generacion": datetime.now().isoformat(),
        "horizonte": "Próximas 4 semanas",
        "recomendaciones": recommendations,
        "resumen": {
            "categorias_alta_prioridad": sum(1 for r in recommendations if r["prioridad"] == "ALTA"),
            "categorias_media_prioridad": sum(1 for r in recommendations if r["prioridad"] == "MEDIA"),
            "unidades_totales_sugeridas": round(sum(r["demanda_predicha_unidades"] for r in recommendations), 1),
        },
    }


@app.get("/history/{categoria}", tags=["datos"])
def get_history(
    categoria: str,
    semanas: int = Query(default=24, ge=1, le=52, description="Número de semanas de historial"),
):
    """
    Historial de ventas semanales de una categoría.
    Útil para graficar tendencia histórica en el dashboard.
    """
    serie = load_csv("serie_semanal.csv")
    if serie is None:
        raise HTTPException(status_code=503, detail="Datos históricos no disponibles.")

    cats = serie["categoria"].unique().tolist()
    cat_match = None
    for c in cats:
        if c.lower() == categoria.lower():
            cat_match = c
            break

    if cat_match is None:
        raise HTTPException(status_code=404, detail=f"Categoría no encontrada. Disponibles: {cats}")

    cat_data = (
        serie[serie["categoria"] == cat_match]
        .sort_values("semana_inicio")
        .tail(semanas)
    )

    return {
        "categoria": cat_match,
        "semanas_solicitadas": semanas,
        "datos": [
            {
                "semana": str(row["semana_inicio"]),
                "unidades": int(row["unidades"]),
                "ingresos": round(float(row["ingresos_total"]), 2),
            }
            for _, row in cat_data.iterrows()
        ],
        "estadisticas": {
            "total_unidades": int(cat_data["unidades"].sum()),
            "promedio_semanal": round(float(cat_data["unidades"].mean()), 2),
            "semana_pico": str(cat_data.loc[cat_data["unidades"].idxmax(), "semana_inicio"]),
            "max_unidades_semana": int(cat_data["unidades"].max()),
        },
    }


@app.post("/train", response_model=TrainResponse, tags=["entrenamiento"])
def retrain_models():
    """
    Re-entrena los modelos con los datos más recientes.
    Ejecuta el pipeline ETL + entrenamiento de modelos.
    """
    try:
        # Paso 1: ETL
        etl_result = subprocess.run(
            ["python3", str(ETL_SCRIPT)],
            capture_output=True, text=True, timeout=120,
        )
        if etl_result.returncode != 0:
            return TrainResponse(
                status="error",
                mensaje=f"Error en ETL: {etl_result.stderr[:500]}",
                timestamp=datetime.now().isoformat(),
            )

        # Paso 2: Entrenamiento
        train_result = subprocess.run(
            ["python3", str(TRAIN_SCRIPT)],
            capture_output=True, text=True, timeout=300,
        )
        if train_result.returncode != 0:
            return TrainResponse(
                status="error",
                mensaje=f"Error en entrenamiento: {train_result.stderr[:500]}",
                timestamp=datetime.now().isoformat(),
            )

        # Cargar métricas actualizadas
        report = load_json("training_report.json")
        metricas = report.get("modelos", {}) if report else None

        return TrainResponse(
            status="success",
            mensaje="Modelos re-entrenados exitosamente.",
            timestamp=datetime.now().isoformat(),
            metricas=metricas,
        )

    except subprocess.TimeoutExpired:
        return TrainResponse(
            status="error",
            mensaje="Timeout: el entrenamiento tardó más de 5 minutos.",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ============================================================================
# MANEJO DE ERRORES
# ============================================================================
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc.detail), "endpoints_disponibles": "/docs"},
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor", "contacto": "Revisar logs"},
    )


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)