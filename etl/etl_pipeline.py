"""
Limpieza, transformación y feature engineering
de datos históricos de ventas para modelos de forecasting.

Entrada:  DTF_s_DATA_CORRECT.xlsx (datos crudos de ventas)
Salidas:  
  - ventas_limpias.csv          → Datos limpios por transacción
  - serie_semanal.csv           → Serie de tiempo semanal por categoría
  - features_modelo.csv         → Dataset listo para entrenar ML
  - resumen_eda.csv             → Resumen exploratorio por categoría
  - baseline_naive.csv          → Predicciones del baseline naive
"""

import pandas as pd
import numpy as np
import re
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
INPUT_FILE = "/mnt/user-data/uploads/DTF_s_DATA_CORRECT.xlsx"
OUTPUT_DIR = Path("/home/claude/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PROJECT_START = pd.Timestamp("2025-10-01")
PROJECT_END = pd.Timestamp("2026-03-16")

# Festividades y eventos relevantes para México + moda DTF
HOLIDAYS_MX = {
    "2025-10-31": "halloween",
    "2025-11-01": "dia_muertos",
    "2025-11-02": "dia_muertos",
    "2025-11-15": "buen_fin",
    "2025-11-16": "buen_fin",
    "2025-11-17": "buen_fin",
    "2025-11-28": "black_friday",
    "2025-12-12": "virgen_guadalupe",
    "2025-12-24": "nochebuena",
    "2025-12-25": "navidad",
    "2025-12-31": "fin_de_ano",
    "2026-01-01": "ano_nuevo",
    "2026-02-14": "san_valentin",
    "2026-03-08": "dia_mujer",
}

# Mapeo de normalización de estados
ESTADO_NORMALIZE = {
    "distrito federal": "CDMX",
    "cdmx": "CDMX",
    "ciudad de méxico": "CDMX",
    "ciudad de mexico": "CDMX",
    "estado de méxico": "Estado de México",
    "estado de mexico": "Estado de México",
    "baja california": "Baja California",
    "nuevo león": "Nuevo León",
    "nuevo leon": "Nuevo León",
    "jalisco": "Jalisco",
    "puebla": "Puebla",
    "querétaro": "Querétaro",
    "queretaro": "Querétaro",
    "michoacán": "Michoacán",
    "michoacan": "Michoacán",
    "veracruz": "Veracruz",
    "morelos": "Morelos",
    "colima": "Colima",
    "aguascalientes": "Aguascalientes",
    "saltillo": "Coahuila",
    "monterrey": "Nuevo León",
    "tijuana": "Baja California",
    "toronto": "Ontario",
}

CATEGORIA_NORMALIZE = {
    "fútbol": "Futbol",
    "futbol": "Futbol",
    "tenis": "Tenis",
    "musica": "Musica",
    "deportiva": "Deportiva",
    "movies": "Movies",
    "sports": "Sports",
}

# ============================================================================
# PASO 1: CARGA Y PARSEO DE FECHAS
# ============================================================================
print("=" * 70)
print("PASO 1: Carga de datos y parseo de fechas")
print("=" * 70)

df_raw = pd.read_excel(INPUT_FILE, header=None)

# Buscar fila de encabezados
header_row = None
for i, row in df_raw.iterrows():
    if any(str(v).strip() == "# de venta" for v in row.values if v is not None):
        header_row = i
        break

df = pd.read_excel(INPUT_FILE, header=header_row)
df = df.dropna(subset=["# de venta"])
df.columns = df.columns.str.strip()

print(f"  Registros cargados: {len(df)}")
print(f"  Columnas: {list(df.columns)}")

# Parseo robusto de fechas en español
MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
}

def parse_fecha_es(fecha_str):
    """Parsea fechas en formato '07 de octubre 2025' o '9 de marzo de 2026'."""
    if pd.isna(fecha_str):
        return pd.NaT
    s = str(fecha_str).lower().strip()
    s = s.replace(" de ", " ").replace("  ", " ")
    parts = s.split()
    if len(parts) < 3:
        return pd.NaT
    try:
        dia = int(parts[0])
        mes = MESES_ES.get(parts[1])
        anio = int(parts[2])
        if mes is None:
            return pd.NaT
        return pd.Timestamp(anio, mes, dia)
    except (ValueError, IndexError):
        return pd.NaT

df["fecha"] = df["Fecha de venta"].apply(parse_fecha_es)

fechas_fallidas = df["fecha"].isna().sum()
if fechas_fallidas > 0:
    print(f"  ⚠ Fechas no parseadas: {fechas_fallidas}")
    print(df[df["fecha"].isna()][["# de venta", "Fecha de venta"]])
else:
    print(f"  ✓ Todas las fechas parseadas correctamente")

print(f"  Rango: {df['fecha'].min().strftime('%Y-%m-%d')} → {df['fecha'].max().strftime('%Y-%m-%d')}")


# ============================================================================
# PASO 2: LIMPIEZA Y NORMALIZACIÓN
# ============================================================================
print("\n" + "=" * 70)
print("PASO 2: Limpieza y normalización de campos")
print("=" * 70)

# Renombrar columnas a snake_case
df = df.rename(columns={
    "# de venta": "venta_id",
    "Estado de venta": "estado_venta",
    "Ingresos": "ingresos",
    "Cargo por venta": "cargo_venta",
    "Costo de envio": "costo_envio",
    "Total": "total_neto",
    "Diseño DTF": "diseno",
    "Estado": "estado_geo",
    "País": "pais",
    "Categoria": "categoria",
    "Tipo de prenda": "tipo_prenda",
})

# Limpiar ID de venta
df["venta_id"] = pd.to_numeric(df["venta_id"], errors="coerce").astype("Int64")

# Limpiar campos numéricos
for col in ["ingresos", "cargo_venta", "costo_envio", "total_neto"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round(2)

# Recalcular total_neto para verificar integridad
df["total_calculado"] = (df["ingresos"] - df["cargo_venta"] - df["costo_envio"]).round(2)
discrepancias = (df["total_neto"] - df["total_calculado"]).abs() > 1
if discrepancias.any():
    n_disc = discrepancias.sum()
    print(f"  ⚠ {n_disc} discrepancias en total_neto, recalculando...")
    df.loc[discrepancias, "total_neto"] = df.loc[discrepancias, "total_calculado"]
else:
    print("  ✓ Todos los totales verificados")

df = df.drop(columns=["total_calculado"])

# Normalizar strings
df["diseno"] = df["diseno"].str.strip()
df["tipo_prenda"] = df["tipo_prenda"].str.strip().str.title()
df["pais"] = df["pais"].str.strip().str.title()

# Normalizar estados geográficos
df["estado_geo_raw"] = df["estado_geo"].str.strip()
df["estado_geo"] = df["estado_geo_raw"].str.lower().str.strip().map(
    lambda x: ESTADO_NORMALIZE.get(x, x.title() if isinstance(x, str) else x)
)

# Normalizar categorías
df["categoria_raw"] = df["categoria"].str.strip()
df["categoria"] = df["categoria_raw"].str.lower().str.strip().map(
    lambda x: CATEGORIA_NORMALIZE.get(x, x.title() if isinstance(x, str) else x)
)

# Normalizar país
df["pais"] = df["pais"].replace({"Cánada": "Canadá", "Canada": "Canadá", "México": "México"})

# Marcar si es venta internacional
df["es_internacional"] = (df["pais"] != "México").astype(int)

print(f"  ✓ {df['categoria'].nunique()} categorías: {sorted(df['categoria'].unique())}")
print(f"  ✓ {df['tipo_prenda'].nunique()} tipos: {sorted(df['tipo_prenda'].unique())}")
print(f"  ✓ {df['estado_geo'].nunique()} estados: {sorted(df['estado_geo'].unique())}")
print(f"  ✓ {df['diseno'].nunique()} diseños únicos")


# ============================================================================
# PASO 3: FEATURE ENGINEERING — VARIABLES TEMPORALES
# ============================================================================
print("\n" + "=" * 70)
print("PASO 3: Feature engineering — variables temporales")
print("=" * 70)

df["anio"] = df["fecha"].dt.year
df["mes"] = df["fecha"].dt.month
df["semana_iso"] = df["fecha"].dt.isocalendar().week.astype(int)
df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=lunes, 6=domingo
df["dia_mes"] = df["fecha"].dt.day
df["semana_del_anio"] = df["fecha"].dt.isocalendar().week.astype(int)

# Etiqueta año-semana para agrupación temporal
df["anio_semana"] = df["fecha"].dt.strftime("%G-W%V")

# Flags de calendario (relevantes para moda en México)
df["es_quincena"] = df["dia_mes"].isin([14, 15, 16, 29, 30, 31]).astype(int)
df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
df["es_inicio_mes"] = (df["dia_mes"] <= 5).astype(int)

# Temporadas de moda/consumo
def get_temporada(fecha):
    m = fecha.month
    if m in [12, 1, 2]: return "invierno"
    if m in [3, 4, 5]: return "primavera"
    if m in [6, 7, 8]: return "verano"
    return "otono"

df["temporada"] = df["fecha"].apply(get_temporada)

# Flags de eventos específicos
def get_evento_cercano(fecha, ventana_dias=7):
    """Verifica si hay un evento relevante dentro de la ventana."""
    for fecha_ev, nombre_ev in HOLIDAYS_MX.items():
        ev = pd.Timestamp(fecha_ev)
        if abs((fecha - ev).days) <= ventana_dias:
            return nombre_ev
    return "ninguno"

df["evento_cercano"] = df["fecha"].apply(get_evento_cercano)
df["tiene_evento"] = (df["evento_cercano"] != "ninguno").astype(int)

# Semanas desde el inicio del negocio (para capturar tendencia de crecimiento)
df["semanas_desde_inicio"] = ((df["fecha"] - df["fecha"].min()).dt.days / 7).astype(int)

# Flag propósitos de año nuevo (Ene 1 - Feb 15)
df["es_resolucion_ano_nuevo"] = (
    ((df["mes"] == 1)) | ((df["mes"] == 2) & (df["dia_mes"] <= 15))
).astype(int)

print(f"  ✓ Variables temporales creadas: {len([c for c in df.columns if c not in ['Fecha de venta']])}")
print(f"  ✓ Eventos detectados: {df[df['tiene_evento'] == 1].shape[0]} ventas cerca de eventos")
print(f"  ✓ Ventas en quincena: {df['es_quincena'].sum()}")


# ============================================================================
# PASO 4: EXPORTAR DATOS LIMPIOS POR TRANSACCIÓN
# ============================================================================
print("\n" + "=" * 70)
print("PASO 4: Exportar datos limpios por transacción")
print("=" * 70)

cols_export = [
    "venta_id", "fecha", "anio_semana", "estado_venta",
    "ingresos", "cargo_venta", "costo_envio", "total_neto",
    "diseno", "categoria", "tipo_prenda",
    "estado_geo", "pais", "es_internacional",
    "anio", "mes", "semana_iso", "dia_semana", "dia_mes",
    "es_quincena", "es_fin_de_semana", "es_inicio_mes",
    "temporada", "evento_cercano", "tiene_evento",
    "semanas_desde_inicio", "es_resolucion_ano_nuevo",
]

df_clean = df[cols_export].sort_values("fecha").reset_index(drop=True)
df_clean.to_csv(OUTPUT_DIR / "ventas_limpias.csv", index=False)
print(f"  ✓ ventas_limpias.csv — {len(df_clean)} registros, {len(cols_export)} columnas")


# ============================================================================
# PASO 5: SERIE DE TIEMPO SEMANAL POR CATEGORÍA
# ============================================================================
print("\n" + "=" * 70)
print("PASO 5: Construir serie de tiempo semanal por categoría")
print("=" * 70)

# Crear rango completo de semanas
all_weeks = pd.date_range(
    start=df["fecha"].min() - timedelta(days=df["fecha"].min().weekday()),
    end=PROJECT_END,
    freq="W-MON"
)

# Categorías principales (las que tienen suficientes datos para modelar)
top_categorias = df["categoria"].value_counts()
print(f"  Distribución de categorías:")
for cat, count in top_categorias.items():
    print(f"    {cat}: {count} ventas")

categorias_modelo = top_categorias[top_categorias >= 2].index.tolist()

# Agregar ventas por semana y categoría
df["semana_inicio"] = df["fecha"] - pd.to_timedelta(df["fecha"].dt.weekday, unit="D")
df["semana_inicio"] = df["semana_inicio"].dt.normalize()

weekly_sales = df.groupby(["semana_inicio", "categoria"]).agg(
    unidades=("venta_id", "count"),
    ingresos_total=("ingresos", "sum"),
    total_neto_sum=("total_neto", "sum"),
    ticket_promedio=("ingresos", "mean"),
    n_disenos_unicos=("diseno", "nunique"),
    n_estados=("estado_geo", "nunique"),
).reset_index()

# Crear grid completo (todas las semanas × todas las categorías)
week_cat_grid = pd.MultiIndex.from_product(
    [all_weeks, categorias_modelo],
    names=["semana_inicio", "categoria"]
).to_frame(index=False)

# Normalizar las fechas de semana para merge
week_cat_grid["semana_inicio"] = pd.to_datetime(week_cat_grid["semana_inicio"]).dt.normalize()
weekly_sales["semana_inicio"] = pd.to_datetime(weekly_sales["semana_inicio"]).dt.normalize()

serie = week_cat_grid.merge(weekly_sales, on=["semana_inicio", "categoria"], how="left")

# Llenar semanas vacías con ceros (no hay ventas = 0 unidades)
fill_cols = ["unidades", "ingresos_total", "total_neto_sum", "ticket_promedio",
             "n_disenos_unicos", "n_estados"]
serie[fill_cols] = serie[fill_cols].fillna(0)

# Agregar features temporales a la serie semanal
serie["anio"] = serie["semana_inicio"].dt.year
serie["mes"] = serie["semana_inicio"].dt.month
serie["semana_iso"] = serie["semana_inicio"].dt.isocalendar().week.astype(int)
serie["anio_semana"] = serie["semana_inicio"].dt.strftime("%G-W%V")

# Temporada
serie["temporada"] = serie["semana_inicio"].apply(get_temporada)

# Eventos en la semana
def semana_tiene_evento(fecha_inicio):
    for i in range(7):
        d = fecha_inicio + timedelta(days=i)
        if d.strftime("%Y-%m-%d") in HOLIDAYS_MX:
            return 1
    return 0

serie["tiene_evento_semana"] = serie["semana_inicio"].apply(semana_tiene_evento)

# Propósitos de año nuevo
serie["es_resolucion_ano_nuevo"] = (
    ((serie["mes"] == 1)) | ((serie["mes"] == 2) & (serie["semana_inicio"].dt.day <= 15))
).astype(int)

# Semanas desde inicio del negocio
inicio_negocio = df["fecha"].min()
serie["semanas_desde_inicio"] = ((serie["semana_inicio"] - inicio_negocio).dt.days / 7).astype(int)

serie = serie.sort_values(["categoria", "semana_inicio"]).reset_index(drop=True)
serie.to_csv(OUTPUT_DIR / "serie_semanal.csv", index=False)

n_weeks = serie["semana_inicio"].nunique()
n_cats = serie["categoria"].nunique()
print(f"  ✓ serie_semanal.csv — {len(serie)} filas ({n_weeks} semanas × {n_cats} categorías)")
print(f"  ✓ Semanas cubiertas: {all_weeks[0].strftime('%Y-%m-%d')} → {all_weeks[-1].strftime('%Y-%m-%d')}")


# ============================================================================
# PASO 6: FEATURE ENGINEERING PARA MODELOS ML
# ============================================================================
print("\n" + "=" * 70)
print("PASO 6: Feature engineering para modelos ML (lag features + rolling)")
print("=" * 70)

features = serie.copy()

# LAG FEATURES (ventas de semanas anteriores como predictores)
for lag in [1, 2, 3, 4]:
    features[f"lag_{lag}w"] = features.groupby("categoria")["unidades"].shift(lag)

# ROLLING AVERAGES (promedios móviles)
for window in [2, 4]:
    features[f"rolling_mean_{window}w"] = (
        features.groupby("categoria")["unidades"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    features[f"rolling_std_{window}w"] = (
        features.groupby("categoria")["unidades"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
    )

# ROLLING MAX (para detectar bursts)
features["rolling_max_4w"] = (
    features.groupby("categoria")["unidades"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).max())
)

# TASA DE CAMBIO (aceleración de demanda)
features["cambio_semanal"] = features.groupby("categoria")["unidades"].diff()
features["cambio_pct"] = features.groupby("categoria")["unidades"].pct_change().replace([np.inf, -np.inf], np.nan)

# VENTAS ACUMULADAS por categoría (tendencia de popularidad)
features["ventas_acumuladas"] = features.groupby("categoria")["unidades"].cumsum()

# ONE-HOT ENCODING de categoría (para Random Forest / XGBoost)
cat_dummies = pd.get_dummies(features["categoria"], prefix="cat", dtype=int)
features = pd.concat([features, cat_dummies], axis=1)

# ONE-HOT ENCODING de temporada
temp_dummies = pd.get_dummies(features["temporada"], prefix="temp", dtype=int)
features = pd.concat([features, temp_dummies], axis=1)

# Columna target (lo que queremos predecir)
features["target"] = features["unidades"]

# Eliminar filas con NaN en lag features (primeras semanas sin historial)
features_complete = features.dropna(subset=["lag_1w"]).copy()

# Rellenar NaN restantes con 0
features_complete = features_complete.fillna(0)

features_complete.to_csv(OUTPUT_DIR / "features_modelo.csv", index=False)

feature_cols = [c for c in features_complete.columns if c.startswith(("lag_", "rolling_", "cambio",
    "ventas_acum", "cat_", "temp_", "es_", "tiene_", "mes", "semana_iso", "semanas_desde"))]
print(f"  ✓ features_modelo.csv — {len(features_complete)} filas, {len(feature_cols)} features predictivos")
print(f"  Features creados:")
print(f"    Lag features:     lag_1w, lag_2w, lag_3w, lag_4w")
print(f"    Rolling stats:    rolling_mean_2w, rolling_mean_4w, rolling_std_2w, rolling_std_4w, rolling_max_4w")
print(f"    Cambio:           cambio_semanal, cambio_pct")
print(f"    Acumulado:        ventas_acumuladas")
print(f"    Calendario:       es_quincena, es_resolucion_ano_nuevo, tiene_evento_semana")
print(f"    Categoría:        {len([c for c in cat_dummies.columns])} dummies")
print(f"    Temporada:        {len([c for c in temp_dummies.columns])} dummies")


# ============================================================================
# PASO 7: BASELINE NAIVE (punto de comparación)
# ============================================================================
print("\n" + "=" * 70)
print("PASO 7: Calcular baseline naive para comparación")
print("=" * 70)

# Baseline: "la predicción de esta semana = ventas de la semana pasada"
baseline = features_complete[["semana_inicio", "categoria", "target", "lag_1w"]].copy()
baseline = baseline.rename(columns={"lag_1w": "prediccion_naive"})
baseline["error_abs"] = (baseline["target"] - baseline["prediccion_naive"]).abs()
baseline["error_pct"] = np.where(
    baseline["target"] > 0,
    baseline["error_abs"] / baseline["target"] * 100,
    np.where(baseline["prediccion_naive"] > 0, 100, 0)
)

# Métricas globales
mae_global = baseline["error_abs"].mean()
non_zero = baseline[baseline["target"] > 0]
mape_global = non_zero["error_pct"].mean() if len(non_zero) > 0 else float("inf")

# R² del baseline
ss_res = ((baseline["target"] - baseline["prediccion_naive"]) ** 2).sum()
ss_tot = ((baseline["target"] - baseline["target"].mean()) ** 2).sum()
r2_global = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

baseline.to_csv(OUTPUT_DIR / "baseline_naive.csv", index=False)

print(f"  ✓ baseline_naive.csv")
print(f"  Métricas del baseline naive (punto de referencia):")
print(f"    MAE  = {mae_global:.3f} unidades")
print(f"    MAPE = {mape_global:.1f}%")
print(f"    R²   = {r2_global:.4f}")
print(f"  → Tu modelo debe superar estas métricas por 20-25% para cumplir el objetivo de la tesis")

target_mae = mae_global * 0.75
target_mape = mape_global * 0.75
print(f"  → Objetivo MAE  ≤ {target_mae:.3f}")
print(f"  → Objetivo MAPE ≤ {target_mape:.1f}%")


# ============================================================================
# PASO 8: RESUMEN EDA (para documentación de tesis)
# ============================================================================
print("\n" + "=" * 70)
print("PASO 8: Resumen exploratorio (EDA) para documentación")
print("=" * 70)

eda = df.groupby("categoria").agg(
    total_ventas=("venta_id", "count"),
    ingresos_total=("ingresos", "sum"),
    total_neto=("total_neto", "sum"),
    ticket_promedio=("ingresos", "mean"),
    n_disenos=("diseno", "nunique"),
    n_estados=("estado_geo", "nunique"),
    primera_venta=("fecha", "min"),
    ultima_venta=("fecha", "max"),
    tipo_prenda_mas_comun=("tipo_prenda", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A"),
).reset_index()

eda["participacion_pct"] = (eda["total_ventas"] / eda["total_ventas"].sum() * 100).round(1)
eda["margen_pct"] = (eda["total_neto"] / eda["ingresos_total"] * 100).round(1)
eda = eda.sort_values("total_ventas", ascending=False)

eda.to_csv(OUTPUT_DIR / "resumen_eda.csv", index=False)
print(f"  ✓ resumen_eda.csv")
print(f"\n  Top categorías:")
for _, row in eda.head(5).iterrows():
    print(f"    {row['categoria']:15s} → {row['total_ventas']:3.0f} ventas ({row['participacion_pct']}%), "
          f"margen {row['margen_pct']}%, {row['n_disenos']} diseños")


# ============================================================================
# PASO 9: METADATA DEL PIPELINE (para trazabilidad)
# ============================================================================
print("\n" + "=" * 70)
print("PASO 9: Guardar metadata del pipeline")
print("=" * 70)

metadata = {
    "pipeline_version": "1.0.0",
    "fecha_ejecucion": datetime.now().isoformat(),
    "archivo_entrada": INPUT_FILE,
    "registros_entrada": len(df),
    "rango_fechas": {
        "inicio": df["fecha"].min().isoformat(),
        "fin": df["fecha"].max().isoformat(),
        "semanas": int(n_weeks),
    },
    "categorias": sorted(df["categoria"].unique().tolist()),
    "tipos_prenda": sorted(df["tipo_prenda"].unique().tolist()),
    "archivos_generados": [
        {"nombre": "ventas_limpias.csv", "filas": len(df_clean), "descripcion": "Transacciones limpias con features temporales"},
        {"nombre": "serie_semanal.csv", "filas": len(serie), "descripcion": "Serie de tiempo semanal por categoría"},
        {"nombre": "features_modelo.csv", "filas": len(features_complete), "descripcion": "Dataset con features para ML"},
        {"nombre": "baseline_naive.csv", "filas": len(baseline), "descripcion": "Predicciones baseline naive"},
        {"nombre": "resumen_eda.csv", "filas": len(eda), "descripcion": "Resumen exploratorio por categoría"},
    ],
    "baseline_metricas": {
        "MAE": round(mae_global, 4),
        "MAPE": round(mape_global, 2),
        "R2": round(r2_global, 4),
    },
    "features_modelo": {
        "lag_features": ["lag_1w", "lag_2w", "lag_3w", "lag_4w"],
        "rolling_features": ["rolling_mean_2w", "rolling_mean_4w", "rolling_std_2w", "rolling_std_4w", "rolling_max_4w"],
        "cambio_features": ["cambio_semanal", "cambio_pct"],
        "acumulado": ["ventas_acumuladas"],
        "calendario": ["es_resolucion_ano_nuevo", "tiene_evento_semana", "mes", "semana_iso", "semanas_desde_inicio"],
        "categoricas": [c for c in cat_dummies.columns] + [c for c in temp_dummies.columns],
    },
    "nota_google_trends": "Pendiente: agregar columnas de Google Trends como features externos. Usar pytrends para consultar términos por categoría.",
}

with open(OUTPUT_DIR / "pipeline_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

print(f"  ✓ pipeline_metadata.json")


# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("PIPELINE ETL COMPLETADO")
print("=" * 70)
print(f"""
  Archivos generados en {OUTPUT_DIR}/:
  
  1. ventas_limpias.csv      → {len(df_clean)} transacciones limpias
  2. serie_semanal.csv       → {len(serie)} filas (serie temporal)
  3. features_modelo.csv     → {len(features_complete)} filas listas para ML
  4. baseline_naive.csv      → {len(baseline)} predicciones de referencia
  5. resumen_eda.csv         → {len(eda)} categorías analizadas
  6. pipeline_metadata.json  → Trazabilidad completa
  
  Baseline naive:
    MAE  = {mae_global:.3f}
    MAPE = {mape_global:.1f}%
    R²   = {r2_global:.4f}
  
  SIGUIENTE PASO:
    → Integrar Google Trends API (pytrends) como features externos
    → Entrenar SARIMA, Random Forest y XGBoost sobre features_modelo.csv
    → Comparar métricas vs baseline_naive.csv
""")