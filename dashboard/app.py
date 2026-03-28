"""
Streamlit Dashboard v4.0 — DTF Fashion Predictive Analytics Platform
5 Tabs: Pronósticos, Comparación de Modelos, Producción, Google Trends, Datos.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.connection import read_sql, engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dashboard")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DTF Fashion — Análisis Predictivo",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Colores del tema
COLORS = {
    "primary": "#1E3A5F",
    "accent": "#3B82F6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "gray": "#6B7280",
    "light": "#F3F4F6",
    "banda": "rgba(59,130,246,0.15)",
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def cargar_serie():
    try:
        return read_sql("SELECT * FROM serie_semanal ORDER BY fecha")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def cargar_predicciones():
    try:
        return read_sql("""
            SELECT p.*, t.modelo_ganador
            FROM predicciones p
            LEFT JOIN training_runs t ON p.run_id = t.run_id
            ORDER BY p.modelo, p.fecha_prediccion
        """)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def cargar_metricas():
    try:
        runs = read_sql("SELECT * FROM training_runs ORDER BY fecha_ejecucion DESC LIMIT 1")
        if runs.empty:
            return pd.DataFrame(), {}
        run = runs.iloc[0]
        metricas = read_sql(f"SELECT * FROM metricas_modelos WHERE run_id = '{run['run_id']}' ORDER BY mape")
        return metricas, run.to_dict()
    except Exception:
        return pd.DataFrame(), {}


@st.cache_data(ttl=60)
def cargar_ventas():
    try:
        return read_sql("SELECT * FROM ventas ORDER BY fecha")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def cargar_factores_hm():
    try:
        return read_sql("SELECT * FROM factores_hm ORDER BY tipo, clave")
    except Exception:
        return pd.DataFrame()


def check_db():
    """Verifica conexión a base de datos."""
    try:
        df = read_sql("SELECT 1 AS ok")
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fabric-textile.png", width=60)
    st.title("DTF Fashion")
    st.caption("Plataforma de Análisis Predictivo con IA")
    st.divider()

    # Status de DB
    db_ok = check_db()
    if db_ok:
        st.success("🟢 Base de datos conectada")
    else:
        st.error("🔴 Sin conexión a base de datos")
        st.info("Verifica que PostgreSQL esté corriendo o que SQLite esté disponible.")

    st.divider()

    # Upload de datos
    st.subheader("📁 Subir datos")
    uploaded_file = st.file_uploader(
        "Excel o CSV con ventas",
        type=["xlsx", "xls", "csv"],
        help="Archivo con columnas: fecha, cantidad/unidades, precio (opcional), producto (opcional), categoría (opcional)",
    )

    if uploaded_file:
        if st.button("🚀 Procesar datos", type="primary", use_container_width=True):
            with st.spinner("Ejecutando pipeline ETL..."):
                try:
                    # Guardar temporalmente
                    import tempfile
                    tmp = Path(tempfile.mkdtemp()) / uploaded_file.name
                    tmp.write_bytes(uploaded_file.read())

                    from etl.etl_pipeline import ejecutar_pipeline
                    resultado = ejecutar_pipeline(str(tmp))

                    st.success(f"✅ {resultado['filas_limpias']} registros procesados")
                    st.json(resultado)

                    # Limpiar cache
                    st.cache_data.clear()

                    # Limpiar temporal
                    import shutil
                    shutil.rmtree(tmp.parent, ignore_errors=True)

                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Entrenamiento
    st.subheader("🧠 Entrenar modelos")
    if st.button("Entrenar SARIMA + Prophet + RF", use_container_width=True):
        with st.spinner("Entrenando modelos... (puede tardar 1-2 min)"):
            try:
                from models.train_models import ejecutar_entrenamiento
                resultado = ejecutar_entrenamiento()

                if resultado["status"] == "ok":
                    st.success(f"🏆 Ganador: {resultado['ganador']}")
                    st.cache_data.clear()
                else:
                    st.error(resultado.get("mensaje", "Error desconocido"))
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.caption("v4.0 — Proyecto de titulación Ibero 2026")
    st.caption("Pipeline: SARIMA · Prophet · Random Forest · H&M Transfer")


# ═══════════════════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Pronósticos",
    "🏆 Comparación de Modelos",
    "🏭 Producción DTF",
    "🔍 Google Trends",
    "📊 Datos",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: PRONÓSTICOS
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Pronóstico de Demanda — Próximos 30 días")

    pred = cargar_predicciones()
    serie = cargar_serie()

    if pred.empty:
        st.warning("No hay predicciones aún. Sube tus datos y entrena los modelos desde la barra lateral.")
        st.stop()

    pred["fecha_prediccion"] = pd.to_datetime(pred["fecha_prediccion"])
    serie["fecha"] = pd.to_datetime(serie["fecha"])

    # Modelo ganador
    ganador = pred["modelo_ganador"].iloc[0] if "modelo_ganador" in pred.columns else "N/A"
    pred_ganador = pred[pred["modelo"] == ganador]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total = pred_ganador["unidades_predichas"].sum()
        st.metric("📦 Unidades estimadas (30d)", f"{total:.0f}")
    with col2:
        rango_inf = pred_ganador["banda_inferior"].sum()
        rango_sup = pred_ganador["banda_superior"].sum()
        st.metric("📊 Rango ±30%", f"{rango_inf:.0f} – {rango_sup:.0f}")
    with col3:
        prom = pred_ganador["unidades_predichas"].mean()
        st.metric("📅 Promedio diario", f"{prom:.1f} uds")
    with col4:
        st.metric("🏆 Modelo ganador", ganador)

    st.divider()

    # Gráfica principal: Histórico + Forecast con bandas
    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Bar(
        x=serie["fecha"],
        y=serie["unidades"],
        name="Ventas reales",
        marker_color=COLORS["accent"],
        opacity=0.6,
    ))

    # Forecast del ganador
    fig.add_trace(go.Scatter(
        x=pred_ganador["fecha_prediccion"],
        y=pred_ganador["unidades_predichas"],
        name=f"Forecast ({ganador})",
        mode="lines+markers",
        line=dict(color=COLORS["success"], width=2.5),
        marker=dict(size=5),
    ))

    # Banda de incertidumbre
    fig.add_trace(go.Scatter(
        x=pd.concat([pred_ganador["fecha_prediccion"], pred_ganador["fecha_prediccion"][::-1]]),
        y=pd.concat([pred_ganador["banda_superior"], pred_ganador["banda_inferior"][::-1]]),
        fill="toself",
        fillcolor=COLORS["banda"],
        line=dict(color="rgba(0,0,0,0)"),
        name="Banda ±30%",
        showlegend=True,
    ))

    fig.update_layout(
        title="Ventas Históricas + Pronóstico 30 días",
        xaxis_title="Fecha",
        yaxis_title="Unidades vendidas",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Selector de modelo para ver todos
    st.subheader("Ver por modelo")
    modelos_disponibles = pred["modelo"].unique().tolist()
    modelo_sel = st.selectbox("Selecciona modelo:", modelos_disponibles, index=0)

    pred_sel = pred[pred["modelo"] == modelo_sel]
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(f"Total {modelo_sel}", f"{pred_sel['unidades_predichas'].sum():.0f} uds")
    with col_b:
        st.metric("Promedio diario", f"{pred_sel['unidades_predichas'].mean():.1f} uds")

    # Tabla de predicciones
    with st.expander("📋 Tabla de predicciones detallada"):
        st.dataframe(
            pred_sel[["fecha_prediccion", "unidades_predichas", "banda_inferior", "banda_superior", "dia_horizonte"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: COMPARACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Comparación de Modelos de ML")

    metricas_df, run_info = cargar_metricas()

    if metricas_df.empty:
        st.warning("No hay métricas. Entrena los modelos primero.")
    else:
        # Info del run
        st.caption(
            f"Run: `{run_info.get('run_id', '?')}` | "
            f"Fecha: {run_info.get('fecha_ejecucion', '?')} | "
            f"Datos: {run_info.get('n_datos', '?')} días"
        )

        # KPIs del run
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Modelo Ganador", run_info.get("modelo_ganador", "?"))
        with col2:
            baseline_mape = run_info.get("baseline_mape", 0)
            st.metric("📌 Baseline MAPE", f"{baseline_mape:.1f}%")
        with col3:
            mejora = run_info.get("mejora_pct", 0)
            color = "normal" if mejora and mejora >= 20 else "off"
            st.metric(
                "📈 Mejora vs Baseline",
                f"{mejora:+.1f}%" if mejora else "N/A",
                delta="✅ Objetivo ≥20%" if mejora and mejora >= 20 else "⚠️ Debajo del objetivo",
            )

        st.divider()

        # Gráfica de barras comparativa
        fig_comp = make_subplots(
            rows=1, cols=3,
            subplot_titles=["MAE (menor = mejor)", "MAPE % (menor = mejor)", "R² (mayor = mejor)"],
        )

        colores = [COLORS["accent"], COLORS["success"], COLORS["warning"], COLORS["danger"]][:len(metricas_df)]
        es_ganador = metricas_df["es_ganador"].astype(bool) if "es_ganador" in metricas_df.columns else [False] * len(metricas_df)
        bar_colors = [COLORS["success"] if g else COLORS["gray"] for g in es_ganador]

        fig_comp.add_trace(go.Bar(
            x=metricas_df["modelo"], y=metricas_df["mae"],
            marker_color=bar_colors, name="MAE", showlegend=False,
            text=metricas_df["mae"].round(2), textposition="outside",
        ), row=1, col=1)

        fig_comp.add_trace(go.Bar(
            x=metricas_df["modelo"], y=metricas_df["mape"],
            marker_color=bar_colors, name="MAPE", showlegend=False,
            text=metricas_df["mape"].round(1).astype(str) + "%", textposition="outside",
        ), row=1, col=2)

        fig_comp.add_trace(go.Bar(
            x=metricas_df["modelo"], y=metricas_df["r2"],
            marker_color=bar_colors, name="R²", showlegend=False,
            text=metricas_df["r2"].round(3), textposition="outside",
        ), row=1, col=3)

        # Agregar línea de baseline al MAPE
        if baseline_mape:
            fig_comp.add_hline(
                y=baseline_mape, line_dash="dash", line_color=COLORS["danger"],
                annotation_text=f"Baseline: {baseline_mape:.1f}%",
                row=1, col=2,
            )

        fig_comp.update_layout(
            height=400, template="plotly_white",
            title="Métricas de Desempeño por Modelo",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Tabla de métricas
        st.subheader("Detalle de métricas")
        display_df = metricas_df[["modelo", "mae", "mape", "r2", "parametros", "es_ganador"]].copy()
        display_df.columns = ["Modelo", "MAE", "MAPE (%)", "R²", "Parámetros", "Ganador"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Forecast comparativo (todos los modelos)
        st.subheader("Forecast comparativo — Todos los modelos")
        pred = cargar_predicciones()
        if not pred.empty:
            pred["fecha_prediccion"] = pd.to_datetime(pred["fecha_prediccion"])
            fig_all = go.Figure()
            line_styles = ["solid", "dash", "dot", "dashdot"]
            for i, (modelo, grupo) in enumerate(pred.groupby("modelo")):
                es_winner = modelo == run_info.get("modelo_ganador")
                fig_all.add_trace(go.Scatter(
                    x=grupo["fecha_prediccion"],
                    y=grupo["unidades_predichas"],
                    name=f"{'🏆 ' if es_winner else ''}{modelo}",
                    mode="lines+markers",
                    line=dict(
                        width=3 if es_winner else 1.5,
                        dash=line_styles[i % len(line_styles)],
                    ),
                    marker=dict(size=4 if es_winner else 2),
                ))
            fig_all.update_layout(
                title="Pronóstico de todos los modelos",
                template="plotly_white",
                height=400,
                hovermode="x unified",
            )
            st.plotly_chart(fig_all, use_container_width=True)

        # Historial de training runs
        with st.expander("📜 Historial de entrenamientos"):
            try:
                runs = read_sql("SELECT * FROM training_runs ORDER BY fecha_ejecucion DESC LIMIT 20")
                if not runs.empty:
                    st.dataframe(runs, use_container_width=True, hide_index=True)
            except Exception:
                st.info("Sin historial de entrenamientos.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: PRODUCCIÓN DTF
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Recomendaciones de Producción DTF")

    pred = cargar_predicciones()
    serie = cargar_serie()
    ventas = cargar_ventas()

    if pred.empty:
        st.warning("Entrena los modelos para ver recomendaciones.")
    else:
        pred["fecha_prediccion"] = pd.to_datetime(pred["fecha_prediccion"])
        serie["fecha"] = pd.to_datetime(serie["fecha"])

        ganador = pred["modelo_ganador"].iloc[0] if "modelo_ganador" in pred.columns else pred["modelo"].iloc[0]
        pg = pred[pred["modelo"] == ganador]

        # Resumen de producción
        st.subheader("📦 Plan de producción sugerido — 30 días")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Escenario conservador",
                f"{pg['banda_inferior'].sum():.0f} uds",
                help="Banda inferior (−30%)",
            )
        with col2:
            st.metric(
                "Escenario central",
                f"{pg['unidades_predichas'].sum():.0f} uds",
                help="Predicción del modelo ganador",
            )
        with col3:
            st.metric(
                "Escenario optimista",
                f"{pg['banda_superior'].sum():.0f} uds",
                help="Banda superior (+30%)",
            )

        st.divider()

        # Forecast semanal
        pg_copy = pg.copy()
        pg_copy["semana"] = pg_copy["fecha_prediccion"].dt.isocalendar().week.astype(int)
        semanal = pg_copy.groupby("semana").agg(
            unidades=("unidades_predichas", "sum"),
            inferior=("banda_inferior", "sum"),
            superior=("banda_superior", "sum"),
        ).reset_index()

        fig_sem = go.Figure()
        fig_sem.add_trace(go.Bar(
            x=semanal["semana"].astype(str).apply(lambda x: f"Sem {x}"),
            y=semanal["unidades"],
            name="Forecast",
            marker_color=COLORS["accent"],
            text=semanal["unidades"].round(0).astype(int),
            textposition="outside",
        ))
        fig_sem.add_trace(go.Scatter(
            x=semanal["semana"].astype(str).apply(lambda x: f"Sem {x}"),
            y=semanal["superior"],
            mode="markers",
            marker=dict(color=COLORS["success"], size=8, symbol="triangle-up"),
            name="Optimista (+30%)",
        ))
        fig_sem.add_trace(go.Scatter(
            x=semanal["semana"].astype(str).apply(lambda x: f"Sem {x}"),
            y=semanal["inferior"],
            mode="markers",
            marker=dict(color=COLORS["warning"], size=8, symbol="triangle-down"),
            name="Conservador (−30%)",
        ))
        fig_sem.update_layout(
            title="Producción sugerida por semana",
            yaxis_title="Unidades",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_sem, use_container_width=True)

        # Análisis por categoría
        if not ventas.empty and "categoria" in ventas.columns:
            st.subheader("🏷️ Rendimiento por categoría")
            col_a, col_b = st.columns(2)

            cat_vol = ventas.groupby("categoria")["cantidad"].sum().sort_values(ascending=False)
            cat_ing = ventas.groupby("categoria")["ingreso_bruto"].sum().sort_values(ascending=False)

            with col_a:
                fig_cat = px.pie(
                    values=cat_vol.values,
                    names=cat_vol.index,
                    title="Distribución por volumen",
                    hole=0.4,
                )
                fig_cat.update_layout(height=350)
                st.plotly_chart(fig_cat, use_container_width=True)

            with col_b:
                fig_ing = px.bar(
                    x=cat_ing.index,
                    y=cat_ing.values,
                    title="Ingreso por categoría ($MXN)",
                    labels={"x": "Categoría", "y": "Ingreso"},
                    color=cat_ing.values,
                    color_continuous_scale="Blues",
                )
                fig_ing.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_ing, use_container_width=True)

        # Recomendaciones textuales
        st.subheader("💡 Recomendaciones accionables")

        prom_hist = serie["unidades"].mean()
        prom_pred = pg["unidades_predichas"].mean()
        cambio = ((prom_pred - prom_hist) / prom_hist * 100) if prom_hist > 0 else 0

        if cambio > 20:
            st.success(f"📈 **Oportunidad**: La demanda estimada supera el promedio histórico en {cambio:.0f}%. Asegura stock de insumos DTF.")
        elif cambio < -10:
            st.warning(f"📉 **Precaución**: La demanda estimada está {abs(cambio):.0f}% debajo del promedio. Reduce tirajes.")
        else:
            st.info(f"➡️ La demanda se mantiene estable ({cambio:+.0f}% vs histórico).")

        pg_copy["dia_semana"] = pg_copy["fecha_prediccion"].dt.dayofweek
        mejor_dia_num = pg_copy.groupby("dia_semana")["unidades_predichas"].mean().idxmax()
        dias = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
        st.info(f"📅 **Día más fuerte**: {dias.get(mejor_dia_num, '?')} — considera concentrar lanzamientos y promociones.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: GOOGLE TRENDS
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Google Trends — Tendencias de diseño")
    st.caption("Detecta tendencias emergentes para diseños DTF usando Google Trends")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        keywords_input = st.text_input(
            "Keywords (separadas por coma, max 5)",
            value="playera personalizada, DTF printing, diseño streetwear",
            help="Ejemplo: streetwear, anime shirts, custom t-shirt",
        )
    with col2:
        timeframe = st.selectbox(
            "Período",
            ["today 1-m", "today 3-m", "today 12-m"],
            index=1,
            format_func=lambda x: {"today 1-m": "1 mes", "today 3-m": "3 meses", "today 12-m": "12 meses"}[x],
        )
    with col3:
        geo = st.selectbox("País", ["MX", "US", "CO", "AR", "ES", ""], index=0,
                           format_func=lambda x: x if x else "Global")

    if st.button("🔍 Buscar tendencias", type="primary"):
        with st.spinner("Consultando Google Trends..."):
            try:
                from pytrends.request import TrendReq
                kw_list = [k.strip() for k in keywords_input.split(",")][:5]

                pytrends = TrendReq(hl="es-MX", tz=360, timeout=(10, 25))
                pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo=geo)

                interest = pytrends.interest_over_time()

                if interest.empty:
                    st.warning("Sin datos para estas keywords.")
                else:
                    if "isPartial" in interest.columns:
                        interest = interest.drop("isPartial", axis=1)

                    # Gráfica de tendencias
                    fig_trends = go.Figure()
                    for kw in kw_list:
                        if kw in interest.columns:
                            fig_trends.add_trace(go.Scatter(
                                x=interest.index,
                                y=interest[kw],
                                name=kw,
                                mode="lines",
                            ))
                    fig_trends.update_layout(
                        title="Interés en el tiempo — Google Trends",
                        yaxis_title="Interés relativo (0-100)",
                        template="plotly_white",
                        height=400,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_trends, use_container_width=True)

                    # Resumen
                    st.subheader("Resumen de interés")
                    summary = interest.describe().loc[["mean", "max", "min"]].round(1)
                    st.dataframe(summary, use_container_width=True)

                    # Related queries
                    try:
                        related = pytrends.related_queries()
                        for kw in kw_list:
                            if kw in related and related[kw]["top"] is not None:
                                with st.expander(f"🔗 Búsquedas relacionadas: {kw}"):
                                    st.dataframe(related[kw]["top"].head(10), use_container_width=True, hide_index=True)
                    except Exception:
                        pass

            except ImportError:
                st.error("pytrends no está instalado. Ejecuta: `pip install pytrends`")
            except Exception as e:
                st.error(f"Error al consultar Google Trends: {e}")

    # Estacionalidad H&M como fallback visual
    st.divider()
    st.subheader("📅 Patrones estacionales H&M (referencia)")

    factores = cargar_factores_hm()
    if not factores.empty:
        col_a, col_b = st.columns(2)

        semanal = factores[factores["tipo"] == "semanal"].copy()
        mensual = factores[factores["tipo"] == "mensual"].copy()

        if not semanal.empty:
            semanal["dia"] = semanal["clave"].astype(int)
            dias_map = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
            semanal["nombre"] = semanal["dia"].map(dias_map)
            semanal = semanal.sort_values("dia")

            with col_a:
                fig_sw = px.bar(
                    semanal, x="nombre", y="valor",
                    title="Índice semanal H&M",
                    labels={"valor": "Índice", "nombre": "Día"},
                    color="valor", color_continuous_scale="Blues",
                )
                fig_sw.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig_sw.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_sw, use_container_width=True)

        if not mensual.empty:
            mensual["mes_num"] = mensual["clave"].astype(int)
            meses_map = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
                         7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
            mensual["nombre"] = mensual["mes_num"].map(meses_map)
            mensual = mensual.sort_values("mes_num")

            with col_b:
                fig_mm = px.bar(
                    mensual, x="nombre", y="valor",
                    title="Índice mensual H&M",
                    labels={"valor": "Índice", "nombre": "Mes"},
                    color="valor", color_continuous_scale="Greens",
                )
                fig_mm.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig_mm.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_mm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: DATOS
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Exploración de Datos")

    serie = cargar_serie()
    ventas = cargar_ventas()

    if serie.empty:
        st.warning("No hay datos cargados. Sube un archivo Excel/CSV desde la barra lateral.")
    else:
        serie["fecha"] = pd.to_datetime(serie["fecha"])

        # KPIs generales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total unidades", f"{serie['unidades'].sum():.0f}")
        with col2:
            total_ingreso = serie["ingreso_bruto"].sum()
            st.metric("Ingreso bruto", f"${total_ingreso:,.0f} MXN")
        with col3:
            dias_activos = (serie["unidades"] > 0).sum()
            st.metric("Días con ventas", f"{dias_activos} de {len(serie)}")
        with col4:
            if not ventas.empty and "producto" in ventas.columns:
                n_productos = ventas["producto"].nunique()
                st.metric("Productos únicos", n_productos)

        st.divider()

        # Gráfica de ventas diarias + ingreso acumulado
        fig_hist = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Ventas diarias", "Ingreso acumulado ($MXN)"],
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4],
        )

        fig_hist.add_trace(go.Bar(
            x=serie["fecha"], y=serie["unidades"],
            name="Unidades", marker_color=COLORS["accent"], opacity=0.7,
        ), row=1, col=1)

        if "ingreso_acumulado" in serie.columns:
            fig_hist.add_trace(go.Scatter(
                x=serie["fecha"], y=serie["ingreso_acumulado"],
                name="Ingreso acumulado", fill="tozeroy",
                line=dict(color=COLORS["success"]),
            ), row=2, col=1)

        fig_hist.update_layout(height=600, template="plotly_white", showlegend=True)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Distribución por día de la semana
        col_a, col_b = st.columns(2)
        with col_a:
            por_dia = serie.groupby("dia_semana")["unidades"].mean()
            dias_map = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
            fig_dia = px.bar(
                x=[dias_map.get(d, d) for d in por_dia.index],
                y=por_dia.values,
                title="Promedio de ventas por día",
                labels={"x": "Día", "y": "Unidades promedio"},
                color=por_dia.values,
                color_continuous_scale="Blues",
            )
            fig_dia.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_dia, use_container_width=True)

        with col_b:
            por_mes = serie.groupby("mes")["unidades"].mean()
            meses_map = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
                         7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
            fig_mes = px.bar(
                x=[meses_map.get(m, m) for m in por_mes.index],
                y=por_mes.values,
                title="Promedio de ventas por mes",
                labels={"x": "Mes", "y": "Unidades promedio"},
                color=por_mes.values,
                color_continuous_scale="Greens",
            )
            fig_mes.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_mes, use_container_width=True)

        # Tabla de datos raw
        with st.expander("📋 Datos crudos — Serie temporal completa"):
            st.dataframe(
                serie.sort_values("fecha", ascending=False).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=400,
            )

        if not ventas.empty:
            with st.expander("📋 Datos crudos — Transacciones individuales"):
                st.dataframe(
                    ventas.sort_values("fecha", ascending=False).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )

        # Descarga
        st.divider()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_serie = serie.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar serie temporal (CSV)",
                csv_serie,
                "dtf_serie_temporal.csv",
                "text/csv",
            )
        with col_d2:
            if not ventas.empty:
                csv_ventas = ventas.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Descargar ventas (CSV)",
                    csv_ventas,
                    "dtf_ventas.csv",
                    "text/csv",
                )