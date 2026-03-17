"""
============================================================================
Dashboard Analítico — Plataforma Predictiva DTF
============================================================================
Dashboard interactivo con Streamlit que visualiza pronósticos de demanda,
rankings de categorías y recomendaciones de producción DTF.

Ejecución:
  streamlit run dashboard/app.py

Consume los CSVs generados por etl_pipeline.py y train_models.py.
No requiere que la API FastAPI esté corriendo.
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
st.set_page_config(
    page_title="DTF Predictive Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# ESTILOS CSS
# ============================================================================
st.markdown("""
<style>
    /* Tipografía general */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Métricas */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #8892a4 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        padding-top: 1.5rem;
    }

    /* Alertas de producción */
    .prod-alta {
        background: rgba(239, 68, 68, 0.12);
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .prod-media {
        background: rgba(245, 158, 11, 0.12);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .prod-baja {
        background: rgba(34, 197, 94, 0.12);
        border-left: 4px solid #22c55e;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGA DE DATOS
# ============================================================================
# Rutas relativas al proyecto
# Ajustar según la estructura de tu repositorio
DATA_PATHS = [
    Path("data/processed"),           # Si corres desde la raíz del proyecto
    Path("data"),                      # Si los datos están planos en data/
    Path("../data/processed"),         # Si corres desde dashboard/
    Path("../data"),                   # Si los datos están planos
    Path("/home/claude/outputs"),      # Desarrollo
]

MODEL_PATHS = [
    Path("models/saved"),
    Path("models"),
    Path("../models/saved"),
    Path("../models"),
    Path("/home/claude/models"),
]


def find_file(filename, search_paths):
    """Busca un archivo en múltiples rutas posibles."""
    for base in search_paths:
        path = base / filename
        if path.exists():
            return path
    return None


@st.cache_data(ttl=300)
def load_data():
    """Carga todos los datasets necesarios."""
    data = {}

    files_data = {
        "ventas": ("ventas_limpias.csv", DATA_PATHS),
        "serie": ("serie_semanal.csv", DATA_PATHS),
        "features": ("features_modelo.csv", DATA_PATHS),
        "baseline": ("baseline_naive.csv", DATA_PATHS),
        "eda": ("resumen_eda.csv", DATA_PATHS),
    }

    files_models = {
        "predicciones": ("predicciones_completas.csv", MODEL_PATHS),
        "comparacion": ("comparacion_modelos.csv", MODEL_PATHS),
        "importance": ("feature_importance.csv", MODEL_PATHS),
    }

    for key, (filename, paths) in {**files_data, **files_models}.items():
        filepath = find_file(filename, paths)
        if filepath:
            data[key] = pd.read_csv(filepath)
            if "semana_inicio" in data[key].columns:
                data[key]["semana_inicio"] = pd.to_datetime(data[key]["semana_inicio"])
            if "fecha" in data[key].columns:
                data[key]["fecha"] = pd.to_datetime(data[key]["fecha"])

    # Cargar reporte JSON
    report_path = find_file("training_report.json", MODEL_PATHS)
    if report_path:
        with open(report_path) as f:
            data["report"] = json.load(f)

    return data


data = load_data()

# Verificar que los datos se cargaron
if not data:
    st.error("No se encontraron datos. Ejecuta primero `python etl/etl_pipeline.py` y `python models/train_models.py`.")
    st.stop()

# ============================================================================
# PALETA DE COLORES
# ============================================================================
COLORS = {
    "Sports": "#ef4444",
    "Gym": "#10b981",
    "Futbol": "#3b82f6",
    "Basketball": "#8b5cf6",
    "Tenis": "#f59e0b",
    "Casual": "#6b7280",
    "Movies": "#ec4899",
    "Musica": "#f97316",
    "Hockey": "#06b6d4",
    "Deportiva": "#14b8a6",
    "Skateboarding": "#64748b",
    "Baseball": "#a855f7",
    "Ufc": "#dc2626",
}

COLOR_LIST = ["#ef4444", "#10b981", "#3b82f6", "#8b5cf6", "#f59e0b",
              "#ec4899", "#f97316", "#06b6d4", "#6b7280", "#14b8a6"]

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🎯 DTF Predictive Platform")
    st.markdown("---")

    # Selector de categoría
    if "serie" in data:
        categorias = sorted(data["serie"]["categoria"].unique())
        categoria_sel = st.selectbox(
            "Categoría",
            ["Todas"] + categorias,
            index=0,
        )
    else:
        categoria_sel = "Todas"

    st.markdown("---")

    # Info del modelo
    if "report" in data:
        report = data["report"]
        mejor = report.get("mejor_modelo", {})
        st.markdown("### Modelo activo")
        st.markdown(f"**{mejor.get('nombre', 'N/A')}**")
        st.markdown(f"Mejora: `{mejor.get('mejora', 'N/A')}`")

        fecha_train = report.get("fecha", "")
        if fecha_train:
            try:
                dt = datetime.fromisoformat(fecha_train)
                st.markdown(f"Entrenado: `{dt.strftime('%d %b %Y %H:%M')}`")
            except Exception:
                pass

    st.markdown("---")

    # Datos del negocio
    if "ventas" in data:
        v = data["ventas"]
        st.markdown("### Datos del negocio")
        st.markdown(f"📦 **{len(v)}** ventas registradas")
        st.markdown(f"📅 {v['fecha'].min().strftime('%d %b %Y')} → {v['fecha'].max().strftime('%d %b %Y')}")
        st.markdown(f"🏷️ **{v['categoria'].nunique()}** categorías")
        st.markdown(f"🎨 **{v['diseno'].nunique()}** diseños únicos")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#6b7280;text-align:center;'>"
        "Plataforma Predictiva DTF v1.0<br>Proyecto de Titulación 2026"
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# HEADER
# ============================================================================
st.markdown("# Plataforma Predictiva DTF")
st.markdown("Análisis predictivo de demanda para producción Direct-to-Film")
st.markdown("---")


# ============================================================================
# MÉTRICAS PRINCIPALES (fila superior)
# ============================================================================
if "ventas" in data and "comparacion" in data:
    v = data["ventas"]
    comp = data["comparacion"]

    # Mejor modelo
    best_row = comp[comp["modelo"] != "Baseline Naive"].sort_values("MAE").iloc[0]
    mejora = best_row.get("mejora_mae_pct", 0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_ventas = len(v)
        st.metric("Total ventas", f"{total_ventas}", delta=f"+{len(v[v['mes'] >= 3])} este mes")

    with col2:
        ingresos = v["ingresos"].sum()
        st.metric("Ingresos totales", f"${ingresos:,.0f} MXN")

    with col3:
        st.metric("Mejor modelo (R²)", f"{best_row['R2']:.2f}", delta=f"+{mejora:.0f}% vs baseline")

    with col4:
        if "predicciones" in data:
            p = data["predicciones"]
            demanda_total = p["pred_ensemble"].sum()
            st.metric("Demanda predicha", f"{demanda_total:.0f} unidades", delta="próximas 4 sem.")

    st.markdown("")


# ============================================================================
# TABS PRINCIPALES
# ============================================================================
tab_pronostico, tab_ranking, tab_alertas, tab_modelos, tab_datos = st.tabs([
    "📈 Pronósticos",
    "🏆 Ranking de Categorías",
    "🚨 Alertas de Producción",
    "🤖 Modelos ML",
    "📊 Exploración de Datos",
])


# ============================================================================
# TAB 1: PRONÓSTICOS
# ============================================================================
with tab_pronostico:
    st.markdown("### Demanda histórica vs pronóstico")

    if "serie" in data and "predicciones" in data:
        serie = data["serie"]
        pred = data["predicciones"]

        # Filtrar por categoría seleccionada
        if categoria_sel != "Todas":
            serie_filt = serie[serie["categoria"] == categoria_sel]
            pred_filt = pred[pred["categoria"] == categoria_sel]
        else:
            # Agregar todas las categorías
            serie_filt = serie.groupby("semana_inicio").agg(
                unidades=("unidades", "sum"),
                ingresos_total=("ingresos_total", "sum"),
            ).reset_index()
            pred_filt = pred.groupby("semana_inicio").agg(
                ventas_reales=("ventas_reales", "sum"),
                pred_ensemble=("pred_ensemble", "sum"),
                pred_rf=("pred_rf", "sum"),
                pred_gb=("pred_gb", "sum"),
            ).reset_index()

        # Gráfica principal: historial + predicción
        fig = go.Figure()

        # Línea histórica
        fig.add_trace(go.Scatter(
            x=serie_filt["semana_inicio"],
            y=serie_filt["unidades"],
            mode="lines+markers",
            name="Ventas reales",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=6),
        ))

        # Línea de predicción
        if len(pred_filt) > 0:
            fig.add_trace(go.Scatter(
                x=pred_filt["semana_inicio"],
                y=pred_filt["pred_ensemble"],
                mode="lines+markers",
                name="Pronóstico (ensemble)",
                line=dict(color="#10b981", width=2.5, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
            ))

            # Ventas reales en periodo de test (para comparar)
            col_real = "ventas_reales" if "ventas_reales" in pred_filt.columns else "unidades"
            if col_real in pred_filt.columns:
                fig.add_trace(go.Scatter(
                    x=pred_filt["semana_inicio"],
                    y=pred_filt[col_real],
                    mode="markers",
                    name="Real (test)",
                    marker=dict(size=10, color="#ef4444", symbol="circle-open", line=dict(width=2)),
                ))

        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Semana",
            yaxis_title="Unidades vendidas",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

        st.plotly_chart(fig, use_container_width=True)

        # Gráfica por categoría (si es "Todas")
        if categoria_sel == "Todas" and "predicciones" in data:
            st.markdown("### Pronóstico por categoría")

            pred_cat = pred.groupby("categoria")["pred_ensemble"].sum().sort_values(ascending=True)

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=pred_cat.index,
                x=pred_cat.values,
                orientation="h",
                marker=dict(
                    color=[COLORS.get(c, "#6b7280") for c in pred_cat.index],
                    cornerradius=4,
                ),
                text=[f"{v:.1f}" for v in pred_cat.values],
                textposition="outside",
            ))
            fig_bar.update_layout(
                height=max(300, len(pred_cat) * 45),
                margin=dict(l=20, r=60, t=20, b=20),
                xaxis_title="Unidades predichas (próximas 4 semanas)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_bar.update_xaxes(gridcolor="rgba(128,128,128,0.1)")

            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("Ejecuta el pipeline de entrenamiento para ver pronósticos.")


# ============================================================================
# TAB 2: RANKING DE CATEGORÍAS
# ============================================================================
with tab_ranking:
    st.markdown("### Ranking de categorías por demanda")

    if "predicciones" in data and "serie" in data:
        pred = data["predicciones"]
        serie = data["serie"]

        ranking = pred.groupby("categoria").agg(
            demanda_predicha=("pred_ensemble", "sum"),
            confianza=("confianza", "mean"),
        ).sort_values("demanda_predicha", ascending=False)

        # Calcular tendencia
        tendencias = {}
        for cat in ranking.index:
            cat_hist = serie[serie["categoria"] == cat].sort_values("semana_inicio")
            if len(cat_hist) >= 4:
                recent = cat_hist["unidades"].tail(4).values
                first = recent[:2].mean()
                second = recent[2:].mean()
                if second > first * 1.2:
                    tendencias[cat] = "📈 Subiendo"
                elif second < first * 0.8:
                    tendencias[cat] = "📉 Bajando"
                else:
                    tendencias[cat] = "➡️ Estable"
            else:
                tendencias[cat] = "❓ Sin datos"

        ranking["tendencia"] = ranking.index.map(tendencias)

        # Mostrar como cards
        for i, (cat, row) in enumerate(ranking.iterrows()):
            col_rank, col_name, col_demand, col_trend, col_conf = st.columns([0.5, 2, 1.5, 1.5, 1])

            with col_rank:
                st.markdown(f"### {i+1}")
            with col_name:
                st.markdown(f"**{cat}**")
                # Mini sparkline del historial
                cat_hist = serie[serie["categoria"] == cat].sort_values("semana_inicio")
                if len(cat_hist) > 0:
                    fig_spark = go.Figure()
                    fig_spark.add_trace(go.Scatter(
                        x=cat_hist["semana_inicio"],
                        y=cat_hist["unidades"],
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color=COLORS.get(cat, "#6b7280"), width=1.5),
                        fillcolor=f"rgba({int(COLORS.get(cat, '#6b7280')[1:3], 16)},{int(COLORS.get(cat, '#6b7280')[3:5], 16)},{int(COLORS.get(cat, '#6b7280')[5:7], 16)},0.15)",
                    ))
                    fig_spark.update_layout(
                        height=50, margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_spark, use_container_width=True, key=f"spark_{cat}")

            with col_demand:
                st.metric("Demanda", f"{row['demanda_predicha']:.1f} u.")
            with col_trend:
                st.markdown(f"<br>{row['tendencia']}", unsafe_allow_html=True)
            with col_conf:
                st.metric("Confianza", f"{row['confianza']:.0f}%")

            if i < len(ranking) - 1:
                st.markdown("---")

    else:
        st.info("Datos de predicción no disponibles.")


# ============================================================================
# TAB 3: ALERTAS DE PRODUCCIÓN
# ============================================================================
with tab_alertas:
    st.markdown("### Recomendaciones de producción DTF")
    st.markdown("Basadas en los pronósticos de demanda para las próximas 4 semanas.")

    if "predicciones" in data:
        pred = data["predicciones"]

        summary = pred.groupby("categoria").agg(
            demanda=("pred_ensemble", "sum"),
            confianza=("confianza", "mean"),
            max_semana=("pred_ensemble", "max"),
        ).sort_values("demanda", ascending=False)

        # Separar por prioridad
        alta = summary[summary["demanda"] >= 4]
        media = summary[(summary["demanda"] >= 2) & (summary["demanda"] < 4)]
        baja = summary[(summary["demanda"] >= 0.5) & (summary["demanda"] < 2)]
        none = summary[summary["demanda"] < 0.5]

        # PRIORIDAD ALTA
        if len(alta) > 0:
            st.markdown("#### 🔴 Prioridad ALTA — Producir inmediatamente")
            for cat, row in alta.iterrows():
                st.markdown(
                    f"""<div class="prod-alta">
                    <strong>{cat}</strong> — {row['demanda']:.0f} unidades predichas<br>
                    <span style="font-size:0.9em;color:#fca5a5;">
                    Acción: Imprimir {int(np.ceil(row['demanda']))}+ unidades esta semana.
                    Pico de {row['max_semana']:.1f} u/semana esperado.
                    Confianza: {row['confianza']:.0f}%
                    </span></div>""",
                    unsafe_allow_html=True,
                )

        # PRIORIDAD MEDIA
        if len(media) > 0:
            st.markdown("#### 🟡 Prioridad MEDIA — Preparar lote moderado")
            for cat, row in media.iterrows():
                st.markdown(
                    f"""<div class="prod-media">
                    <strong>{cat}</strong> — {row['demanda']:.1f} unidades predichas<br>
                    <span style="font-size:0.9em;color:#fcd34d;">
                    Acción: Preparar {int(np.ceil(row['demanda']))} unidades. Monitorear tendencia.
                    Confianza: {row['confianza']:.0f}%
                    </span></div>""",
                    unsafe_allow_html=True,
                )

        # PRIORIDAD BAJA
        if len(baja) > 0:
            st.markdown("#### 🟢 Prioridad BAJA — Impresión bajo demanda")
            for cat, row in baja.iterrows():
                st.markdown(
                    f"""<div class="prod-baja">
                    <strong>{cat}</strong> — {row['demanda']:.1f} unidades predichas<br>
                    <span style="font-size:0.9em;color:#86efac;">
                    Acción: No acumular stock. Imprimir cuando se reciba pedido.
                    </span></div>""",
                    unsafe_allow_html=True,
                )

        # SIN PRODUCCIÓN
        if len(none) > 0:
            st.markdown("#### ⚪ Sin señal de demanda")
            cats_none = ", ".join(none.index.tolist())
            st.markdown(f"Categorías sin demanda significativa predicha: **{cats_none}**")

        # Resumen visual
        st.markdown("---")
        st.markdown("### Resumen de producción")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Categorías prioridad alta", len(alta))
        with col_b:
            total_producir = summary["demanda"].sum()
            st.metric("Unidades totales sugeridas", f"{total_producir:.0f}")
        with col_c:
            conf_media = summary["confianza"].mean()
            st.metric("Confianza promedio", f"{conf_media:.0f}%")

    else:
        st.info("Ejecuta el entrenamiento para generar recomendaciones.")


# ============================================================================
# TAB 4: MODELOS ML
# ============================================================================
with tab_modelos:
    st.markdown("### Comparación de modelos")

    if "comparacion" in data:
        comp = data["comparacion"]

        # Tabla de comparación
        st.dataframe(
            comp[["modelo", "MAE", "MAPE", "R2", "mejora_mae_pct"]].rename(columns={
                "modelo": "Modelo",
                "mejora_mae_pct": "Mejora vs Baseline (%)"
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Gráfica de barras comparativa
        col_mae, col_r2 = st.columns(2)

        with col_mae:
            fig_mae = go.Figure()
            colors_bar = ["#6b7280", "#f59e0b", "#3b82f6", "#10b981"]
            fig_mae.add_trace(go.Bar(
                x=comp["modelo"],
                y=comp["MAE"],
                marker_color=colors_bar[:len(comp)],
                text=[f"{v:.3f}" for v in comp["MAE"]],
                textposition="outside",
            ))
            fig_mae.update_layout(
                title="MAE por modelo (menor = mejor)",
                height=350, margin=dict(l=20, r=20, t=50, b=80),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis_title="MAE",
            )
            fig_mae.update_xaxes(tickangle=20, gridcolor="rgba(128,128,128,0.1)")
            fig_mae.update_yaxes(gridcolor="rgba(128,128,128,0.1)")
            st.plotly_chart(fig_mae, use_container_width=True)

        with col_r2:
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=comp["modelo"],
                y=comp["R2"],
                marker_color=colors_bar[:len(comp)],
                text=[f"{v:.3f}" for v in comp["R2"]],
                textposition="outside",
            ))
            fig_r2.update_layout(
                title="R² por modelo (mayor = mejor)",
                height=350, margin=dict(l=20, r=20, t=50, b=80),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis_title="R²",
            )
            fig_r2.update_xaxes(tickangle=20, gridcolor="rgba(128,128,128,0.1)")
            fig_r2.update_yaxes(gridcolor="rgba(128,128,128,0.1)")
            st.plotly_chart(fig_r2, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.markdown("### Importancia de features")
    st.markdown("¿Qué variables son más predictivas para la demanda?")

    if "importance" in data:
        imp = data["importance"].sort_values("importance_avg", ascending=True).tail(12)

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            y=imp["feature"],
            x=imp["importance_rf"],
            name="Random Forest",
            orientation="h",
            marker_color="#3b82f6",
        ))
        fig_imp.add_trace(go.Bar(
            y=imp["feature"],
            x=imp["importance_gb"],
            name="Gradient Boosting",
            orientation="h",
            marker_color="#10b981",
        ))
        fig_imp.update_layout(
            barmode="group",
            height=max(350, len(imp) * 35),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Importancia",
        )
        fig_imp.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
        st.plotly_chart(fig_imp, use_container_width=True)

        # Interpretación
        st.markdown("""
        **Interpretación de los features principales:**
        - **cambio_semanal** — Aceleración de demanda. Si las ventas subieron esta semana vs la pasada, el modelo predice que seguirán subiendo.
        - **lag_1w** — Ventas de la semana anterior. El comportamiento reciente es altamente predictivo.
        - **ventas_acumuladas** — Popularidad total de la categoría. Categorías con más historial tienden a mantener demanda.
        - **rolling_mean_2w** — Promedio móvil de 2 semanas. Suaviza ruido y captura tendencia a corto plazo.
        """)


# ============================================================================
# TAB 5: EXPLORACIÓN DE DATOS
# ============================================================================
with tab_datos:
    st.markdown("### Exploración del dataset")

    if "ventas" in data:
        v = data["ventas"]

        col_left, col_right = st.columns(2)

        with col_left:
            # Ventas por mes
            st.markdown("#### Ventas por mes")
            monthly = v.groupby(v["fecha"].dt.to_period("M")).size().reset_index()
            monthly.columns = ["mes", "ventas"]
            monthly["mes"] = monthly["mes"].astype(str)

            fig_month = go.Figure()
            fig_month.add_trace(go.Bar(
                x=monthly["mes"],
                y=monthly["ventas"],
                marker_color="#3b82f6",
                marker_cornerradius=4,
                text=monthly["ventas"],
                textposition="outside",
            ))
            fig_month.update_layout(
                height=300, margin=dict(l=20, r=20, t=20, b=40),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Unidades",
            )
            fig_month.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
            fig_month.update_yaxes(gridcolor="rgba(128,128,128,0.1)")
            st.plotly_chart(fig_month, use_container_width=True)

        with col_right:
            # Distribución por tipo de prenda
            st.markdown("#### Tipo de prenda")
            tipo_dist = v["tipo_prenda"].value_counts()

            fig_tipo = go.Figure()
            fig_tipo.add_trace(go.Pie(
                labels=tipo_dist.index,
                values=tipo_dist.values,
                hole=0.4,
                marker=dict(colors=COLOR_LIST[:len(tipo_dist)]),
                textinfo="label+percent",
                textposition="outside",
            ))
            fig_tipo.update_layout(
                height=300, margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_tipo, use_container_width=True)

        # Mapa de calor: categoría × mes
        st.markdown("#### Heatmap: categorías por mes")
        v["mes_str"] = v["fecha"].dt.strftime("%Y-%m")
        heatmap_data = v.pivot_table(
            index="categoria",
            columns="mes_str",
            values="venta_id",
            aggfunc="count",
            fill_value=0,
        )

        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="Blues",
            text=heatmap_data.values,
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        fig_heat.update_layout(
            height=max(300, len(heatmap_data) * 30),
            margin=dict(l=20, r=20, t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Distribución geográfica
        st.markdown("#### Top estados por ventas")
        geo = v["estado_geo"].value_counts().head(10)
        fig_geo = go.Figure()
        fig_geo.add_trace(go.Bar(
            x=geo.values,
            y=geo.index,
            orientation="h",
            marker_color="#8b5cf6",
            marker_cornerradius=4,
            text=geo.values,
            textposition="outside",
        ))
        fig_geo.update_layout(
            height=350, margin=dict(l=20, r=40, t=20, b=20),
            xaxis_title="Ventas",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_geo.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
        st.plotly_chart(fig_geo, use_container_width=True)

        # Tabla de datos crudos
        st.markdown("#### Datos de ventas")
        cols_show = ["venta_id", "fecha", "diseno", "categoria", "tipo_prenda",
                     "ingresos", "total_neto", "estado_geo"]
        cols_available = [c for c in cols_show if c in v.columns]
        st.dataframe(
            v[cols_available].sort_values("fecha", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=400,
        )

    else:
        st.info("Datos de ventas no disponibles.")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6b7280;font-size:0.8rem;'>"
    "Plataforma Predictiva DTF · Proyecto de Titulación 2026 · "
    "Metodología APQP · Python + FastAPI + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)