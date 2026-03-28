-- ============================================================================
-- DTF Predictive Platform — Esquema PostgreSQL
-- ============================================================================
-- Se ejecuta automáticamente al iniciar el contenedor de PostgreSQL
-- via docker-entrypoint-initdb.d/
--
-- Tablas:
--   ventas            → Transacciones limpias (output del ETL)
--   serie_semanal     → Serie de tiempo semanal por categoría
--   features          → Features para modelos ML
--   predicciones      → Pronósticos generados por los modelos
--   metricas_modelos  → Comparación de rendimiento de modelos
--   training_runs     → Log de cada entrenamiento ejecutado
--   factores_hm       → Factores de corrección H&M → DTF
-- ============================================================================

-- ── Ventas (transacciones individuales) ──
CREATE TABLE IF NOT EXISTS ventas (
    id              SERIAL PRIMARY KEY,
    venta_id        INTEGER UNIQUE,
    fecha           DATE NOT NULL,
    anio_semana     VARCHAR(10),
    estado_venta    VARCHAR(50),
    ingresos        NUMERIC(10,2) DEFAULT 0,
    cargo_venta     NUMERIC(10,2) DEFAULT 0,
    costo_envio     NUMERIC(10,2) DEFAULT 0,
    total_neto      NUMERIC(10,2) DEFAULT 0,
    diseno          VARCHAR(200),
    categoria       VARCHAR(100) NOT NULL,
    tipo_prenda     VARCHAR(100),
    estado_geo      VARCHAR(100),
    pais            VARCHAR(100) DEFAULT 'México',
    es_internacional BOOLEAN DEFAULT FALSE,
    mes             INTEGER,
    semana_iso      INTEGER,
    dia_semana      INTEGER,
    temporada       VARCHAR(20),
    evento_cercano  VARCHAR(50),
    tiene_evento    BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ventas_fecha ON ventas(fecha);
CREATE INDEX IF NOT EXISTS idx_ventas_categoria ON ventas(categoria);
CREATE INDEX IF NOT EXISTS idx_ventas_fecha_cat ON ventas(fecha, categoria);

-- ── Serie semanal (agregación por semana × categoría) ──
CREATE TABLE IF NOT EXISTS serie_semanal (
    id              SERIAL PRIMARY KEY,
    semana_inicio   DATE NOT NULL,
    categoria       VARCHAR(100) NOT NULL,
    unidades        INTEGER DEFAULT 0,
    ingresos_total  NUMERIC(12,2) DEFAULT 0,
    total_neto_sum  NUMERIC(12,2) DEFAULT 0,
    ticket_promedio NUMERIC(10,2) DEFAULT 0,
    n_disenos       INTEGER DEFAULT 0,
    n_estados       INTEGER DEFAULT 0,
    mes             INTEGER,
    semana_iso      INTEGER,
    temporada       VARCHAR(20),
    tiene_evento    BOOLEAN DEFAULT FALSE,
    semanas_desde_inicio INTEGER DEFAULT 0,
    -- Índices H&M
    hm_idx_mensual  NUMERIC(6,3) DEFAULT 1.0,
    hm_idx_combinado NUMERIC(6,3) DEFAULT 1.0,
    UNIQUE(semana_inicio, categoria)
);

CREATE INDEX IF NOT EXISTS idx_serie_semana ON serie_semanal(semana_inicio);
CREATE INDEX IF NOT EXISTS idx_serie_cat ON serie_semanal(categoria);

-- ── Features para modelos ML ──
CREATE TABLE IF NOT EXISTS features (
    id              SERIAL PRIMARY KEY,
    semana_inicio   DATE NOT NULL,
    categoria       VARCHAR(100) NOT NULL,
    target          NUMERIC(10,2) DEFAULT 0,
    -- Lag features
    lag_1w          NUMERIC(10,2) DEFAULT 0,
    lag_2w          NUMERIC(10,2) DEFAULT 0,
    lag_3w          NUMERIC(10,2) DEFAULT 0,
    lag_4w          NUMERIC(10,2) DEFAULT 0,
    -- Rolling
    rolling_mean_2w NUMERIC(10,4) DEFAULT 0,
    rolling_mean_4w NUMERIC(10,4) DEFAULT 0,
    rolling_std_2w  NUMERIC(10,4) DEFAULT 0,
    rolling_std_4w  NUMERIC(10,4) DEFAULT 0,
    rolling_max_4w  NUMERIC(10,2) DEFAULT 0,
    -- Cambio
    cambio_semanal  NUMERIC(10,2) DEFAULT 0,
    ventas_acumuladas NUMERIC(10,2) DEFAULT 0,
    -- Calendario
    mes             INTEGER,
    semana_iso      INTEGER,
    semanas_desde_inicio INTEGER DEFAULT 0,
    es_resolucion   BOOLEAN DEFAULT FALSE,
    tiene_evento    BOOLEAN DEFAULT FALSE,
    -- H&M transfer
    hm_idx_mensual  NUMERIC(6,3) DEFAULT 1.0,
    hm_factor_correccion NUMERIC(6,3) DEFAULT 1.0,
    hm_pred_escalada NUMERIC(10,4) DEFAULT 0,
    UNIQUE(semana_inicio, categoria)
);

-- ── Predicciones (output de los modelos) ──
CREATE TABLE IF NOT EXISTS predicciones (
    id              SERIAL PRIMARY KEY,
    training_run_id INTEGER,
    semana_inicio   DATE NOT NULL,
    categoria       VARCHAR(100) NOT NULL,
    ventas_reales   NUMERIC(10,2),
    -- Predicciones por modelo
    pred_sarima     NUMERIC(10,2) DEFAULT 0,
    pred_prophet    NUMERIC(10,2) DEFAULT 0,
    pred_rf         NUMERIC(10,2) DEFAULT 0,
    pred_transfer   NUMERIC(10,2) DEFAULT 0,
    -- Mejor modelo (auto-seleccionado)
    pred_mejor      NUMERIC(10,2) DEFAULT 0,
    modelo_ganador  VARCHAR(50),
    -- Bandas de incertidumbre
    pred_lower      NUMERIC(10,2) DEFAULT 0,
    pred_upper      NUMERIC(10,2) DEFAULT 0,
    confianza       NUMERIC(5,1) DEFAULT 50,
    recomendacion   VARCHAR(30),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_semana ON predicciones(semana_inicio);
CREATE INDEX IF NOT EXISTS idx_pred_cat ON predicciones(categoria);
CREATE INDEX IF NOT EXISTS idx_pred_run ON predicciones(training_run_id);

-- ── Métricas de modelos (comparación) ──
CREATE TABLE IF NOT EXISTS metricas_modelos (
    id              SERIAL PRIMARY KEY,
    training_run_id INTEGER,
    modelo          VARCHAR(50) NOT NULL,
    mae             NUMERIC(10,4),
    rmse            NUMERIC(10,4),
    mape            NUMERIC(10,2),
    r2              NUMERIC(10,4),
    mejora_pct      NUMERIC(10,1) DEFAULT 0,
    es_ganador      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Log de entrenamientos ──
CREATE TABLE IF NOT EXISTS training_runs (
    id              SERIAL PRIMARY KEY,
    fecha           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    n_registros     INTEGER,
    n_categorias    INTEGER,
    cutoff_date     DATE,
    test_weeks      INTEGER DEFAULT 4,
    modelo_ganador  VARCHAR(50),
    mejor_mae       NUMERIC(10,4),
    mejor_mape      NUMERIC(10,2),
    mejor_r2        NUMERIC(10,4),
    mejora_vs_baseline NUMERIC(10,1),
    status          VARCHAR(20) DEFAULT 'completed',
    metadata_json   JSONB
);

-- ── Factores de corrección H&M ──
CREATE TABLE IF NOT EXISTS factores_hm (
    id              SERIAL PRIMARY KEY,
    mes             INTEGER NOT NULL UNIQUE,
    nombre_mes      VARCHAR(20),
    factor_correccion NUMERIC(6,3) DEFAULT 1.0,
    idx_mensual_hm  NUMERIC(6,3) DEFAULT 1.0,
    interpretacion  VARCHAR(100)
);

-- ── Insert índices H&M por defecto ──
INSERT INTO factores_hm (mes, nombre_mes, idx_mensual_hm) VALUES
    (1,  'Enero',      0.913),
    (2,  'Febrero',    0.872),
    (3,  'Marzo',      0.842),
    (4,  'Abril',      0.893),
    (5,  'Mayo',       1.126),
    (6,  'Junio',      1.431),
    (7,  'Julio',      1.167),
    (8,  'Agosto',     0.933),
    (9,  'Septiembre', 1.004),
    (10, 'Octubre',    0.964),
    (11, 'Noviembre',  1.004),
    (12, 'Diciembre',  0.852)
ON CONFLICT (mes) DO NOTHING;