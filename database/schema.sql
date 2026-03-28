-- ============================================================================
-- DTF Predictive Platform — Esquema PostgreSQL v4.0
-- ============================================================================
-- Se ejecuta automáticamente al iniciar el contenedor de PostgreSQL
-- via docker-entrypoint-initdb.d/
--
-- IMPORTANTE: Este schema está alineado 1:1 con el código v4.0 de:
--   etl/etl_pipeline.py      → escribe a: ventas, serie_semanal, features, factores_hm
--   models/train_models.py   → escribe a: training_runs, metricas_modelos, predicciones
--   api/main.py              → lee todas las tablas via JOINs con run_id
--   dashboard/app.py         → lee todas las tablas via database.connection
--
-- Tablas (7):
--   1. ventas            → Transacciones limpias (output del ETL)
--   2. serie_semanal     → Serie de tiempo DIARIA agregada
--   3. features          → Features para modelos ML (lags, rolling, H&M)
--   4. predicciones      → Pronósticos generados por los modelos
--   5. metricas_modelos  → Comparación de rendimiento de modelos
--   6. training_runs     → Log de cada entrenamiento ejecutado
--   7. factores_hm       → Índices estacionales H&M + correcciones DTF
-- ============================================================================


-- ══════════════════════════════════════════════════════════════════════════
-- 1. VENTAS — Transacciones individuales del usuario
-- ══════════════════════════════════════════════════════════════════════════
-- Escrito por: etl_pipeline.py → escribir_a_db()
-- Columnas vienen de: limpiar_datos() → cols_ventas
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ventas (
    id              SERIAL PRIMARY KEY,
    venta_id        VARCHAR(20),
    fecha           TIMESTAMP NOT NULL,
    producto        VARCHAR(255) DEFAULT 'general',
    categoria       VARCHAR(100) DEFAULT 'General',
    cantidad        INTEGER DEFAULT 1,
    precio_unitario NUMERIC(12,2) DEFAULT 0,
    ingreso_bruto   NUMERIC(12,2) DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ventas_fecha ON ventas(fecha);
CREATE INDEX IF NOT EXISTS idx_ventas_categoria ON ventas(categoria);


-- ══════════════════════════════════════════════════════════════════════════
-- 2. SERIE_SEMANAL — Serie temporal DIARIA agregada
-- ══════════════════════════════════════════════════════════════════════════
-- Escrito por: etl_pipeline.py → agregar_serie_semanal() + escribir_a_db()
-- Leído por:  api/main.py → /history, /trends, /seasonality, /recommendations
--             dashboard/app.py → cargar_serie()
-- Nota: A pesar del nombre "semanal", en v4 es agregación DIARIA.
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS serie_semanal (
    id                  SERIAL PRIMARY KEY,
    fecha               DATE NOT NULL,
    unidades            INTEGER DEFAULT 0,
    ingreso_bruto       NUMERIC(12,2) DEFAULT 0,
    num_transacciones   INTEGER DEFAULT 0,
    productos_unicos    INTEGER DEFAULT 0,
    dia_semana          SMALLINT,
    dia_nombre          VARCHAR(20),
    semana_iso          SMALLINT,
    mes                 SMALLINT,
    anio                SMALLINT,
    es_fin_semana       SMALLINT DEFAULT 0,
    ingreso_acumulado   NUMERIC(14,2) DEFAULT 0,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_serie_fecha ON serie_semanal(fecha);


-- ══════════════════════════════════════════════════════════════════════════
-- 3. FEATURES — Features para modelos ML
-- ══════════════════════════════════════════════════════════════════════════
-- Escrito por: etl_pipeline.py → generar_features()
-- Leído por:  models/train_models.py → entrenar_random_forest()
-- Incluye: lags (1,7,14,21,28), rolling (7,14,30), momentum,
--          índices H&M, encoding cíclico, tendencia
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS features (
    id                   SERIAL PRIMARY KEY,
    fecha                DATE NOT NULL,
    unidades             INTEGER DEFAULT 0,
    dia_semana           SMALLINT,
    mes                  SMALLINT,
    es_fin_semana        SMALLINT DEFAULT 0,
    -- Lags
    lag_1                NUMERIC(10,2) DEFAULT 0,
    lag_7                NUMERIC(10,2) DEFAULT 0,
    lag_14               NUMERIC(10,2) DEFAULT 0,
    lag_21               NUMERIC(10,2) DEFAULT 0,
    lag_28               NUMERIC(10,2) DEFAULT 0,
    -- Rolling means
    rolling_mean_7       NUMERIC(10,4) DEFAULT 0,
    rolling_mean_14      NUMERIC(10,4) DEFAULT 0,
    rolling_mean_30      NUMERIC(10,4) DEFAULT 0,
    -- Rolling std
    rolling_std_7        NUMERIC(10,4) DEFAULT 0,
    rolling_std_14       NUMERIC(10,4) DEFAULT 0,
    rolling_std_30       NUMERIC(10,4) DEFAULT 0,
    -- Rolling max/min
    rolling_max_7        NUMERIC(10,2) DEFAULT 0,
    rolling_min_7        NUMERIC(10,2) DEFAULT 0,
    -- Cambio y momentum
    cambio_semanal_pct   NUMERIC(10,4) DEFAULT 0,
    momentum_7d          NUMERIC(10,4) DEFAULT 0,
    -- Índices H&M
    hm_indice_semanal    NUMERIC(6,4) DEFAULT 1,
    hm_indice_mensual    NUMERIC(6,4) DEFAULT 1,
    hm_correccion_dtf    NUMERIC(6,4) DEFAULT 1,
    hm_indice_combinado  NUMERIC(8,4) DEFAULT 1,
    -- Encoding cíclico
    dia_sin              NUMERIC(8,6) DEFAULT 0,
    dia_cos              NUMERIC(8,6) DEFAULT 0,
    mes_sin              NUMERIC(8,6) DEFAULT 0,
    mes_cos              NUMERIC(8,6) DEFAULT 0,
    -- Tendencia
    dias_desde_inicio    INTEGER DEFAULT 0,
    tendencia_norm       NUMERIC(8,6) DEFAULT 0,
    --
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_features_fecha ON features(fecha);


-- ══════════════════════════════════════════════════════════════════════════
-- 4. PREDICCIONES — Pronósticos generados por los modelos
-- ══════════════════════════════════════════════════════════════════════════
-- Escrito por: models/train_models.py → guardar_resultados()
-- Leído por:  api/main.py → /predict, /recommendations
--             dashboard/app.py → cargar_predicciones()
-- JOIN clave: predicciones.run_id = training_runs.run_id
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS predicciones (
    id                  SERIAL PRIMARY KEY,
    run_id              VARCHAR(20) NOT NULL,
    modelo              VARCHAR(50) NOT NULL,
    fecha_prediccion    DATE NOT NULL,
    unidades_predichas  NUMERIC(10,2),
    banda_inferior      NUMERIC(10,2),
    banda_superior      NUMERIC(10,2),
    dia_horizonte       INTEGER,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_run ON predicciones(run_id, modelo);
CREATE INDEX IF NOT EXISTS idx_pred_fecha ON predicciones(fecha_prediccion);


-- ══════════════════════════════════════════════════════════════════════════
-- 5. METRICAS_MODELOS — Comparación de rendimiento
-- ══════════════════════════════════════════════════════════════════════════
-- Escrito por: models/train_models.py → guardar_resultados()
-- Leído por:  api/main.py → /metrics
--             dashboard/app.py → cargar_metricas()
-- JOIN clave: metricas_modelos.run_id = training_runs.run_id
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS metricas_modelos (
    id          SERIAL PRIMARY KEY,
    run_id      VARCHAR(20) NOT NULL,
    modelo      VARCHAR(50) NOT NULL,
    mae         NUMERIC(12,4),
    mape        NUMERIC(8,2),
    r2          NUMERIC(8,4),
    parametros  TEXT,
    es_ganador  BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metricas_run ON metricas_modelos(run_id);


-- ══════════════════════════════════════════════════════════════════════════
-- 6. TRAINING_RUNS — Log de cada entrenamiento ejecutado
-- ══════════════════════════════════════════════════════════════════════════
-- Escrito por: models/train_models.py → guardar_resultados()
-- Leído por:  api/main.py → /metrics, /predict, /recommendations, /runs
--             dashboard/app.py → cargar_metricas()
-- Clave de unión: run_id (VARCHAR, UUID corto generado por train_models)
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS training_runs (
    id                SERIAL PRIMARY KEY,
    run_id            VARCHAR(20) NOT NULL UNIQUE,
    fecha_ejecucion   TIMESTAMP NOT NULL,
    modelo_ganador    VARCHAR(50),
    n_datos           INTEGER,
    horizonte         INTEGER DEFAULT 30,
    baseline_mae      NUMERIC(12,4),
    baseline_mape     NUMERIC(8,2),
    mejor_mae         NUMERIC(12,4),
    mejor_mape        NUMERIC(8,2),
    mejora_pct        NUMERIC(8,2),
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_fecha ON training_runs(fecha_ejecucion);
CREATE INDEX IF NOT EXISTS idx_runs_runid ON training_runs(run_id);


-- ══════════════════════════════════════════════════════════════════════════
-- 7. FACTORES_HM — Índices estacionales H&M + correcciones DTF
-- ══════════════════════════════════════════════════════════════════════════
-- Escrito por: etl_pipeline.py → escribir_a_db() (3 tipos: semanal, mensual, correccion_dtf)
-- Leído por:  api/main.py → /trends
--             dashboard/app.py → cargar_factores_hm()
-- Nota: En v4, los factores se manejan con (tipo, clave) en lugar de solo (mes).
--       tipo = 'semanal' | 'mensual' | 'correccion_dtf'
--       clave = '0'-'6' (días) o '1'-'12' (meses)
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS factores_hm (
    id          SERIAL PRIMARY KEY,
    tipo        VARCHAR(30) NOT NULL,
    clave       VARCHAR(10) NOT NULL,
    valor       NUMERIC(8,4) NOT NULL,
    descripcion TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_factores_tipo ON factores_hm(tipo);