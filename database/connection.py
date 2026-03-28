"""
Database Connection v4.0 — DTF Fashion Predictive Analytics Platform

Módulo de conexión a base de datos con PostgreSQL como motor principal
y fallback a SQLite para desarrollo local sin Docker.

Exporta 3 funciones usadas por todo el proyecto:
  - engine()          → retorna el SQLAlchemy engine (singleton)
  - read_sql(query)   → ejecuta una query SELECT y retorna DataFrame
  - write_dataframe() → escribe un DataFrame a una tabla

Variables de entorno:
  DATABASE_URL  →  postgresql://user:pass@host:5432/dbname
                   Si no está definida, usa SQLite en data/dtf_fashion.db

Usado por:
  - etl/etl_pipeline.py     → engine(), write_dataframe()
  - models/train_models.py  → engine(), read_sql(), write_dataframe()
  - api/main.py             → read_sql(), engine()
  - dashboard/app.py        → read_sql(), engine()
"""

import os
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

log = logging.getLogger("database")

# ── Singleton ──
_engine = None


def _get_database_url() -> str:
    """
    Obtiene la URL de conexión.
    Prioridad:
      1. Variable de entorno DATABASE_URL (Docker / producción)
      2. Fallback a SQLite local (desarrollo sin Docker)
    """
    url = os.getenv("DATABASE_URL")
    if url:
        log.info(f"Usando DATABASE_URL: {url.split('@')[0]}@***")
        return url

    # Fallback: SQLite en la carpeta data/
    db_path = Path(__file__).resolve().parent.parent / "data" / "dtf_fashion.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log.warning(f"DATABASE_URL no definida — fallback a SQLite: {db_path}")
    return f"sqlite:///{db_path}"


def engine():
    """
    Retorna el SQLAlchemy engine (singleton).

    - PostgreSQL: pool de 5 conexiones con pre-ping
    - SQLite: StaticPool (una sola conexión, thread-safe)

    En SQLite, inicializa las tablas automáticamente desde schema.sql.
    """
    global _engine
    if _engine is not None:
        return _engine

    url = _get_database_url()
    is_sqlite = url.startswith("sqlite")

    if is_sqlite:
        _engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
        _init_sqlite_tables(_engine)
    else:
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )

    db_type = "SQLite" if is_sqlite else "PostgreSQL"
    log.info(f"Engine creado: {db_type}")
    return _engine


def _init_sqlite_tables(eng):
    """
    Crea las tablas en SQLite usando schema.sql.
    Adapta la sintaxis de PostgreSQL a SQLite al vuelo.
    Solo se ejecuta si DATABASE_URL no está definida.
    """
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    if not schema_path.exists():
        log.warning("schema.sql no encontrado — tablas no inicializadas")
        return

    sql_raw = schema_path.read_text(encoding="utf-8")

    # ── Adaptar PostgreSQL → SQLite ──
    replacements = [
        ("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("SMALLINT", "INTEGER"),
        ("BOOLEAN", "INTEGER"),
        ("DEFAULT CURRENT_TIMESTAMP", "DEFAULT (datetime('now'))"),
        ("TIMESTAMP", "TEXT"),
    ]
    sql = sql_raw
    for old, new in replacements:
        sql = sql.replace(old, new)

    # Eliminar precisión numérica (SQLite la ignora pero puede romper el parse)
    import re
    sql = re.sub(r"NUMERIC\(\d+,\d+\)", "REAL", sql)
    sql = re.sub(r"VARCHAR\(\d+\)", "TEXT", sql)

    # Ejecutar cada statement por separado
    with eng.begin() as conn:
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt and not stmt.startswith("--"):
                try:
                    conn.execute(text(stmt))
                except Exception as e:
                    # Ignorar errores de "ya existe" u otros no-críticos
                    err = str(e).lower()
                    if "already exists" not in err and "duplicate" not in err:
                        log.debug(f"SQLite init skip: {e}")

    log.info("SQLite: tablas inicializadas desde schema.sql")


def read_sql(query: str) -> pd.DataFrame:
    """
    Ejecuta una query SELECT y retorna un pandas DataFrame.

    Args:
        query: SQL string (puede usar text() internamente)

    Returns:
        pd.DataFrame con los resultados

    Raises:
        Exception si la conexión falla o la query es inválida

    Ejemplo:
        df = read_sql("SELECT * FROM ventas ORDER BY fecha")
        df = read_sql("SELECT COUNT(*) as n FROM serie_semanal")
    """
    eng = engine()
    with eng.connect() as conn:
        return pd.read_sql(text(query), conn)


def write_dataframe(
    df: pd.DataFrame,
    table: str,
    if_exists: str = "append",
) -> int:
    """
    Escribe un DataFrame a una tabla de la base de datos.

    Args:
        df: DataFrame a escribir
        table: Nombre de la tabla destino
        if_exists: 'append' (default), 'replace', o 'fail'

    Returns:
        Número de filas escritas

    Ejemplo:
        write_dataframe(df_ventas, "ventas", if_exists="append")
    """
    eng = engine()
    rows = df.to_sql(table, eng, if_exists=if_exists, index=False)
    return rows if rows is not None else len(df)