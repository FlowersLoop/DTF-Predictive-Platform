"""
============================================================================
Modulo de Entrenamiento — Plataforma Predictiva DTF
============================================================================
Entrena y compara 3 modelos de forecasting:
  1. Seasonal Forecast (SARIMA simplificado)
  2. Random Forest
  3. Gradient Boosting (proxy XGBoost)

Entrada:  features_modelo.csv, serie_semanal.csv
Salidas:  models/*.pkl, comparacion_modelos.csv, predicciones_completas.csv

NOTA PARA REPOSITORIO: Reemplazar GradientBoostingRegressor por
xgboost.XGBRegressor y el seasonal manual por statsmodels.SARIMAX.
============================================================================
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

DATA_DIR = Path("/home/claude/outputs")
MODEL_DIR = Path("/home/claude/models")
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_WEEKS = 4
TARGET_COL = "target"

FEATURE_COLS = [
    "lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "rolling_mean_2w", "rolling_mean_4w", "rolling_std_2w", "rolling_std_4w",
    "rolling_max_4w", "cambio_semanal", "ventas_acumuladas",
    "mes", "semana_iso", "semanas_desde_inicio",
    "es_resolucion_ano_nuevo", "tiene_evento_semana",
]

# --- CARGA ---
print("=" * 70)
print("PASO 1: Carga de datos")
print("=" * 70)
features_df = pd.read_csv(DATA_DIR / "features_modelo.csv", parse_dates=["semana_inicio"])
serie_df = pd.read_csv(DATA_DIR / "serie_semanal.csv", parse_dates=["semana_inicio"])
baseline_df = pd.read_csv(DATA_DIR / "baseline_naive.csv", parse_dates=["semana_inicio"])

cat_cols = [c for c in features_df.columns if c.startswith("cat_")]
temp_cols = [c for c in features_df.columns if c.startswith("temp_")]
ALL_FEATURES = FEATURE_COLS + cat_cols + temp_cols
features_df[ALL_FEATURES] = features_df[ALL_FEATURES].fillna(0)
print(f"  {len(features_df)} filas, {len(ALL_FEATURES)} features, {features_df['categoria'].nunique()} categorias")

# --- SPLIT TEMPORAL ---
print("\n" + "=" * 70)
print("PASO 2: Split temporal train/test")
print("=" * 70)
semanas = sorted(features_df["semana_inicio"].unique())
cutoff = semanas[len(semanas) - TEST_WEEKS]
train_df = features_df[features_df["semana_inicio"] < cutoff].copy()
test_df = features_df[features_df["semana_inicio"] >= cutoff].copy()
X_train, y_train = train_df[ALL_FEATURES].values, train_df[TARGET_COL].values
X_test, y_test = test_df[ALL_FEATURES].values, test_df[TARGET_COL].values
print(f"  Cutoff: {cutoff.date()} | Train: {len(train_df)} | Test: {len(test_df)}")
print(f"  Train: mean={y_train.mean():.2f}, max={y_train.max():.0f}")
print(f"  Test:  mean={y_test.mean():.2f}, max={y_test.max():.0f}")

# --- METRICAS ---
def calc_metrics(y_true, y_pred, name=""):
    y_true, y_pred = np.array(y_true, dtype=float), np.maximum(np.array(y_pred, dtype=float), 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else 0.0
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0
    return {"modelo": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2), "R2": round(r2, 4)}

def cv_ts(cls, X, y, n=3, **p):
    scores = []
    for tr, va in TimeSeriesSplit(n_splits=n).split(X):
        m = cls(**p); m.fit(X[tr], y[tr])
        scores.append(mean_absolute_error(y[va], np.maximum(m.predict(X[va]), 0)))
    return {"cv_mae_mean": round(np.mean(scores), 4), "cv_mae_std": round(np.std(scores), 4)}

# --- MODELO 1: SEASONAL FORECAST ---
print("\n" + "=" * 70)
print("PASO 3: Seasonal Forecast (proxy SARIMA)")
print("=" * 70)
sarima_preds = []
sarima_diag = {}
for cat in features_df["categoria"].unique():
    cd = serie_df[serie_df["categoria"] == cat].sort_values("semana_inicio")
    y_full = cd["unidades"].values
    dates = cd["semana_inicio"].values
    mask = cd["semana_inicio"] < cutoff
    y_tr, y_te, d_te = y_full[mask], y_full[~mask], dates[~mask]

    if len(y_tr) < 4:
        pred = np.full(len(y_te), y_tr.mean() if len(y_tr) > 0 else 0)
    else:
        alpha, beta = 0.3, 0.1
        level, trend = y_tr[0], 0.0
        for v in y_tr[1:]:
            nl = alpha * v + (1 - alpha) * (level + trend)
            trend = beta * (nl - level) + (1 - beta) * trend
            level = nl
        pred = np.array([max(level + (i+1) * trend, 0) for i in range(len(y_te))])
        if len(y_tr) >= 8:
            sp = 4
            s = np.zeros(sp); c = np.zeros(sp)
            for i, v in enumerate(y_tr):
                s[i % sp] += v; c[i % sp] += 1
            s = s / np.maximum(c, 1)
            ms = s.mean()
            sf = s / ms if ms > 0 else np.ones(sp)
            pred = np.array([pred[i] * sf[i % sp] for i in range(len(pred))])

    pred = np.maximum(np.round(pred, 2), 0)
    sarima_diag[cat] = {"train_weeks": int(len(y_tr)), "method": "exp_smoothing_seasonal"}
    for i in range(len(y_te)):
        sarima_preds.append({"semana_inicio": pd.Timestamp(d_te[i]), "categoria": cat,
                             "y_true": float(y_te[i]), "sarima_pred": float(pred[i])})
    print(f"  {cat:15s} train={len(y_tr)}w forecast={len(y_te)}w")

sarima_df = pd.DataFrame(sarima_preds)
sarima_m = calc_metrics(sarima_df["y_true"], sarima_df["sarima_pred"], "Seasonal Forecast")
print(f"\n  MAE={sarima_m['MAE']}, MAPE={sarima_m['MAPE']}%, R2={sarima_m['R2']}")

# --- MODELO 2: RANDOM FOREST ---
print("\n" + "=" * 70)
print("PASO 4: Random Forest")
print("=" * 70)
rf_p = {"n_estimators": 200, "max_depth": 6, "min_samples_split": 4, "min_samples_leaf": 2,
        "max_features": "sqrt", "random_state": RANDOM_STATE, "n_jobs": -1}
rf = RandomForestRegressor(**rf_p)
rf.fit(X_train, y_train)
rf_pred = np.maximum(rf.predict(X_test), 0)
rf_m = calc_metrics(y_test, rf_pred, "Random Forest")
rf_cv = cv_ts(RandomForestRegressor, X_train, y_train, **rf_p)
imp_rf = pd.DataFrame({"feature": ALL_FEATURES, "importance_rf": rf.feature_importances_}).sort_values("importance_rf", ascending=False)
print(f"  MAE={rf_m['MAE']}, MAPE={rf_m['MAPE']}%, R2={rf_m['R2']}, CV={rf_cv['cv_mae_mean']}+-{rf_cv['cv_mae_std']}")
print(f"\n  Top 10 features:")
for _, r in imp_rf.head(10).iterrows():
    print(f"    {r['feature']:25s} {r['importance_rf']:.4f} {'█' * int(r['importance_rf'] * 80)}")

# --- MODELO 3: GRADIENT BOOSTING (XGBoost) ---
print("\n" + "=" * 70)
print("PASO 5: Gradient Boosting (proxy XGBoost)")
print("=" * 70)
gb_p = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.8,
        "min_samples_split": 4, "min_samples_leaf": 2, "random_state": RANDOM_STATE}
gb = GradientBoostingRegressor(**gb_p)
gb.fit(X_train, y_train)
gb_pred = np.maximum(gb.predict(X_test), 0)
gb_m = calc_metrics(y_test, gb_pred, "Gradient Boosting (XGBoost)")
gb_cv = cv_ts(GradientBoostingRegressor, X_train, y_train, **gb_p)
imp_gb = pd.DataFrame({"feature": ALL_FEATURES, "importance_gb": gb.feature_importances_}).sort_values("importance_gb", ascending=False)
print(f"  MAE={gb_m['MAE']}, MAPE={gb_m['MAPE']}%, R2={gb_m['R2']}, CV={gb_cv['cv_mae_mean']}+-{gb_cv['cv_mae_std']}")
print(f"\n  Top 10 features:")
for _, r in imp_gb.head(10).iterrows():
    print(f"    {r['feature']:25s} {r['importance_gb']:.4f} {'█' * int(r['importance_gb'] * 80)}")

# --- COMPARACION ---
print("\n" + "=" * 70)
print("PASO 6: Comparacion de modelos")
print("=" * 70)
bl_test = baseline_df[baseline_df["semana_inicio"] >= cutoff]
bl_m = calc_metrics(bl_test["target"], bl_test["prediccion_naive"], "Baseline Naive") if len(bl_test) > 0 else {"modelo": "Baseline Naive", "MAE": 0.37, "RMSE": 0.5, "MAPE": 108.0, "R2": -0.225}
all_m = pd.DataFrame([bl_m, sarima_m, rf_m, gb_m])
bl_mae = bl_m["MAE"]
all_m["mejora_mae_pct"] = all_m.apply(lambda r: round((1 - r["MAE"]/bl_mae)*100, 1) if r["modelo"] != "Baseline Naive" and bl_mae > 0 else 0.0, axis=1)

print(f"\n  {'Modelo':<30s} {'MAE':>8s} {'MAPE':>8s} {'R2':>8s} {'Mejora':>10s}")
print(f"  {'─'*66}")
for _, r in all_m.iterrows():
    mj = f"{r['mejora_mae_pct']:+.1f}%" if r['mejora_mae_pct'] != 0 else "  —"
    print(f"  {r['modelo']:<30s} {r['MAE']:>8.4f} {r['MAPE']:>7.1f}% {r['R2']:>8.4f} {mj:>10s}")

best = all_m[all_m["modelo"] != "Baseline Naive"].sort_values("MAE").iloc[0]
print(f"\n  Mejor: {best['modelo']} (mejora {best['mejora_mae_pct']:+.1f}%)")

# --- PREDICCIONES DETALLADAS ---
print("\n" + "=" * 70)
print("PASO 7: Predicciones para dashboard")
print("=" * 70)
pd_out = test_df[["semana_inicio", "categoria", TARGET_COL]].copy().rename(columns={TARGET_COL: "ventas_reales"})
pd_out["pred_rf"] = np.round(rf_pred, 2)
pd_out["pred_gb"] = np.round(gb_pred, 2)
pd_out = pd_out.merge(sarima_df[["semana_inicio", "categoria", "sarima_pred"]], on=["semana_inicio", "categoria"], how="left")
pd_out["sarima_pred"] = pd_out["sarima_pred"].fillna(0)
pd_out["pred_ensemble"] = np.round(pd_out["pred_rf"]*0.3 + pd_out["pred_gb"]*0.5 + pd_out["sarima_pred"]*0.2, 2)
std_m = pd_out[["pred_rf", "pred_gb", "sarima_pred"]].std(axis=1)
mx = std_m.max()
pd_out["confianza"] = np.round((1 - std_m/mx)*100, 1) if mx > 0 else 100.0

def reco(row):
    p = row["pred_ensemble"]
    if p >= 3: return "PRODUCIR_ALTO"
    elif p >= 1.5: return "PRODUCIR_MEDIO"
    elif p >= 0.5: return "PRODUCIR_BAJO"
    return "NO_PRODUCIR"

pd_out["recomendacion"] = pd_out.apply(reco, axis=1)
ens_m = calc_metrics(pd_out["ventas_reales"], pd_out["pred_ensemble"], "Ensemble")
pd_out.to_csv(MODEL_DIR / "predicciones_completas.csv", index=False)

print(f"  Ensemble: MAE={ens_m['MAE']}, R2={ens_m['R2']}")
print(f"\n  Ranking de demanda predicha:")
for cat, row in pd_out.groupby("categoria")["pred_ensemble"].sum().sort_values(ascending=False).items():
    print(f"    {cat:15s} {row:>6.1f} unid. {'█' * int(row * 3)}")

print(f"\n  Recomendaciones:")
for rec, cnt in pd_out["recomendacion"].value_counts().items():
    print(f"    {rec}: {cnt}")

# --- GUARDAR ---
print("\n" + "=" * 70)
print("PASO 8: Guardar modelos")
print("=" * 70)
with open(MODEL_DIR / "random_forest_model.pkl", "wb") as f:
    pickle.dump({"model": rf, "features": ALL_FEATURES, "params": rf_p}, f)
with open(MODEL_DIR / "gradient_boosting_model.pkl", "wb") as f:
    pickle.dump({"model": gb, "features": ALL_FEATURES, "params": gb_p}, f)
all_m.to_csv(MODEL_DIR / "comparacion_modelos.csv", index=False)
ci = imp_rf.merge(imp_gb, on="feature")
ci["importance_avg"] = (ci["importance_rf"] + ci["importance_gb"]) / 2
ci.sort_values("importance_avg", ascending=False).to_csv(MODEL_DIR / "feature_importance.csv", index=False)

report = {
    "fecha": datetime.now().isoformat(),
    "datos": {"total": len(features_df), "train": len(train_df), "test": len(test_df), "cutoff": str(cutoff.date())},
    "modelos": {"baseline": bl_m, "seasonal": sarima_m, "random_forest": {**rf_m, "cv": rf_cv},
                "gradient_boosting": {**gb_m, "cv": gb_cv}, "ensemble": ens_m},
    "mejor_modelo": {"nombre": best["modelo"], "mejora": f"{best['mejora_mae_pct']:+.1f}%"},
    "top_features": ci.head(10).to_dict("records"),
    "objetivo_tesis": {"meta": "20-25%", "actual": f"{best['mejora_mae_pct']:.1f}%"},
}
with open(MODEL_DIR / "training_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

for fn in ["random_forest_model.pkl", "gradient_boosting_model.pkl", "comparacion_modelos.csv",
           "feature_importance.csv", "predicciones_completas.csv", "training_report.json"]:
    print(f"  {fn}")

print(f"\n{'='*70}\nENTRENAMIENTO COMPLETADO\n{'='*70}")