#!/usr/bin/env python3
"""
MODELO/training.py
Entrenamiento compacto (CLI) que:
 - carga 'venta_ready.csv' (o el CSV que indiques)
 - construye X e y usando la lista de features del notebook
 - split train/test
 - entrena LinearRegression, RandomForest and XGBoost (si está instalado)
 - evalúa usando MAE en escala log y en soles (expm1)
 - guarda el mejor modelo en formato joblib y un CSV resumen de métricas

Uso:
python MODELO/training.py --data path/to/venta_ready.csv --out MODELO/artifacts --model auto

Recomendado: ejecutar en un entorno con scikit-learn, pandas, numpy y joblib.
XGBoost es opcional (se usa si está instalado).
"""
import os
import argparse
import logging
from pathlib import Path
import joblib
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Intentamos importar xgboost (opcional)
try:
    import xgboost as xgb  # type: ignore
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# Import helpers
from model_io import save_model, load_model, get_feature_importance_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------------------------
# CONFIG: características finales (copiado del notebook Entrenamiento.ipynb)
# ---------------------------
FINAL_FEATURES = [
    'antiguedad', 'banos', 'distrito_Barranco',
    'distrito_Bellavista', 'distrito_Breña', 'distrito_Carabayllo',
    'distrito_Cercado de Lima', 'distrito_Chorrillos', 'distrito_Comas',
    'distrito_Jesús María', 'distrito_La Molina', 'distrito_La Perla',
    'distrito_La Victoria', 'distrito_Lince', 'distrito_Los Olivos',
    'distrito_Magdalena', 'distrito_Miraflores', 'distrito_Pueblo Libre',
    'distrito_San Borja', 'distrito_San Isidro', 'distrito_San Miguel',
    'distrito_Surco', 'distrito_Surquillo', 'garajes',
    'habitaciones', 'log_superficie', 'piso',
    'vista_exterior'
]
TARGET = 'log_precio'  # nombre de la columna target


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    logging.info("Dataset cargado: %s (shape=%s)", csv_path, df.shape)
    missing = [c for c in FINAL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Faltan columnas requeridas en CSV: {missing}")
    return df


def evaluate_model(model, X_test, y_test):
    """Devuelve métricas en escala log y en soles (revirtiendo np.expm1)."""
    y_pred_log = model.predict(X_test)
    mae_log = mean_absolute_error(y_test, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))

    # Revertir log para obtener MAE en monedas (soles)
    y_test_soles = np.expm1(y_test)
    y_pred_soles = np.expm1(y_pred_log)
    mae_soles = mean_absolute_error(y_test_soles, y_pred_soles)

    return {
        "mae_log": float(mae_log),
        "rmse_log": float(rmse_log),
        "mae_soles": float(mae_soles)
    }


def train_and_evaluate(df: pd.DataFrame, out_dir: Path, random_state: int = 42, test_size: float = 0.2):
    X = df[FINAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logging.info("Split hecho: X_train=%s X_test=%s", X_train.shape, X_test.shape)

    results = []

    # 1) Linear Regression
    logging.info("Entrenando LinearRegression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    metrics_lr = evaluate_model(lr, X_test, y_test)
    logging.info("Linear MAE(soles)=%.2f", metrics_lr["mae_soles"])
    out_lr = out_dir / "linear_regression.joblib"
    save_model(lr, out_lr)
    results.append({"model": "linear", "path": str(out_lr), **metrics_lr})

    # 2) Random Forest
    logging.info("Entrenando RandomForestRegressor (n_jobs=-1)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    metrics_rf = evaluate_model(rf, X_test, y_test)
    logging.info("RandomForest MAE(soles)=%.2f", metrics_rf["mae_soles"])
    out_rf = out_dir / "random_forest.joblib"
    save_model(rf, out_rf)
    results.append({"model": "random_forest", "path": str(out_rf), **metrics_rf})

    # 3) XGBoost (si está disponible)
    if HAVE_XGB:
        logging.info("Entrenando XGBoost (XGBRegressor)...")
        xgbr = xgb.XGBRegressor(objective='reg:squarederror',
                                n_estimators=100,
                                learning_rate=0.1,
                                random_state=random_state,
                                n_jobs=-1)
        xgbr.fit(X_train, y_train)
        metrics_xgb = evaluate_model(xgbr, X_test, y_test)
        logging.info("XGBoost MAE(soles)=%.2f", metrics_xgb["mae_soles"])
        out_xgb = out_dir / "xgboost.joblib"
        save_model(xgbr, out_xgb)
        results.append({"model": "xgboost", "path": str(out_xgb), **metrics_xgb})
    else:
        logging.warning("xgboost no está instalado. Omitiendo XGBoost.")

    # Guardar resultados resumen
    results_df = pd.DataFrame(results).sort_values("mae_soles")
    summary_csv = out_dir / "training_summary.csv"
    results_df.to_csv(summary_csv, index=False)
    logging.info("Resumen guardado en: %s", summary_csv)

    # Guardar importancias del mejor modelo si aplica
    best = results_df.iloc[0]
    best_path = Path(best["path"])
    best_model = load_model(best_path)
    fi_df = get_feature_importance_df(best_model, FINAL_FEATURES)
    if fi_df is not None:
        fi_csv = out_dir / "feature_importances.csv"
        fi_df.to_csv(fi_csv, index=False)
        logging.info("Feature importances guardadas en: %s", fi_csv)
    else:
        logging.info("El mejor modelo no tiene feature_importances_ ni coef_. (p.ej. no es tree o linear)")

    logging.info("Mejor modelo: %s (MAE(soles)=%.2f)", best["model"], best["mae_soles"])
    return results_df


def main():
    parser = argparse.ArgumentParser(prog="training.py", description="Entrena modelos y guarda artefactos")
    parser.add_argument("--data", required=True, help="CSV de entrada (venta_ready.csv)")
    parser.add_argument("--out", default="MODELO/artifacts", help="Directorio donde se guardan modelos y reportes")
    parser.add_argument("--test-size", type=float, default=0.2, help="Porcentaje de test (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Salida: %s", out_dir)

    df = load_data(args.data)
    results_df = train_and_evaluate(df, out_dir, random_state=args.random_state, test_size=args.test_size)

    # Guardar metadata de la corrida
    meta = {
        "n_rows": int(df.shape[0]),
        "n_features": len(FINAL_FEATURES),
        "features": FINAL_FEATURES,
        "target": TARGET,
        "results": results_df.to_dict(orient="records"),
    }
    with open(out_dir / "training_metadata.json", "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logging.info("Metadata guardada.")


if __name__ == "__main__":
    main()