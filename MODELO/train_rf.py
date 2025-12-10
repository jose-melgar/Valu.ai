#!/usr/bin/env python3
"""
MODELO/train_rf.py

Preprocesamiento + entrenamiento RandomForest siguiendo las reglas del repo/usuario.

Uso (desde la raíz del repo):
python MODELO/train_rf.py --data Data/venta_ready.csv --out artifacts --random-state 42

Salida:
- artifacts/model.joblib                -> modelo RandomForest + metadata (dict)
- artifacts/preprocessor.joblib         -> diccionario con info de preprocesamiento (cols y dummies)
- artifacts/training_summary.csv        -> métricas (MAE log, RMSE log, MAE soles antes/después corrección)
- artifacts/feature_importances.csv     -> importancias de features
- artifacts/training_metadata.json      -> metadatos de la corrida
"""
import os
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def safe_col_choice(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def preprocess(df):
    # columnas flexibles
    col_precio_usd = safe_col_choice(df, ['precio_usd', 'precio_usd'])
    col_tipo_cambio = safe_col_choice(df, ['tipo_cambio', 'tipo_cambio'])
    col_ipc = safe_col_choice(df, ['ipc', 'ipc'])
    col_superficie = safe_col_choice(df, ['superficie_total', 'superficie', 'superficie_m2', 'superficie_total_m2'])
    col_precio_m2_usd = safe_col_choice(df, ['precio_m2_usd', 'precio_m2'])

    if col_precio_usd is None or col_tipo_cambio is None or col_ipc is None or col_superficie is None:
        missing = []
        for name, col in [('precio_usd', col_precio_usd), ('tipo_cambio', col_tipo_cambio),
                          ('ipc', col_ipc), ('superficie_total', col_superficie)]:
            if col is None:
                missing.append(name)
        raise RuntimeError(f"Faltan columnas requeridas en el CSV: {missing}")

    # Filtrado de outliers (reglas de negocio)
    if col_precio_m2_usd is not None:
        df = df[(df[col_precio_m2_usd] >= 500) & (df[col_precio_m2_usd] <= 6000)]
    df = df[(df[col_superficie] >= 30) & (df[col_superficie] <= 500)]

    # Deflactar a Soles constantes 2009 (IPC base = 77.45)
    df['precio_soles_corrientes'] = df[col_precio_usd] * df[col_tipo_cambio]
    df['factor_deflacion'] = df[col_ipc] / 77.45
    df['precio_target'] = df['precio_soles_corrientes'] / df['factor_deflacion']

    # Target logarítmico
    df['log_precio'] = np.log1p(df['precio_target'])

    # log_superficie
    df['log_superficie'] = np.log1p(df[col_superficie])

    # Imputaciones
    if 'garajes' in df.columns:
        df['garajes'] = df['garajes'].fillna(0)
    else:
        df['garajes'] = 0

    if 'banos' in df.columns:
        median_banos = int(df['banos'].median()) if not df['banos'].dropna().empty else 1
        df['banos'] = df['banos'].fillna(median_banos)
    else:
        df['banos'] = 1

    # habitaciones y antiguedad
    if 'habitaciones' not in df.columns:
        df['habitaciones'] = 0
    if 'antiguedad' not in df.columns:
        df['antiguedad'] = 0

    # Dummies de distrito:
    # Si ya existen columnas 'distrito_' las preservamos; si existe columna 'distrito' la codificamos
    existing_distrito_cols = [c for c in df.columns if c.startswith('distrito_')]
    if existing_distrito_cols:
        distrito_cols = existing_distrito_cols
    elif 'distrito' in df.columns:
        dummies = pd.get_dummies(df['distrito'], prefix='distrito', drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        distrito_cols = [c for c in df.columns if c.startswith('distrito_')]
    else:
        distrito_cols = []

    # Variables a excluir por data leakage
    for drop_col in ['precio_usd', 'precio_m2_usd', 'precio_m2', 'ipc', 'tipo_cambio', 'anio']:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    # Características finales en orden definido
    features = ['log_superficie', 'banos', 'garajes', 'antiguedad', 'habitaciones'] + distrito_cols
    # Filtramos features que no existan (por si no hay dummies)
    features = [c for c in features if c in df.columns]

    return df, features

def evaluate_and_save(model, X_test, y_test, artifacts_dir, preproc):
    # Predicción en espacio log
    y_pred_log = model.predict(X_test)

    # Métricas en log
    mae_log = mean_absolute_error(y_test, y_pred_log)
    # Compatibilidad con versiones de sklearn que no aceptan 'squared' -> calcular RMSE explícitamente
    rmse_log = float(np.sqrt(mean_squared_error(y_test, y_pred_log)))

    # Convertir a escala soles (Soles constantes 2009)
    y_test_soles = np.expm1(y_test)
    y_pred_soles = np.expm1(y_pred_log)
    mae_soles = mean_absolute_error(y_test_soles, y_pred_soles)

    # Corrección de sesgo (ME en espacio log)
    bias = np.mean(y_test - y_pred_log)
    y_pred_log_bc = y_pred_log + bias
    y_pred_soles_bc = np.expm1(y_pred_log_bc)
    mae_soles_bc = mean_absolute_error(y_test_soles, y_pred_soles_bc)
    mae_log_bc = mean_absolute_error(y_test, y_pred_log_bc)

    # Guardar resumen de métricas
    summary = {
        'mae_log': float(mae_log),
        'rmse_log': float(rmse_log),
        'mae_soles_before_bias': float(mae_soles),
        'mae_log_after_bias': float(mae_log_bc),
        'mae_soles_after_bias': float(mae_soles_bc),
        'bias_log': float(bias)
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(artifacts_dir, 'training_summary.csv'), index=False)

    # Guardar metadatos
    metadata = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'n_test': int(len(y_test)),
        'summary': summary,
        'preprocessor': preproc
    }
    with open(os.path.join(artifacts_dir, 'training_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return summary, bias

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    print(f"Dataset cargado: {args.data} -> shape {df.shape}")

    df_proc, features = preprocess(df)
    print(f"Después preproc shape: {df_proc.shape}. Features usadas: {len(features)}")

    # Target y X
    if 'log_precio' not in df_proc.columns:
        raise RuntimeError("Después del preprocesamiento falta 'log_precio' en el dataframe.")
    X = df_proc[features].copy()
    y = df_proc['log_precio'].values

    # Train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Entrenar RandomForest con hiperparámetros dados
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=30,
        max_features='sqrt',
        min_samples_leaf=1,
        random_state=args.random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Importancias
    try:
        importances = rf.feature_importances_
        fi = pd.DataFrame({'feature': X.columns, 'importance': importances})
        fi = fi.sort_values('importance', ascending=False)
        fi.to_csv(os.path.join(out_dir, 'feature_importances.csv'), index=False)
    except Exception:
        fi = None

    # Evaluar y aplicar correction
    preproc = {
        'features': features,
        'distrito_cols': [c for c in features if c.startswith('distrito_')],
        'imputations': {'garajes': 0, 'banos': 'median'},
        'outlier_rules': {'precio_m2_usd': [500, 6000], 'superficie_total': [30, 500]},
        'ipc_base': 77.45
    }

    summary, bias = evaluate_and_save(rf, X_test, y_test, str(out_dir), preproc)

    # Guardar modelo y preproc
    model_bundle = {
        'model': rf,
        'features': features,
        'bias_log': float(bias),
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'hyperparameters': {
            'n_estimators': 300,
            'max_depth': 30,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'random_state': args.random_state
        }
    }
    joblib.dump(model_bundle, os.path.join(out_dir, 'model.joblib'))
    joblib.dump(preproc, os.path.join(out_dir, 'preprocessor.joblib'))

    print("Entrenamiento finalizado. Artefactos guardados en:", out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrena RandomForest para Valu.ai')
    parser.add_argument('--data', required=True, help='Path al CSV (ej: Data/venta_ready.csv)')
    parser.add_argument('--out', default='artifacts', help='Directorio de salida para artefactos')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fracción de test')
    parser.add_argument('--random-state', type=int, default=42, help='Semilla')
    args = parser.parse_args()
    main(args)