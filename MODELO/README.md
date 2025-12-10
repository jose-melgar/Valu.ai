# MODELO — Entrenamiento compacto

Este directorio contiene un entrenamiento compacto y funcional para el notebook `Entrenamiento.ipynb`.

Contenido:
- `training.py` — script principal para entrenar modelos (Linear, RandomForest, XGBoost si está instalado).
- `model_io.py` — utilitarios mínimos (guardar/cargar modelos, extraer importancias).
- `artifacts/` — (se crea al ejecutar) donde se almacenan modelos `.joblib`, `training_summary.csv`, `feature_importances.csv` y `training_metadata.json`.

Requisitos
- Python 3.8+
- pandas, numpy, scikit-learn, joblib
- opcional: xgboost (si quieres entrenarlo)

Instalación rápida (ejemplo con pip):
```
pip install pandas numpy scikit-learn joblib
# opcional:
pip install xgboost
```

Uso
Desde la raíz del repo ejecuta:

```
python MODELO/training.py --data path/to/venta_ready.csv --out MODELO/artifacts
```

Parámetros:
- `--data` : path al CSV (ej: `data/venta_ready.csv` o `venta_ready.csv`)
- `--out`  : directorio donde guardar artefactos (por defecto `MODELO/artifacts`)
- `--test-size` : fracción de test (por defecto 0.2)
- `--random-state` : semilla para reproducibilidad (por defecto 42)

Salida importante:
- `MODELO/artifacts/training_summary.csv` — métricas por modelo (mae_log, rmse_log, mae_soles)
- `MODELO/artifacts/*.joblib` — modelos guardados
- `MODELO/artifacts/feature_importances.csv` — si el modelo ganador tiene importancias
- `MODELO/artifacts/training_metadata.json` — metadata de la corrida

Notas y recomendaciones rápidas
- Antes de ejecutar, asegúrate de que `venta_ready.csv` contenga exactamente las columnas listadas en el script (`FINAL_FEATURES + ['log_precio']`).
- Si quieres menos archivos, puedes ignorar `model_io.py` y usar joblib directamente en `training.py`, pero separarlo mantiene claridad.
- Si la columna `log_superficie` no existe en tu CSV, añadela antes o ajusta `FINAL_FEATURES` en `training.py`.

Si quieres, puedo:
- Adaptar `training.py` para admitir pipelines de preprocessing (StandardScaler, OneHot) y guardar el pipeline junto al modelo (recomendado para producción).
- Generar una versión que use un único modelo "campeón" preseleccionado y que exponga una API REST ligera (Flask/FastAPI) para inferencia.