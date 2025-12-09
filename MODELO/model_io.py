"""
MODELO/model_io.py
Funciones simples para guardar/cargar modelos y extraer importancias mínimas.
Usamos joblib para persistencia.
"""
from pathlib import Path
from typing import Optional, List
import joblib
import pandas as pd


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path):
    return joblib.load(path)


def get_feature_importance_df(model, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Devuelve DataFrame con columnas ['feature','importance'] si el modelo tiene:
     - feature_importances_ (tree-based)
     - coef_ (linear)
    En otro caso devuelve None.
    """
    import numpy as np

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False)
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef)})
        return df.sort_values("importance", ascending=False)
    return None