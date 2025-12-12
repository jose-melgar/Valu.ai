#!/usr/bin/env python3
"""
Frontend/gui_tkinter.py

GUI en Tkinter para cargar el modelo guardado (artifacts/model.joblib)
y realizar predicciones de precios actuales en Soles y Dólares.

Cambios importantes respecto a la versión anterior:
- Carga Data/venta_ready.csv al inicio y calcula FACTOR_SOLES y FACTOR_USD
  usando el año más reciente (max(anio)). Si el CSV no existe, usa valores
  de fallback y muestra una advertencia en consola.
- En la salida al usuario muestra SÓLO:
    - Precio estimado en Dólares actuales ($ XX,XXX.XX)
    - Precio estimado en Soles actuales (S/ XX,XXX.XX)
  con formato y fuente más grande/negrita.
- Internamente se sigue soportando la corrección de sesgo (checkbox),
  pero no se muestra información técnica (log space, bias, Soles 2009).
"""
import os
import sys
import math
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont

# Rutas por defecto que buscará la GUI
DEFAULT_MODEL_PATHS = [
    os.path.join("artifacts", "model.joblib"),
    os.path.join("MODELO", "artifacts", "model.joblib"),
    os.path.join("MODELO", "model.joblib"),
    "model.joblib",
]

DEFAULT_PREPROC_PATHS = [
    os.path.join("artifacts", "preprocessor.joblib"),
    os.path.join("MODELO", "artifacts", "preprocessor.joblib"),
    os.path.join("MODELO", "preprocessor.joblib"),
    "preprocessor.joblib",
]

VENTA_READY_PATH = os.path.join("Data", "venta_ready.csv")

# Fallback factors si no se encuentra el CSV o no se pueden calcular
FALLBACK_FACTOR_SOLES = 1.65
FALLBACK_FACTOR_USD = 0.45


def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def compute_conversion_factors(venta_path=VENTA_READY_PATH):
    """
    Intenta cargar venta_ready.csv y calcular FACTOR_SOLES y FACTOR_USD
    basados en el año más reciente (max(anio)):

    FACTOR_SOLES = mean(precio_soles_corrientes / precio_target)
    FACTOR_USD = mean(precio_usd / precio_target)

    Si no es posible, devuelve los factores de fallback y escribe una advertencia.
    """
    if not os.path.exists(venta_path):
        print(f"[WARN] No se encontró {venta_path}. Usando factores de fallback.")
        return FALLBACK_FACTOR_SOLES, FALLBACK_FACTOR_USD

    try:
        df = pd.read_csv(venta_path)
    except Exception as e:
        print(f"[WARN] Error leyendo {venta_path}: {e}. Usando factores de fallback.")
        return FALLBACK_FACTOR_SOLES, FALLBACK_FACTOR_USD

    # Ver columnas necesarias; si faltan, intentar calcular precio_soles_corrientes y precio_target
    required = {'anio', 'precio_target', 'precio_soles_corrientes', 'precio_usd'}
    cols = set(df.columns)
    # If precio_soles_corrientes or precio_target missing, try to compute from available cols
    if 'precio_soles_corrientes' not in cols and {'precio_usd', 'tipo_cambio'}.issubset(cols):
        df['precio_soles_corrientes'] = df['precio_usd'] * df['tipo_cambio']
    if 'precio_target' not in cols and {'precio_usd', 'tipo_cambio', 'ipc'}.issubset(cols):
        # IPC base 2009 = 77.45
        df['factor_deflacion'] = df['ipc'] / 77.45
        df['precio_soles_corrientes'] = df.get('precio_soles_corrientes', df['precio_usd'] * df['tipo_cambio'])
        df['precio_target'] = df['precio_soles_corrientes'] / df['factor_deflacion']

    # Check again that necessary columns exist
    cols = set(df.columns)
    if not {'anio', 'precio_target'}.issubset(cols):
        print(f"[WARN] El archivo {venta_path} no contiene columnas necesarias para calcular factores. Usando fallback.")
        return FALLBACK_FACTOR_SOLES, FALLBACK_FACTOR_USD

    # Filtrar por año más reciente
    try:
        max_year = int(df['anio'].max())
        df_recent = df[df['anio'] == max_year].copy()
        if df_recent.empty:
            df_recent = df.copy()
    except Exception:
        # si anio no convertible o ausente, usar todo el dataframe
        df_recent = df.copy()

    # Evitar divisiones por cero o NaNs
    df_recent = df_recent.replace([np.inf, -np.inf], np.nan).dropna(subset=['precio_target'])

    if df_recent.empty:
        print(f"[WARN] No hay filas válidas en {venta_path} para calcular factores. Usando fallback.")
        return FALLBACK_FACTOR_SOLES, FALLBACK_FACTOR_USD

    # precio_soles_corrientes may be missing -> try to compute or drop
    if 'precio_soles_corrientes' not in df_recent.columns and {'precio_usd', 'tipo_cambio'}.issubset(df_recent.columns):
        df_recent['precio_soles_corrientes'] = df_recent['precio_usd'] * df_recent['tipo_cambio']

    # Compute ratios, handle missing
    factor_soles_vals = []
    factor_usd_vals = []
    if 'precio_soles_corrientes' in df_recent.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = df_recent['precio_soles_corrientes'] / df_recent['precio_target']
            factor_soles_vals = ratio.replace([np.inf, -np.inf], np.nan).dropna().values

    if 'precio_usd' in df_recent.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_usd = df_recent['precio_usd'] / df_recent['precio_target']
            factor_usd_vals = ratio_usd.replace([np.inf, -np.inf], np.nan).dropna().values

    factor_soles = float(np.nanmean(factor_soles_vals)) if len(factor_soles_vals) > 0 else np.nan
    factor_usd = float(np.nanmean(factor_usd_vals)) if len(factor_usd_vals) > 0 else np.nan

    if np.isnan(factor_soles) or np.isnan(factor_usd):
        print(f"[WARN] No se pudieron calcular factores completos desde {venta_path}. Usando fallback donde falte.")
        if np.isnan(factor_soles):
            factor_soles = FALLBACK_FACTOR_SOLES
        if np.isnan(factor_usd):
            factor_usd = FALLBACK_FACTOR_USD

    return factor_soles, factor_usd


def load_model_and_preproc():
    """
    Carga el modelo joblib (bundle con 'model' y 'features' esperado).
    Devuelve: model, features(list), distrito_map(dict col->label), bias_log(float), factor_soles, factor_usd
    """
    model_path = find_first_existing(DEFAULT_MODEL_PATHS)
    preproc_path = find_first_existing(DEFAULT_PREPROC_PATHS)

    # Si no existe modelo en rutas por defecto, pedimos al usuario que lo seleccione
    if model_path is None:
        model_path = filedialog.askopenfilename(
            title="Selecciona model.joblib",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")],
        )
        if not model_path:
            raise RuntimeError("No se seleccionó ningún modelo.")

    bundle = joblib.load(model_path)
    # bundle puede ser dict con keys o el propio estimador
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    features = bundle.get("features", None) if isinstance(bundle, dict) else None
    bias_log = float(bundle.get("bias_log", 0.0)) if isinstance(bundle, dict) else 0.0

    preproc = None
    if preproc_path:
        try:
            preproc = joblib.load(preproc_path)
        except Exception:
            preproc = None

    # Si no hay features en el bundle, buscar en preproc
    if features is None and preproc is not None:
        features = preproc.get("features", None)

    # Fallback conservador si no hay features guardadas
    if features is None:
        features = ["log_superficie", "banos", "garajes", "antiguedad", "habitaciones"]

    distrito_cols = [c for c in features if c.startswith("distrito_")]
    distrito_map = {c: c.replace("distrito_", "") for c in distrito_cols}  # col -> label

    # Calcular factores de conversión basados en Data/venta_ready.csv
    factor_soles, factor_usd = compute_conversion_factors()

    return model, features, distrito_map, bias_log, factor_soles, factor_usd


class SimpleValuGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Valu.ai - Estimador de Precio (Frontend)")
        self.geometry("600x500")
        self.resizable(False, False)

        try:
            loaded = load_model_and_preproc()
            (self.model,
             self.features,
             self.distrito_map,
             self.bias_log,
             self.factor_soles,
             self.factor_usd) = loaded
        except Exception as e:
            messagebox.showerror("Error cargando modelo", str(e))
            self.destroy()
            return

        # reverse map: label -> column
        self.label_to_col = {label: col for col, label in self.distrito_map.items()}
        self.create_widgets()

    def create_widgets(self):
        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        row = 0
        ttk.Label(frm, text="Superficie (m²):").grid(column=0, row=row, sticky=tk.W, **pad)
        self.superficie_var = tk.StringVar(value="50")
        ttk.Entry(frm, textvariable=self.superficie_var, width=20).grid(column=1, row=row, **pad)

        row += 1
        ttk.Label(frm, text="Baños:").grid(column=0, row=row, sticky=tk.W, **pad)
        self.banos_var = tk.StringVar(value="1")
        ttk.Entry(frm, textvariable=self.banos_var, width=20).grid(column=1, row=row, **pad)

        row += 1
        ttk.Label(frm, text="Garajes:").grid(column=0, row=row, sticky=tk.W, **pad)
        self.garajes_var = tk.StringVar(value="0")
        ttk.Entry(frm, textvariable=self.garajes_var, width=20).grid(column=1, row=row, **pad)

        row += 1
        ttk.Label(frm, text="Antigüedad (años):").grid(column=0, row=row, sticky=tk.W, **pad)
        self.antig_var = tk.StringVar(value="5")
        ttk.Entry(frm, textvariable=self.antig_var, width=20).grid(column=1, row=row, **pad)

        row += 1
        ttk.Label(frm, text="Habitaciones:").grid(column=0, row=row, sticky=tk.W, **pad)
        self.hab_var = tk.StringVar(value="2")
        ttk.Entry(frm, textvariable=self.hab_var, width=20).grid(column=1, row=row, **pad)

        row += 1
        ttk.Label(frm, text="Distrito:").grid(column=0, row=row, sticky=tk.W, **pad)
        distrito_values = sorted(list(self.label_to_col.keys()))
        if not distrito_values:
            distrito_values = ["(ninguno)"]
        self.distrito_var = tk.StringVar(value=distrito_values[0])
        self.distrito_cb = ttk.Combobox(frm, textvariable=self.distrito_var,
                                        values=distrito_values, width=37, state="readonly")
        self.distrito_cb.grid(column=1, row=row, **pad)

        row += 1
        self.apply_bias_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Aplicar corrección de sesgo (interno)", variable=self.apply_bias_var).grid(
            column=0, row=row, columnspan=2, sticky=tk.W, **pad
        )

        row += 1
        ttk.Button(frm, text="Predecir", command=self.on_predict).grid(column=0, row=row, **pad)
        ttk.Button(frm, text="Cargar otro modelo", command=self.on_load_model).grid(column=1, row=row, **pad)

        row += 1
        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=2, sticky="ew", pady=8)

        row += 1
        ttk.Label(frm, text="Precio estimado:").grid(column=0, row=row, sticky=tk.W, **pad)
        row += 1

        # Text widget para mostrar solo los dos valores (USD y PEN), con formato grande/negrita
        self.output_text = tk.Text(frm, height=4, width=60, wrap=tk.WORD)
        self.output_text.grid(column=0, row=row, columnspan=2, **pad)
        self.output_text.configure(state=tk.DISABLED)

        # Configurar tag para valores grandes y negrita
        bold_font = tkfont.Font(self.output_text, self.output_text.cget("font"))
        bold_font.configure(size=14, weight="bold")
        self.output_text.tag_configure("big", font=bold_font, justify="center")

    def safe_float(self, s, default=0.0):
        try:
            return float(str(s).strip())
        except Exception:
            return default

    def on_load_model(self):
        path = filedialog.askopenfilename(title="Selecciona model.joblib",
                                          filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")])
        if not path:
            return
        try:
            bundle = joblib.load(path)
            self.model = bundle.get("model", bundle)
            self.features = bundle.get("features", self.features)
            self.bias_log = float(bundle.get("bias_log", self.bias_log))
            # actualizar distrito_map y label_to_col
            self.distrito_map = {c: c.replace("distrito_", "") for c in self.features if c.startswith("distrito_")}
            self.label_to_col = {label: col for col, label in self.distrito_map.items()}
            distrito_values = sorted(list(self.label_to_col.keys())) or ["(ninguno)"]
            self.distrito_cb["values"] = distrito_values
            self.distrito_cb.set(distrito_values[0])
            messagebox.showinfo("Modelo cargado", f"Modelo cargado desde: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")

    def on_predict(self):
        try:
            superficie = self.safe_float(self.superficie_var.get(), default=float("nan"))
            if math.isnan(superficie) or superficie <= 0:
                messagebox.showerror("Error de entrada", "Superficie inválida (>0).")
                return
            banos = int(self.safe_float(self.banos_var.get(), default=1))
            garajes = int(self.safe_float(self.garajes_var.get(), default=0))
            antig = int(self.safe_float(self.antig_var.get(), default=0))
            hab = int(self.safe_float(self.hab_var.get(), default=0))
        except Exception as e:
            messagebox.showerror("Error", f"Entrada inválida: {e}")
            return

        # Construir fila con ceros para todas las features esperadas
        row = {f: 0.0 for f in self.features}

        # log_superficie
        row["log_superficie"] = float(np.log1p(superficie))
        # imputaciones/otros
        if "banos" in row:
            row["banos"] = banos
        if "garajes" in row:
            row["garajes"] = garajes
        if "antiguedad" in row:
            row["antiguedad"] = antig
        if "habitaciones" in row:
            row["habitaciones"] = hab

        # distrito:
        distrito_label = self.distrito_var.get()
        if distrito_label and distrito_label != "(ninguno)":
            col = self.label_to_col.get(distrito_label)
            if col in row:
                row[col] = 1.0

        X = pd.DataFrame([row], columns=self.features)

        try:
            # Predicción en espacio log y aplicación de bias internamente si se selecciona
            y_pred_log = float(self.model.predict(X)[0])
            if self.apply_bias_var.get():
                y_pred_log = y_pred_log + float(self.bias_log)

            # Convertir a Soles 2009
            pred_soles_2009 = float(np.expm1(y_pred_log))

            # Convertir a precios actuales usando factores calculados al inicio
            pred_soles_actual = pred_soles_2009 * float(self.factor_soles)
            pred_usd_actual = pred_soles_2009 * float(self.factor_usd)

            # Formatear valores
            usd_str = f"$ {pred_usd_actual:,.2f}"
            pen_str = f"S/ {pred_soles_actual:,.2f}"

            # Mostrar SÓLO los dos valores en la caja de texto, con formato grande/negrita
            self.output_text.configure(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            # Insert USD line, tag with big
            self.output_text.insert(tk.END, usd_str + "\n", "big")
            self.output_text.insert(tk.END, pen_str + "\n", "big")
            self.output_text.configure(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error predicción", str(e))


def main():
    app = SimpleValuGui()
    if not app.winfo_exists():
        sys.exit(1)
    app.mainloop()


if __name__ == "__main__":
    main()