#!/usr/bin/env python3
"""
Valu.ai - Enterprise Real Estate Estimator (v2.0)
Desarrollado para el Informe de Prácticas Profesionales - UPC.
Autor: José Melgar
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

# --- CONFIGURACIÓN ESTÉTICA ---
COLOR_BG = "#F5F6F7"
COLOR_PRIMARY = "#2C3E50"  # Azul Profesional
COLOR_ACCENT = "#27AE60"   # Verde Éxito
COLOR_TEXT = "#34495E"

# --- RUTAS POR DEFECTO ---
DEFAULT_MODEL_PATHS = [
    os.path.join("artifacts", "model.joblib"),
    "model.joblib"
]
DEFAULT_PREPROC_PATHS = [
    os.path.join("artifacts", "preprocessor.joblib"),
    "preprocessor.joblib"
]
VENTA_READY_PATH = os.path.join("Data", "venta_ready.csv")

# Factores de contingencia (BCRP)
FALLBACK_FACTOR_SOLES = 1.65
FALLBACK_FACTOR_USD = 0.45

# --- FUNCIONES DE LÓGICA Y CÁLCULO ---

def compute_conversion_factors(venta_path=VENTA_READY_PATH):
    """Calcula multiplicadores para pasar de Soles 2009 a precios actuales."""
    if not os.path.exists(venta_path):
        return FALLBACK_FACTOR_SOLES, FALLBACK_FACTOR_USD

    try:
        df = pd.read_csv(venta_path)
        # Filtrar por el año más reciente para tener el tipo de cambio e IPC actual
        max_year = int(df['anio'].max())
        df_recent = df[df['anio'] == max_year].copy()
        
        # Factor Soles = Precio Corriente / Precio Constante 2009
        factor_soles = (df_recent['precio_soles_corrientes'] / df_recent['precio_target']).mean()
        # Factor USD = Precio USD / Precio Constante 2009
        factor_usd = (df_recent['precio_usd'] / df_recent['precio_target']).mean()
        
        return float(factor_soles), float(factor_usd)
    except:
        return FALLBACK_FACTOR_SOLES, FALLBACK_FACTOR_USD

def load_engine():
    """Carga el modelo y los metadatos necesarios para la predicción."""
    # Buscar archivos
    model_p = next((p for p in DEFAULT_MODEL_PATHS if os.path.exists(p)), None)
    
    if not model_p:
        model_p = filedialog.askopenfilename(title="Seleccionar model.joblib", filetypes=[("Joblib", "*.joblib")])
    
    if not model_p:
        raise RuntimeError("No se encontró el archivo del modelo.")

    bundle = joblib.load(model_p)
    
    # Extraer componentes del bundle
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    features = bundle.get("features", []) if isinstance(bundle, dict) else []
    bias_log = float(bundle.get("bias_log", 0.0)) if isinstance(bundle, dict) else 0.0
    
    distrito_map = {c: c.replace("distrito_", "") for c in features if c.startswith("distrito_")}
    f_soles, f_usd = compute_conversion_factors()
    
    return model, features, distrito_map, bias_log, f_soles, f_usd

# --- INTERFAZ GRÁFICA ---

class ValuAiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Valu.ai | Enterprise Real Estate Estimator")
        self.geometry("650x680")
        self.configure(bg=COLOR_BG)
        self.resizable(False, False)

        try:
            (self.model, self.features, self.distrito_map, 
             self.bias_log, self.f_soles, self.f_usd) = load_engine()
            self.label_to_col = {label: col for col, label in self.distrito_map.items()}
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar motor de IA: {e}")
            self.destroy()
            return

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("Action.TButton", font=("Segoe UI", 11, "bold"), foreground="white", background=COLOR_PRIMARY)
        style.configure("Group.TLabelframe", background=COLOR_BG)
        style.configure("Group.TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground=COLOR_PRIMARY)

    def create_widgets(self):
        # Header
        header = tk.Frame(self, bg=COLOR_PRIMARY, height=90)
        header.pack(fill=tk.X)
        tk.Label(header, text="VALU.AI", font=("Segoe UI", 26, "bold"), fg="white", bg=COLOR_PRIMARY).pack(pady=(15, 0))
        tk.Label(header, text="Análisis Predictivo de Mercado Inmobiliario", font=("Segoe UI", 9), fg="#BDC3C7", bg=COLOR_PRIMARY).pack(pady=(0, 10))

        main_container = tk.Frame(self, bg=COLOR_BG, padx=30, pady=20)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Formulario
        prop_group = ttk.LabelFrame(main_container, text=" Especificaciones Técnicas ", style="Group.TLabelframe")
        prop_group.pack(fill=tk.X, pady=10)
        
        grid_params = {'padx': 10, 'pady': 8, 'sticky': 'w'}
        
        ttk.Label(prop_group, text="Superficie (m²):").grid(row=0, column=0, **grid_params)
        self.superficie_var = tk.StringVar(value="120")
        ttk.Entry(prop_group, textvariable=self.superficie_var, width=15).grid(row=0, column=1, **grid_params)

        ttk.Label(prop_group, text="Habitaciones:").grid(row=0, column=2, **grid_params)
        self.hab_var = tk.StringVar(value="3")
        ttk.Entry(prop_group, textvariable=self.hab_var, width=15).grid(row=0, column=3, **grid_params)

        ttk.Label(prop_group, text="Baños:").grid(row=1, column=0, **grid_params)
        self.banos_var = tk.StringVar(value="2")
        ttk.Entry(prop_group, textvariable=self.banos_var, width=15).grid(row=1, column=1, **grid_params)

        ttk.Label(prop_group, text="Cocheras:").grid(row=1, column=2, **grid_params)
        self.garajes_var = tk.StringVar(value="1")
        ttk.Entry(prop_group, textvariable=self.garajes_var, width=15).grid(row=1, column=3, **grid_params)

        # Ubicación
        loc_group = ttk.LabelFrame(main_container, text=" Ubicación y Antigüedad ", style="Group.TLabelframe")
        loc_group.pack(fill=tk.X, pady=10)

        ttk.Label(loc_group, text="Distrito:").grid(row=0, column=0, **grid_params)
        distritos = sorted(list(self.label_to_col.keys()))
        self.distrito_var = tk.StringVar(value=distritos[0] if distritos else "")
        self.distrito_cb = ttk.Combobox(loc_group, textvariable=self.distrito_var, values=distritos, state="readonly", width=42)
        self.distrito_cb.grid(row=0, column=1, columnspan=3, **grid_params)

        ttk.Label(loc_group, text="Antigüedad (años):").grid(row=1, column=0, **grid_params)
        self.antig_var = tk.StringVar(value="0")
        ttk.Entry(loc_group, textvariable=self.antig_var, width=15).grid(row=1, column=1, **grid_params)

        # Botón
        ttk.Button(main_container, text="GENERAR ESTIMACIÓN", style="Action.TButton", command=self.on_predict).pack(fill=tk.X, pady=20)

        # Resultados
        self.res_group = tk.Frame(main_container, bg="white", highlightbackground="#DCDDE1", highlightthickness=1)
        self.res_group.pack(fill=tk.BOTH, expand=True, pady=10)

        tk.Label(self.res_group, text="PRECIO ESTIMADO DE CIERRE", font=("Segoe UI", 10, "bold"), bg="white", fg=COLOR_TEXT).pack(pady=10)
        self.lbl_usd = tk.Label(self.res_group, text="$ 0.00", font=("Segoe UI", 32, "bold"), bg="white", fg=COLOR_PRIMARY)
        self.lbl_usd.pack()
        self.lbl_pen = tk.Label(self.res_group, text="S/ 0.00", font=("Segoe UI", 16), bg="white", fg="#7F8C8D")
        self.lbl_pen.pack(pady=(0, 15))

        # Footer
        footer = tk.Frame(self, bg="#ECF0F1", height=30)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(footer, text="Powered by Century 21 Evolution | Engine: Random Forest Regressor", font=("Segoe UI", 8), bg="#ECF0F1", fg="#95A5A6").pack(side=tk.LEFT, padx=15)

    def on_predict(self):
        """Maneja la entrada de datos, predicción y actualización de UI."""
        try:
            # 1. Recolección y Limpieza
            sup = float(self.superficie_var.get())
            ban = int(self.banos_var.get())
            gar = int(self.garajes_var.get())
            ant = int(self.antig_var.get())
            hab = int(self.hab_var.get())
            dist = self.distrito_var.get()

            # 2. Creación del vector de características (X)
            row = {f: 0.0 for f in self.features}
            row["log_superficie"] = np.log1p(sup)
            row["banos"] = ban
            row["garajes"] = gar
            row["antiguedad"] = ant
            row["habitaciones"] = hab
            
            col_dist = self.label_to_col.get(dist)
            if col_dist in row:
                row[col_dist] = 1.0

            X = pd.DataFrame([row], columns=self.features)

            # 3. Inferencia
            y_log = float(self.model.predict(X)[0])
            
            # Aplicar corrección de sesgo (Bias Correction)
            y_log += self.bias_log

            # 4. Post-procesamiento (Escalado de Soles 2009 a Actualidad)
            p_constante_2009 = np.expm1(y_log)
            p_usd = p_constante_2009 * self.f_usd
            p_pen = p_constante_2009 * self.f_soles

            # 5. Actualización de Interfaz
            self.lbl_usd.config(text=f"$ {p_usd:,.2f}")
            self.lbl_pen.config(text=f"S/ {p_pen:,.2f}")
            
        except Exception as e:
            messagebox.showerror("Error de Cálculo", f"Verifique los datos ingresados: {e}")

if __name__ == "__main__":
    app = ValuAiApp()
    app.mainloop()