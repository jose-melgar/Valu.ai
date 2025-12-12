Carpeta Frontend: contiene una GUI simple en Tkinter para probar el modelo entrenado.

Archivos:
- gui_tkinter.py : script principal de la interfaz.

Requisitos mínimos:
- Python 3.8+
- Instalar dependencias:
  pip install pandas numpy joblib

Ubicación esperada de artefactos:
- artifacts/model.joblib
- artifacts/preprocessor.joblib (opcional, ayuda a listar distritos)

Cómo ejecutar:
1. Coloca `gui_tkinter.py` dentro de la carpeta `Frontend/`.
2. Desde la raíz del repo ejecuta:
   python Frontend/gui_tkinter.py
3. Si el script no encuentra `model.joblib` en las rutas por defecto te permitirá seleccionar el archivo manualmente.

Notas:
- La GUI construye las features esperadas (log_superficie, banos, garajes, antiguedad, habitaciones y dummies de distrito) y las pasa al modelo.
- Las predicciones se muestran en "Soles constantes 2009" (usar np.expm1 sobre la predicción en log).
- Si el modelo guardado incluye 'bias_log' el usuario puede optar por aplicar la corrección de sesgo.