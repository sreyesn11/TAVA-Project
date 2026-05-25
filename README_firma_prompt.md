# Prompt para Claude: integrar firma geometrica automatica

## Objetivo
Agregar al repositorio un flujo que, a partir de las detecciones de bordes (mascara/contorno) de las botellas, extraiga automaticamente la firma geometrica radial, la grafique y la compare contra la firma de referencia (real) ya construida en `firma_botella/firma_completa_0_360.csv`. El resultado debe cuantificar el error entre ambas firmas.

## Contexto del repo
- Pipeline actual de bordes y contornos en `src/edge_detection.py` y `src/evaluation.py`.
- Medicion geomterica generica en `src/contour_measurement.py`.
- Ejecucion por script en `run_pipeline.py`.
- Carpeta `firma_botella/` contiene:
  - `Results_0_90.csv` y `Results_0_m90.csv`: medidas ImageJ.
  - `generar_firma_botella.py`: script que genera `firma_completa_0_360.csv`.
  - `firma_completa_0_360.csv`: firma de referencia (real).

## Requerimientos funcionales
1. **Extraer firma radial desde el contorno/mascara detectada**:
   - Entrada: mascara binaria o contorno principal (por imagen).
   - Salida: firma como pares (angulo_deg, distancia_px) de 0 a 360.
   - Metodo sugerido:
     - Calcular centroide del contorno.
     - Para cada angulo (0..359 o step configurable), lanzar un rayo y hallar el primer pixel de borde en la direccion del rayo (interseccion con contorno/silueta).
     - Convertir distancia a pixeles y guardar.
   - Debe ser robusto a huecos menores en el borde (puede usar mascara rellenada).

2. **Procesar y graficar la firma generada**:
   - Guardar CSV: `outputs/firma/firma_modelo_<IMG>.csv`.
   - Guardar grafica: `outputs/firma/grafica_firma_<IMG>.png`.

3. **Comparar con firma de referencia**:
   - Cargar `firma_botella/firma_completa_0_360.csv`.
   - Interpolar si hay diferencias de angulo/step.
   - Calcular metricas: MAE, RMSE, MAPE (%), error maximo.
   - Guardar resumen por imagen en `outputs/firma/resumen_firma.csv`.
   - (Opcional) Graficar comparacion (modelo vs referencia) en un solo plot.

4. **Integracion en pipeline**:
   - Exponer una opcion en `run_pipeline.py` (por ejemplo `--signature`) que ejecute:
     - pipeline de bordes
     - extraccion de firma
     - comparacion con referencia
   - No romper comportamiento actual.

## Requerimientos tecnicos
- Mantener estilo y estructura actual del repo.
- Reusar funciones existentes cuando sea posible.
- Evitar dependencias nuevas si no son estrictamente necesarias.
- Usar solo archivos ASCII.

## Entregables esperados
- Nuevas funciones/clases para firma radial (probablemente en `src/contour_measurement.py` o nuevo modulo `src/signature.py`).
- Modificacion controlada del pipeline para ejecutar la firma.
- CSVs y graficas de salida.
- Comparacion cuantitativa con la referencia.

## Notas
- La firma de referencia ya esta en milimetros, pero la firma del modelo estara en pixeles. Si quieres comparar directamente, puedes:
  - Normalizar ambas firmas (p. ej. dividir por el maximo), o
  - Escalar por un factor si existe una conversion px->mm conocida.
  - Documenta la decision.

## Prompt sugerido para Claude
Necesito que integres al repo una extraccion automatica de firma geometrica radial a partir de la mascara/contorno de botella detectado por el pipeline. La firma debe cubrir 0-360 grados y guardarse como CSV y grafica. Luego compara esa firma con la referencia en `firma_botella/firma_completa_0_360.csv`, calculando MAE, RMSE, MAPE y error maximo, y guarda un resumen por imagen. Integra esto al script `run_pipeline.py` con un flag nuevo (ej. `--signature`) sin romper el comportamiento actual. Evita dependencias nuevas. Usa ASCII. Documenta cualquier normalizacion o escala aplicada para comparar firmas en mm vs px. Produce cambios listos para ejecutar y salidas en `outputs/firma/`.
