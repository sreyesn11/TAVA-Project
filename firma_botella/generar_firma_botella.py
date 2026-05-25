from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

# Carpeta donde estarán los dos archivos CSV exportados desde ImageJ/Fiji
CARPETA_TRABAJO = Path(".")
# Archivos de entrada
ARCHIVO_SUPERIOR = CARPETA_TRABAJO / "Results_0_90.csv"
ARCHIVO_INFERIOR = CARPETA_TRABAJO / "Results_0_m90.csv"

# Archivos de salida
ARCHIVO_FIRMA_COMPLETA = CARPETA_TRABAJO / "firma_completa_0_360.csv"
ARCHIVO_GRAFICA = CARPETA_TRABAJO / "grafica_firma_0_360.png"


# ============================================================
# FUNCIÓN PARA LEER LOS CSV DE IMAGEJ/FIJI
# ============================================================

def leer_csv_imagej(ruta_archivo):
    """
    Lee un archivo CSV exportado desde ImageJ/Fiji.

    Se espera que el archivo tenga al menos las columnas:
    - Angle
    - Length

    La columna Angle contiene el ángulo.
    La columna Length contiene la distancia medida en milímetros.
    """

    if not ruta_archivo.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    # ImageJ suele exportar los resultados separados por coma o por punto y coma.
    # Primero intentamos con punto y coma.
    try:
        df = pd.read_csv(ruta_archivo, sep=";")
        if len(df.columns) == 1:
            df = pd.read_csv(ruta_archivo, sep=",")
    except Exception:
        df = pd.read_csv(ruta_archivo, sep=",")

    # Limpiar nombres de columnas
    df.columns = [str(col).strip() for col in df.columns]

    # Buscar columnas sin depender de mayúsculas/minúsculas
    columnas = {col.lower(): col for col in df.columns}

    if "angle" not in columnas:
        raise ValueError(
            f"El archivo {ruta_archivo} no tiene columna 'Angle'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    if "length" not in columnas:
        raise ValueError(
            f"El archivo {ruta_archivo} no tiene columna 'Length'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    col_angle = columnas["angle"]
    col_length = columnas["length"]

    datos = df[[col_angle, col_length]].copy()
    datos.columns = ["Angle", "Length_mm"]

    # Convertir a números
    datos["Angle"] = pd.to_numeric(datos["Angle"], errors="coerce")
    datos["Length_mm"] = pd.to_numeric(datos["Length_mm"], errors="coerce")

    # Eliminar filas inválidas
    datos = datos.dropna(subset=["Angle", "Length_mm"])

    return datos


# ============================================================
# FUNCIONES DE ÁNGULOS
# ============================================================

def normalizar_angulo(angulo):
    """
    Convierte cualquier ángulo a escala 0° - 360°.

    Ejemplos:
    -90°  -> 270°
    -45°  -> 315°
    0°    -> 0°
    90°   -> 90°
    360°  -> 0°
    """
    return angulo % 360


def reflejar_por_simetria(angulo):
    """
    Refleja los ángulos del lado derecho hacia el lado izquierdo.

    Convención usada:

    90°           parte superior
     |
     |
180° ---- 0°      eje horizontal
     |
     |
    270°          parte inferior

    Como la botella es simétrica izquierda-derecha,
    el lado izquierdo se calcula así:

    ángulo_izquierdo = (180 - ángulo_derecho) mod 360
    """
    return (180 - angulo) % 360


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def generar_firma_completa():
    """
    Genera la firma radial completa de la botella de 0° a 360°.
    """

    # Crear la carpeta si no existe
    CARPETA_TRABAJO.mkdir(exist_ok=True)

    print("Leyendo archivos de entrada...")
    datos_superiores = leer_csv_imagej(ARCHIVO_SUPERIOR)
    datos_inferiores = leer_csv_imagej(ARCHIVO_INFERIOR)

    # Identificar origen de cada medición
    datos_superiores["Origen"] = "derecha_superior_0_a_90"
    datos_inferiores["Origen"] = "derecha_inferior_0_a_m90"

    # Unir los datos medidos del lado derecho
    datos_derecha = pd.concat(
        [datos_superiores, datos_inferiores],
        ignore_index=True
    )

    # Convertir ángulos negativos a 0-360
    datos_derecha["Angle_deg"] = datos_derecha["Angle"].apply(normalizar_angulo)

    # Crear el lado izquierdo usando simetría
    datos_izquierda = datos_derecha.copy()
    datos_izquierda["Angle_deg"] = datos_izquierda["Angle_deg"].apply(reflejar_por_simetria)
    datos_izquierda["Origen"] = "izquierda_reflejada"

    # Unir lado derecho medido + lado izquierdo reflejado
    datos_completos = pd.concat(
        [datos_derecha, datos_izquierda],
        ignore_index=True
    )

    # En algunos ángulos puede haber datos repetidos, por ejemplo 0°.
    # Si hay repetidos, se promedia la medida.
    firma_completa = (
        datos_completos
        .groupby("Angle_deg", as_index=False)
        .agg(Length_mm=("Length_mm", "mean"))
        .sort_values("Angle_deg")
        .reset_index(drop=True)
    )

    # Redondear para limpiar la tabla
    firma_completa["Angle_deg"] = firma_completa["Angle_deg"].round(6)
    firma_completa["Length_mm"] = firma_completa["Length_mm"].round(6)

    # Guardar CSV final
    firma_completa.to_csv(
        ARCHIVO_FIRMA_COMPLETA,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Firma completa guardada en: {ARCHIVO_FIRMA_COMPLETA}")

    return firma_completa


# ============================================================
# GRAFICAR FIRMA
# ============================================================

def graficar_firma(firma_completa):
    """
    Grafica la firma radial completa.

    Eje X: ángulo en grados, de 0° a 360°.
    Eje Y: distancia desde el centro hasta el borde, en mm.
    """

    firma_plot = firma_completa.copy()

    # Para cerrar visualmente la curva, agregamos 360° con el mismo valor de 0°
    if 0 in firma_plot["Angle_deg"].values:
        valor_0 = firma_plot.loc[
            firma_plot["Angle_deg"] == 0,
            "Length_mm"
        ].iloc[0]

        fila_360 = pd.DataFrame({
            "Angle_deg": [360.0],
            "Length_mm": [valor_0]
        })

        firma_plot = pd.concat(
            [firma_plot, fila_360],
            ignore_index=True
        )

    plt.figure(figsize=(12, 6))

    plt.plot(
        firma_plot["Angle_deg"],
        firma_plot["Length_mm"],
        marker="o",
        linewidth=2
    )

    plt.title("Firma radial completa de la botella, 0° a 360°")
    plt.xlabel("Ángulo, grados")
    plt.ylabel("Distancia desde el centro al borde, mm")

    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True)
    plt.tight_layout()

    # Guardar imagen
    plt.savefig(ARCHIVO_GRAFICA, dpi=300)
    print(f"Gráfica guardada en: {ARCHIVO_GRAFICA}")

    # Mostrar gráfica en pantalla
    plt.show()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    firma = generar_firma_completa()
    graficar_firma(firma)

    print("\nPrimeras filas de la firma completa:")
    print(firma.head())

    print("\nÚltimas filas de la firma completa:")
    print(firma.tail())