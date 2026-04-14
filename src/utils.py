"""
utils.py
Utilidades generales: visualizacion, guardado de resultados, logging.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


def show_image(img: np.ndarray, title: str = "", cmap: str = "gray", figsize=(8, 6)):
    """Muestra una imagen en matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap=cmap if img.ndim == 2 else None)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def show_pipeline_steps(gray: np.ndarray, prep: np.ndarray,
                         edges: np.ndarray, mask: np.ndarray,
                         title: str = "Pipeline"):
    """
    Muestra los 4 pasos del pipeline en una figura de 2x2.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = [
        (gray,  "1. Imagen original (gris)"),
        (prep,  "2. Preprocesado"),
        (edges, "3. Deteccion de bordes"),
        (mask,  "4. Mascara de botella"),
    ]

    for ax, (img, step_title) in zip(axes.flat, steps):
        ax.imshow(img, cmap="gray")
        ax.set_title(step_title, fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def annotate_measurements(gray: np.ndarray, measures: dict, contour=None) -> np.ndarray:
    """
    Dibuja sobre la imagen las medidas geometricas:
    bounding box, lineas de ancho relativo y texto.
    Retorna imagen RGB anotada.
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    bbox = measures.get("bbox", (0, 0, 0, 0))
    x, y, w, h = bbox

    if w == 0 or h == 0:
        return vis

    # Bounding box
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Lineas de ancho relativo
    fractions = [0.25, 0.50, 0.75]
    colors = [(255, 100, 0), (0, 200, 255), (200, 0, 255)]
    width_keys = ["width_25_px", "width_50_px", "width_75_px"]
    labels = ["25%", "50%", "75%"]

    for frac, color, wkey, lbl in zip(fractions, colors, width_keys, labels):
        row = int(y + frac * h)
        w_px = measures.get(wkey, 0)
        if w_px > 0:
            cx = x + w // 2
            x1 = cx - w_px // 2
            x2 = cx + w_px // 2
            cv2.line(vis, (x1, row), (x2, row), color, 2)
            cv2.putText(vis, f"{lbl}: {w_px}px", (x2 + 5, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Contorno de botella
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 128, 255), 2)

    # Texto de medidas principales
    text_lines = [
        f"Alto: {measures.get('height_px', 0)} px",
        f"Ancho max: {measures.get('width_max_px', 0)} px",
        f"Ratio W/H: {measures.get('ratio_w_h', 0):.3f}",
        f"Cuello/Base: {measures.get('ratio_neck_base', 0):.3f}",
    ]
    text_x = max(x - 200, 5)
    text_y = y
    for i, line in enumerate(text_lines):
        cv2.putText(vis, line, (text_x, text_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, line, (text_x, text_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return vis


def save_outputs(df: pd.DataFrame, output_folder: str):
    """Guarda el DataFrame resumen como CSV y HTML."""
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "resumen_medidas.csv")
    html_path = os.path.join(output_folder, "resumen_medidas.html")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_html(html_path, index=False)
    print(f"  CSV  guardado: {csv_path}")
    print(f"  HTML guardado: {html_path}")


def format_measures_table(measures_baseline: dict, measures_improved: dict) -> pd.DataFrame:
    """
    Crea un DataFrame de dos columnas (Baseline / Mejorado) con las medidas.
    """
    keys = [
        ("height_px",       "Altura (px)"),
        ("width_max_px",    "Ancho maximo (px)"),
        ("width_25_px",     "Ancho al 25% altura (px)"),
        ("width_50_px",     "Ancho al 50% altura (px)"),
        ("width_75_px",     "Ancho al 75% altura (px)"),
        ("width_neck_px",   "Ancho cuello (px)"),
        ("width_base_px",   "Ancho base (px)"),
        ("ratio_w_h",       "Ratio Ancho/Alto"),
        ("ratio_neck_base", "Ratio Cuello/Base"),
        ("area_px",         "Area (px²)"),
        ("perimeter_px",    "Perimetro (px)"),
    ]
    rows = []
    for k, label in keys:
        rows.append({
            "Medida": label,
            "Baseline": measures_baseline.get(k, "-"),
            "Mejorado": measures_improved.get(k, "-"),
        })
    return pd.DataFrame(rows)
