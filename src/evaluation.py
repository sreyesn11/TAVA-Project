"""
evaluation.py
Comparacion cuantitativa entre pipeline baseline y mejorado.
Genera tablas de metricas y graficas comparativas.
"""

import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from src.preprocessing import baseline_preprocess, improved_preprocess
from src.edge_detection import (
    baseline_edges, baseline_mask,
    improved_edges, improved_mask,
    get_bottle_contour,
)
from src.contour_measurement import measure_bottle


# ── PIPELINE COMPLETO ─────────────────────────────────────────────────────────

def run_baseline(gray: np.ndarray) -> dict:
    """Ejecuta pipeline baseline. Retorna dict con resultados y tiempo."""
    t0 = time.time()
    prep = baseline_preprocess(gray)
    edges = baseline_edges(prep)
    mask = baseline_mask(edges)
    contour = get_bottle_contour(mask)
    measures = measure_bottle(mask, contour)
    elapsed = time.time() - t0
    return {
        "preprocessed": prep,
        "edges": edges,
        "mask": mask,
        "contour": contour,
        "measures": measures,
        "time_s": round(elapsed, 4),
    }


def run_improved(gray: np.ndarray) -> dict:
    """Ejecuta pipeline mejorado. Retorna dict con resultados y tiempo."""
    t0 = time.time()
    prep = improved_preprocess(gray)
    edges = improved_edges(prep)
    mask = improved_mask(edges, prep)
    contour = get_bottle_contour(mask)
    measures = measure_bottle(mask, contour)
    elapsed = time.time() - t0
    return {
        "preprocessed": prep,
        "edges": edges,
        "mask": mask,
        "contour": contour,
        "measures": measures,
        "time_s": round(elapsed, 4),
    }


# ── METRICAS DE CALIDAD DE BORDE ──────────────────────────────────────────────

def edge_quality_metrics(edges: np.ndarray) -> dict:
    """
    Metricas proxy de calidad de borde (sin mascara de referencia):
      - density      : proporcion de pixeles de borde
      - continuity   : porcentaje de bordes con al menos un vecino (conectividad)
      - edge_count   : numero total de pixeles de borde
    """
    h, w = edges.shape
    total = h * w
    edge_count = int(np.sum(edges > 0))
    density = edge_count / total

    # Continuidad: cuantos pixeles de borde tienen vecinos de borde
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    # Pixeles de borde que tocan otros pixeles de borde (en vecindad 3x3)
    neighbor_count = cv2.filter2D((edges > 0).astype(np.uint8), -1, kernel)
    connected = int(np.sum((edges > 0) & (neighbor_count > 1)))
    continuity = connected / edge_count if edge_count > 0 else 0.0

    return {
        "edge_count": edge_count,
        "edge_density": round(density, 6),
        "edge_continuity": round(continuity, 4),
    }


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU entre dos mascaras binarias."""
    a = mask_a > 0
    b = mask_b > 0
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return round(float(intersection / union), 4) if union > 0 else 0.0


# ── TABLA RESUMEN ─────────────────────────────────────────────────────────────

def build_summary_table(results: list) -> pd.DataFrame:
    """
    Construye DataFrame resumen con medidas baseline y mejorado por imagen.

    results: lista de dicts con claves
      'name', 'baseline', 'improved'
    """
    rows = []
    metric_keys = [
        "height_px", "width_max_px", "width_25_px", "width_50_px", "width_75_px",
        "width_neck_px", "width_base_px", "ratio_w_h", "ratio_neck_base",
        "area_px", "perimeter_px",
    ]

    for r in results:
        row = {"imagen": r["name"]}
        bm = r["baseline"]["measures"]
        im = r["improved"]["measures"]
        for k in metric_keys:
            row[f"base_{k}"] = bm.get(k, 0)
            row[f"impr_{k}"] = im.get(k, 0)
        row["base_time_s"] = r["baseline"]["time_s"]
        row["impr_time_s"] = r["improved"]["time_s"]

        # Calidad de borde
        bq = edge_quality_metrics(r["baseline"]["edges"])
        iq = edge_quality_metrics(r["improved"]["edges"])
        row["base_edge_density"] = bq["edge_density"]
        row["impr_edge_density"] = iq["edge_density"]
        row["base_edge_continuity"] = bq["edge_continuity"]
        row["impr_edge_continuity"] = iq["edge_continuity"]

        # IoU entre mascaras
        row["iou_masks"] = mask_iou(r["baseline"]["mask"], r["improved"]["mask"])

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_stats(df: pd.DataFrame):
    """Imprime estadisticas descriptivas comparativas."""
    measure_cols = ["height_px", "width_max_px", "width_50_px", "ratio_w_h", "ratio_neck_base"]
    print("\n=== ESTADISTICAS COMPARATIVAS (media ± std) ===\n")
    print(f"{'Medida':<22} {'Baseline':>18} {'Mejorado':>18} {'Delta%':>10}")
    print("-" * 72)
    for col in measure_cols:
        bc = f"base_{col}"
        ic = f"impr_{col}"
        if bc in df.columns and ic in df.columns:
            bm = df[bc].mean()
            bs = df[bc].std()
            im = df[ic].mean()
            is_ = df[ic].std()
            delta = ((im - bm) / bm * 100) if bm != 0 else 0
            print(f"{col:<22} {bm:>8.1f} ± {bs:<6.1f}   {im:>8.1f} ± {is_:<6.1f}  {delta:>+8.1f}%")
    print()


# ── GRAFICAS ──────────────────────────────────────────────────────────────────

def plot_comparison_bar(df: pd.DataFrame, output_path: str = None):
    """
    Grafica de barras comparativa de medidas clave entre baseline y mejorado.
    """
    metrics = ["height_px", "width_max_px", "width_50_px", "ratio_w_h", "ratio_neck_base"]
    labels = ["Altura", "Ancho Max", "Ancho 50%", "Ratio W/H", "Ratio Cuello/Base"]

    base_means = [df[f"base_{m}"].mean() for m in metrics]
    impr_means = [df[f"impr_{m}"].mean() for m in metrics]
    base_stds = [df[f"base_{m}"].std() for m in metrics]
    impr_stds = [df[f"impr_{m}"].std() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_b = ax.bar(x - width / 2, base_means, width, yerr=base_stds,
                    label="Baseline", color="#4c72b0", capsize=5, alpha=0.85)
    bars_i = ax.bar(x + width / 2, impr_means, width, yerr=impr_stds,
                    label="Mejorado", color="#dd8452", capsize=5, alpha=0.85)

    ax.set_xlabel("Medida")
    ax.set_ylabel("Valor (pixeles o ratio)")
    ax.set_title("Comparacion de medidas: Baseline vs Mejorado\n(media ± std entre imagenes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig


def plot_edge_quality(df: pd.DataFrame, output_path: str = None):
    """
    Grafica comparativa de calidad de borde y tiempo de procesamiento.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1) Densidad de borde
    ax = axes[0]
    ax.bar(df["imagen"], df["base_edge_density"], label="Baseline", alpha=0.8, color="#4c72b0")
    ax.bar(df["imagen"], df["impr_edge_density"], label="Mejorado", alpha=0.6, color="#dd8452")
    ax.set_title("Densidad de borde")
    ax.set_ylabel("Pixeles borde / total")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    # 2) Continuidad de borde
    ax = axes[1]
    ax.bar(df["imagen"], df["base_edge_continuity"], label="Baseline", alpha=0.8, color="#4c72b0")
    ax.bar(df["imagen"], df["impr_edge_continuity"], label="Mejorado", alpha=0.6, color="#dd8452")
    ax.set_title("Continuidad de borde")
    ax.set_ylabel("Fraccion bordes conectados")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    # 3) Tiempo de procesamiento
    ax = axes[2]
    ax.plot(df["imagen"], df["base_time_s"], "o-", label="Baseline", color="#4c72b0")
    ax.plot(df["imagen"], df["impr_time_s"], "s-", label="Mejorado", color="#dd8452")
    ax.set_title("Tiempo de procesamiento")
    ax.set_ylabel("Segundos")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.suptitle("Metricas de calidad de borde y rendimiento", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig


def plot_contour_overlay(gray: np.ndarray,
                         baseline_result: dict,
                         improved_result: dict,
                         title: str = "",
                         output_path: str = None):
    """
    Muestra la imagen original con overlay de contornos baseline (azul) y mejorado (naranja).
    """
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Dibujar contorno baseline en azul
    if baseline_result["contour"] is not None:
        cv2.drawContours(display, [baseline_result["contour"]], -1, (66, 114, 176), 3)

    # Dibujar contorno mejorado en naranja
    if improved_result["contour"] is not None:
        cv2.drawContours(display, [improved_result["contour"]], -1, (221, 132, 82), 3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original (gris)")
    axes[0].axis("off")

    axes[1].imshow(baseline_result["edges"], cmap="gray")
    axes[1].set_title("Bordes Baseline")
    axes[1].axis("off")

    axes[2].imshow(display)
    axes[2].set_title("Overlay contornos\nAzul=Baseline  Naranja=Mejorado")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig
