"""
utils.py  --  v2.3
Utilidades generales: visualizacion, comparacion de profundidad de bits,
overlay con imagen de referencia, guardado de resultados.

v2.1 additions:
  - plot_projections(): visualiza proyecciones de fila/columna + bbox
  - plot_edge_methods(): compara los 5 metodos de deteccion de bordes
  - plot_full_pipeline_v2(): visualizacion completa del pipeline v2.1
  - plot_closing_comparison(): comparacion con y sin cierre morfologico
  - plot_bbox_sources(): visualiza las 4 fuentes del bounding box
v2.3 additions:
  - preview_crop_window(): ventana emergente cv2 para confirmar el recorte
  - plot_bottle_measurements(): visualizacion completa de medidas por zona
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
    Muestra los 4 pasos del pipeline en figura 2x2.
    'gray' debe ser la imagen de display (uint8), no la uint16 original.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = [
        (gray,  "1. Imagen (display uint8)"),
        (prep,  "2. Preprocesado"),
        (edges, "3. Deteccion de bordes"),
        (mask,  "4. Mascara de estructura"),
    ]
    for ax, (img, step_title) in zip(axes.flat, steps):
        ax.imshow(img, cmap="gray")
        ax.set_title(step_title, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def show_bit_depth_comparison(img_raw: np.ndarray, title: str = "") -> plt.Figure:
    """
    Compara la imagen original (uint16 o uint8) con su conversion a uint8.
    Muestra histogramas para cuantificar la perdida de informacion tonal.

    La conversion uint16->uint8 colapsa 65536 niveles de intensidad a 256,
    perdiendo 256x de resolucion tonal. Esto puede afectar la deteccion de
    bordes suaves o estructuras de bajo contraste.
    """
    from src.preprocessing import to_gray

    if img_raw.ndim == 3:
        gray_orig = to_gray(img_raw)
    else:
        gray_orig = img_raw.copy()

    if gray_orig.dtype == np.uint16:
        gray8 = (gray_orig / 257).astype(np.uint8)
        orig_label = f"uint16 ({gray_orig.shape})"
        orig_bins = 512
        orig_range = (0, 65535)
    else:
        gray8 = gray_orig.copy()
        orig_label = f"uint8 ({gray_orig.shape})"
        orig_bins = 256
        orig_range = (0, 255)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Imagenes
    axes[0, 0].imshow(gray_orig, cmap="gray",
                      vmin=orig_range[0], vmax=orig_range[1])
    axes[0, 0].set_title(f"Original  --  {orig_label}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray8, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Conversion a uint8  --  {gray8.shape}")
    axes[0, 1].axis("off")

    # Histogramas
    axes[1, 0].hist(gray_orig.ravel(), bins=orig_bins, range=orig_range,
                    color="#4c72b0", alpha=0.85)
    axes[1, 0].set_title(f"Histograma original ({orig_bins} bins)")
    axes[1, 0].set_xlabel("Valor de pixel")
    axes[1, 0].set_ylabel("Frecuencia")

    axes[1, 1].hist(gray8.ravel(), bins=256, range=(0, 255),
                    color="#dd8452", alpha=0.85)
    axes[1, 1].set_title("Histograma uint8 (256 bins)")
    axes[1, 1].set_xlabel("Valor de pixel")

    fig.suptitle(f"Analisis de profundidad de bits  --  {title}", fontsize=12)
    fig.tight_layout()
    return fig


def annotate_measurements(gray: np.ndarray, measures: dict, contour=None) -> np.ndarray:
    """
    Dibuja medidas geometricas genericas sobre la imagen.
    Compatible con metricas v2: compacidad, solidez, aspect_ratio.
    (Eliminadas referencias a cuello/base de botella.)
    """
    if gray.ndim == 2:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        vis = gray.copy()

    bbox = measures.get("bbox", (0, 0, 0, 0))
    x, y, w, h = bbox
    if w == 0 or h == 0:
        return vis

    # Bounding box
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Lineas de ancho relativo
    fractions  = [0.25, 0.50, 0.75]
    colors     = [(255, 100, 0), (0, 200, 255), (200, 0, 255)]
    width_keys = ["width_25_px", "width_50_px", "width_75_px"]
    labels     = ["25%", "50%", "75%"]

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

    # Contorno principal
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 128, 255), 2)

    # Texto de medidas genericas (v2)
    text_lines = [
        f"Alto: {measures.get('height_px', 0)} px",
        f"Ancho max: {measures.get('width_max_px', 0)} px",
        f"Compacidad: {measures.get('compactness', 0):.3f}",
        f"Solidez: {measures.get('solidity', 0):.3f}",
        f"Aspect ratio: {measures.get('aspect_ratio', 0):.3f}",
    ]
    text_x = max(x - 200, 5)
    text_y = y
    for i, line in enumerate(text_lines):
        cv2.putText(vis, line, (text_x, text_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, line, (text_x, text_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return vis


def show_morphology_metrics_table(morph_results: dict) -> pd.DataFrame:
    """
    Construye tabla comparativa de metricas para cada operacion morfologica.
    morph_results: retorno de evaluation.run_morphology_analysis().
    """
    rows = []
    for op_name, data in morph_results.items():
        m = data["metrics"]
        rows.append({
            "Operacion":      op_name,
            "edge_count":     m.get("edge_count", 0),
            "edge_density":   m.get("edge_density", 0),
            "edge_continuity":m.get("edge_continuity", 0),
            "n_components":   m.get("n_components", 0),
            "mean_comp_size": m.get("mean_comp_size", 0),
            "fragmentation":  m.get("fragmentation", 0),
        })
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, output_folder: str):
    """Guarda el DataFrame resumen como CSV y HTML."""
    os.makedirs(output_folder, exist_ok=True)
    csv_path  = os.path.join(output_folder, "resumen_medidas.csv")
    html_path = os.path.join(output_folder, "resumen_medidas.html")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_html(html_path, index=False)
    print(f"  CSV  guardado: {csv_path}")
    print(f"  HTML guardado: {html_path}")


def format_measures_table(measures_baseline: dict, measures_improved: dict) -> pd.DataFrame:
    """
    DataFrame comparativo de dos columnas (Baseline / Mejorado).
    Metricas genericas v2 (sin cuello/base de botella).
    """
    keys = [
        ("height_px",    "Altura (px)"),
        ("width_max_px", "Ancho maximo (px)"),
        ("width_25_px",  "Ancho al 25% altura (px)"),
        ("width_50_px",  "Ancho al 50% altura (px)"),
        ("width_75_px",  "Ancho al 75% altura (px)"),
        ("area_px",      "Area (px^2)"),
        ("perimeter_px", "Perimetro (px)"),
        ("compactness",  "Compacidad (circularidad)"),
        ("solidity",     "Solidez"),
        ("extent",       "Extent (area/bbox)"),
        ("aspect_ratio", "Ratio Ancho/Alto"),
    ]
    rows = []
    for k, label in keys:
        rows.append({
            "Medida":   label,
            "Baseline": measures_baseline.get(k, "-"),
            "Mejorado": measures_improved.get(k, "-"),
        })
    return pd.DataFrame(rows)


# =============================================================================
# v2.1 – NUEVAS VISUALIZACIONES
# =============================================================================

# ── PROYECCIONES DE BORDE ─────────────────────────────────────────────────────

def plot_projections(edges: np.ndarray,
                      col_proj: np.ndarray,
                      row_proj: np.ndarray,
                      bbox: tuple = None,
                      title: str = "",
                      output_path: str = None) -> plt.Figure:
    """
    Visualiza el mapa de bordes junto con las proyecciones por columnas y filas.

    La proyeccion por columnas (eje X) muestra donde se concentran los bordes
    horizontalmente -- los picos corresponden a los bordes laterales del objeto.
    La proyeccion por filas (eje Y) muestra la distribucion vertical de bordes.

    bbox: (x, y, w, h) calculado desde las proyecciones. Se superpone en verde.
    """
    fig = plt.figure(figsize=(18, 8), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[2.5, 1.5, 1.5], hspace=0.4, wspace=0.35)

    # Panel izquierdo: mapa de bordes con bbox
    ax_edges = fig.add_subplot(gs[:, 0])
    ax_edges.imshow(edges, cmap="gray")
    if bbox is not None:
        x, y, bw, bh = bbox
        rect = mpatches.Rectangle(
            (x, y), bw, bh,
            linewidth=2, edgecolor="lime", facecolor="none", label="bbox"
        )
        ax_edges.add_patch(rect)
        ax_edges.legend(loc="upper right", fontsize=8)
    ax_edges.set_title("Mapa de bordes + Bounding Box estimado", fontsize=10)
    ax_edges.axis("off")

    # Proyeccion por columnas (densidad horizontal)
    ax_col = fig.add_subplot(gs[0, 1:])
    x_idx = np.arange(len(col_proj))
    ax_col.fill_between(x_idx, col_proj, alpha=0.4, color="#4c72b0")
    ax_col.plot(x_idx, col_proj, color="#4c72b0", linewidth=0.7)
    ax_col.set_title("Proyeccion vertical (suma por columna)", fontsize=9)
    ax_col.set_xlabel("Columna (x)", fontsize=8)
    ax_col.set_ylabel("Pixeles activos", fontsize=8)
    if bbox is not None:
        ax_col.axvline(bbox[0],          color="lime", linestyle="--",
                       linewidth=1.2, label="x_min")
        ax_col.axvline(bbox[0] + bbox[2], color="lime", linestyle=":",
                       linewidth=1.2, label="x_max")
        ax_col.legend(fontsize=7)
    ax_col.tick_params(labelsize=7)
    ax_col.grid(alpha=0.2)

    # Proyeccion por filas (densidad vertical)
    ax_row = fig.add_subplot(gs[1, 1:])
    y_idx = np.arange(len(row_proj))
    ax_row.fill_between(y_idx, row_proj, alpha=0.4, color="#dd8452")
    ax_row.plot(y_idx, row_proj, color="#dd8452", linewidth=0.7)
    ax_row.set_title("Proyeccion horizontal (suma por fila)", fontsize=9)
    ax_row.set_xlabel("Fila (y)", fontsize=8)
    ax_row.set_ylabel("Pixeles activos", fontsize=8)
    if bbox is not None:
        ax_row.axvline(bbox[1],          color="lime", linestyle="--",
                       linewidth=1.2, label="y_min")
        ax_row.axvline(bbox[1] + bbox[3], color="lime", linestyle=":",
                       linewidth=1.2, label="y_max")
        ax_row.legend(fontsize=7)
    ax_row.tick_params(labelsize=7)
    ax_row.grid(alpha=0.2)

    fig.suptitle(title or "Analisis de proyecciones matriciales", fontsize=12)

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig


# ── COMPARACION DE METODOS DE BORDE ──────────────────────────────────────────

def plot_edge_methods(edge_maps: dict,
                       edge_metrics: dict,
                       title: str = "",
                       output_path: str = None) -> plt.Figure:
    """
    Compara visualmente los 5 metodos de deteccion de bordes con sus metricas.

    Para cada metodo muestra: mapa de bordes + densidad + continuidad + fragmentacion.
    Permite comparar visualmente cual metodo detecta mejor los bordes de la botella.
    """
    methods = list(edge_maps.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        ax.imshow(edge_maps[method], cmap="gray")
        m = edge_metrics.get(method, {})
        info = (
            f"{method}\n"
            f"dens={m.get('edge_density', 0):.4f}\n"
            f"cont={m.get('edge_continuity', 0):.3f}\n"
            f"frag={m.get('fragmentation', 0):.5f}\n"
            f"comp={m.get('n_components', 0)}"
        )
        ax.set_title(info, fontsize=8)
        ax.axis("off")

    fig.suptitle(title or "Comparacion de metodos de deteccion de bordes", fontsize=12)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig


# ── COMPARACION CON/SIN CIERRE MORFOLOGICO ────────────────────────────────────

def plot_closing_comparison(edges_no_closing: np.ndarray,
                              edges_with_closing: np.ndarray,
                              metrics_no: dict,
                              metrics_with: dict,
                              title: str = "",
                              output_path: str = None) -> plt.Figure:
    """
    Compara los bordes con y sin cierre morfologico.

    El cierre morfologico une extremos de bordes cercanos y reduce la
    fragmentacion, pero puede introducir bordes falsos si el kernel es grande.
    Esta visualizacion cuantifica el trade-off entre conectividad y precision.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, edges, metrics, label in [
        (axes[0], edges_no_closing,   metrics_no,   "Sin cierre morfologico"),
        (axes[1], edges_with_closing, metrics_with, "Con cierre morfologico"),
    ]:
        ax.imshow(edges, cmap="gray")
        info = (
            f"{label}\n"
            f"edge_count={metrics.get('edge_count', 0):,}\n"
            f"density={metrics.get('edge_density', 0):.4f}\n"
            f"continuity={metrics.get('edge_continuity', 0):.3f}\n"
            f"n_components={metrics.get('n_components', 0):,}\n"
            f"fragmentation={metrics.get('fragmentation', 0):.5f}"
        )
        ax.set_title(info, fontsize=9)
        ax.axis("off")

    fig.suptitle(title or "Experimento: cierre morfologico sobre bordes combinados", fontsize=12)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig


# ── FUENTES DEL BOUNDING BOX ─────────────────────────────────────────────────

def plot_bbox_sources(display_img: np.ndarray,
                       bbox_result: dict,
                       title: str = "",
                       output_path: str = None) -> plt.Figure:
    """
    Visualiza el bounding box detectado por cada fuente matricial:
      - Proyeccion (verde)
      - Componentes conectados (azul)
      - Contorno puntuado (rojo)
      - Hough Lines (amarillo)
      - Consenso final (blanco, grueso)

    Permite comparar la concordancia entre fuentes y detectar outliers.
    """
    if display_img.ndim == 2:
        canvas = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = display_img.copy()

    # Colores BGR para cada fuente
    color_map = {
        "bbox_projection":  (0,   200, 0),    # verde
        "bbox_components":  (255, 100, 0),    # azul
        "bbox_contour":     (0,   0,   220),  # rojo
        "bbox_hough":       (0,   220, 220),  # amarillo
    }
    legend_labels = []

    for key, color in color_map.items():
        bbox = bbox_result.get(key)
        if bbox is not None:
            x, y, bw, bh = bbox
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
            label_names = {
                "bbox_projection":  "Proyeccion",
                "bbox_components":  "Componentes",
                "bbox_contour":     "Contorno",
                "bbox_hough":       "Hough",
            }
            legend_labels.append((color, label_names[key]))

    # Consenso final (mas grueso)
    consensus = bbox_result.get("bbox_consensus")
    if consensus is not None:
        x, y, bw, bh = consensus
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (255, 255, 255), 3)
        legend_labels.append(((255, 255, 255), "Consenso"))

    # Score del mejor contorno
    score = bbox_result.get("contour_score", 0.0)
    n_cand = bbox_result.get("n_candidates", 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(display_img, cmap="gray")
    axes[0].set_title("Imagen original (display)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Fuentes del bounding box\n"
        f"Score contorno={score:.3f}  Candidatos componentes={n_cand}",
        fontsize=9
    )
    axes[1].axis("off")

    # Leyenda de colores
    patches = []
    color_names_plt = {
        "Proyeccion":   "#00c800",
        "Componentes":  "#ff6400",
        "Contorno":     "#0000dc",
        "Hough":        "#00dcdc",
        "Consenso":     "#ffffff",
    }
    for _, label in legend_labels:
        c = color_names_plt.get(label, "#aaaaaa")
        patches.append(mpatches.Patch(color=c, label=label))
    if patches:
        axes[1].legend(handles=patches, loc="lower right", fontsize=8,
                       framealpha=0.7)

    fig.suptitle(title or "Analisis de fuentes del bounding box", fontsize=12)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig


# ── PIPELINE COMPLETO v2.1 ────────────────────────────────────────────────────

def plot_full_pipeline_v2(pipeline_result: dict,
                           img_name: str = "",
                           output_path: str = None) -> plt.Figure:
    """
    Visualizacion completa del pipeline v2.2 en 6 paneles:
      1. Imagen original normalizada para display
      2. Imagen preprocesada (bilateral + CLAHE)
      3. Bordes combinados (5 metodos, voto mayoritario)
      4. Silueta externa (cierre fuerte + relleno) -- fuente principal del bbox
      5. Bounding box detectado superpuesto con fuentes de color
      6. Recorte final sobre la imagen original

    v2.2: El panel 4 muestra la silueta externa (no los bordes+cierre).
    La silueta prioriza la forma global del objeto e ignora detalles internos.
    """
    disp      = pipeline_result["img_display"]
    prep      = pipeline_result["preprocessed"]
    comb      = pipeline_result["combined_edges"]
    sil_mask  = pipeline_result.get("silhouette_mask",
                                     pipeline_result["edges_with_closing"])
    crop      = pipeline_result.get("crop_display", disp)

    # Panel 5: display con bboxes de todas las fuentes
    if disp.ndim == 2:
        bbox_canvas = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    else:
        bbox_canvas = disp.copy()

    bbox        = pipeline_result.get("bbox_consensus")
    bbox_result = pipeline_result.get("bbox_result", {})
    sil_bbox    = pipeline_result.get("silhouette_bbox")

    # Fuentes secundarias en colores finos
    color_map_bgr = {
        "bbox_projection": (0,   200, 0),
        "bbox_components": (200, 100, 0),
        "bbox_contour":    (0,   0,   200),
        "bbox_hough":      (0,   200, 200),
    }
    for key, color in color_map_bgr.items():
        b = bbox_result.get(key)
        if b is not None:
            x_, y_, w_, h_ = b
            cv2.rectangle(bbox_canvas, (x_, y_), (x_ + w_, y_ + h_), color, 1)

    # Silueta externa en magenta
    if sil_bbox is not None:
        sx, sy, sw_, sh_ = sil_bbox
        cv2.rectangle(bbox_canvas, (sx, sy), (sx + sw_, sy + sh_), (200, 0, 200), 2)

    # Consenso final en blanco grueso
    if bbox is not None:
        bx, by, bw_, bh_ = bbox
        cv2.rectangle(bbox_canvas, (bx, by), (bx + bw_, by + bh_), (255, 255, 255), 3)

    # Diagnostico
    diag    = pipeline_result.get("diagnosis", {})
    ok_diag = diag.get("ok", False)
    src     = pipeline_result.get("bbox_source", "?")
    warns   = diag.get("warnings", [])
    warn_txt = " | ".join(warns[:2]) if warns else "Sin advertencias"
    panel5_title = (
        f"5. Bounding box detectado  [{src.upper()}]\n"
        f"{'OK' if ok_diag else 'ADVERTENCIA'}: {warn_txt[:60]}\n"
        "Magenta=Silueta  Blanco=Consensus"
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    panels = [
        (disp,         "1. Original (display uint8)",             True),
        (prep,         "2. Preprocesado\n(Bilateral + CLAHE)",    True),
        (comb,         "3. Bordes combinados\n(5 metodos, voto)", True),
        (sil_mask,     "4. Silueta externa\n"
                       "(cierre fuerte + relleno)",               True),
        (cv2.cvtColor(bbox_canvas, cv2.COLOR_BGR2RGB),
                       panel5_title,                              False),
        (crop,         "6. Recorte final\n(sobre imagen original)",True),
    ]

    for ax, (img, panel_title, use_gray) in zip(axes.flat, panels):
        cmap = "gray" if use_gray and img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(panel_title, fontsize=7.5)
        ax.axis("off")

    # Metricas en el supertitulo
    bm = pipeline_result.get("bbox_metrics", {})
    vm = pipeline_result.get("validation", {})
    t  = pipeline_result.get("time_total_s", 0)

    sup = (
        f"{img_name}  |  "
        f"{vm.get('resolution_mpx', '?')} MPx  "
        f"{vm.get('bit_depth', '?')} bits  "
        f"{vm.get('n_channels', '?')} canales  |  "
        f"Tiempo: {t}s  |  "
        f"BBox: {bm.get('width', '?')} x {bm.get('height', '?')} px  "
        f"aspect={bm.get('aspect_ratio', '?')}  "
        f"score={bm.get('contour_score', 0):.3f}"
    )
    fig.suptitle(sup, fontsize=9, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig


# ── VENTANA EMERGENTE DE PREVIEW ─────────────────────────────────────────────

def preview_crop_window(img_display: np.ndarray,
                         bbox: tuple,
                         img_name: str = "") -> bool:
    """
    Muestra una ventana emergente con el recorte propuesto sobre la imagen
    completa para que el usuario confirme si es correcto antes de continuar.

    Lado izquierdo: imagen completa (reescalada) con el bbox en verde.
    Lado derecho:   recorte ampliado para verificar detalle.
    Parte inferior: instrucciones de teclado.

    Controles
    ---------
    ENTER o C  →  continuar el pipeline
    Q o ESC    →  cancelar el pipeline

    Retorna
    -------
    True  si el usuario aprueba o si el entorno no tiene display.
    False si el usuario cancela.
    """
    if bbox is None:
        print("[PREVIEW] Sin bbox disponible, continuando automaticamente.")
        return True

    bx, by, bw, bh = [int(v) for v in bbox]

    # Convertir a BGR uint8 para cv2.imshow
    disp = img_display.copy()
    if disp.dtype == np.uint16:
        disp = (disp / 257).astype(np.uint8)
    if disp.ndim == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    elif disp.ndim == 3 and disp.shape[2] == 3:
        # Asumir RGB (de rawpy) → BGR para cv2
        disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

    h_full, w_full = disp.shape[:2]

    # Escalar la imagen completa a un ancho maximo de 900 px
    max_w = 900
    scale = min(max_w / w_full, 800 / h_full, 1.0)
    if scale < 1.0:
        small = cv2.resize(disp, (int(w_full * scale), int(h_full * scale)),
                           interpolation=cv2.INTER_AREA)
        bx_s = int(bx * scale)
        by_s = int(by * scale)
        bw_s = max(3, int(bw * scale))
        bh_s = max(3, int(bh * scale))
    else:
        small = disp.copy()
        bx_s, by_s, bw_s, bh_s = bx, by, bw, bh

    h_s, w_s = small.shape[:2]

    # Rectangulo verde sobre imagen completa
    canvas = small.copy()
    cv2.rectangle(canvas, (bx_s, by_s), (bx_s + bw_s, by_s + bh_s),
                  (0, 230, 0), 2)

    # Recorte original (sin escalar primero, luego ajustar altura)
    y1c = max(0, by);  y2c = min(h_full, by + bh)
    x1c = max(0, bx);  x2c = min(w_full, bx + bw)
    crop_bgr = disp[y1c:y2c, x1c:x2c]

    if crop_bgr.size > 0:
        crop_h, crop_w = crop_bgr.shape[:2]
        # Escalar recorte para que tenga la misma altura que canvas
        crop_scale = h_s / crop_h if crop_h > 0 else 1.0
        crop_scaled = cv2.resize(crop_bgr,
                                  (max(1, int(crop_w * crop_scale)), h_s),
                                  interpolation=cv2.INTER_AREA)
    else:
        crop_scaled = np.zeros((h_s, 200, 3), dtype=np.uint8)

    # Separador vertical gris
    sep = np.full((h_s, 6, 3), 80, dtype=np.uint8)

    # Panel de instrucciones en la parte inferior del canvas
    bar_h = 50
    bar   = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar,
                f"ENTER o C = continuar  |  Q o ESC = cancelar    "
                f"bbox: x={bx} y={by} w={bw} h={bh}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 0), 1,
                cv2.LINE_AA)
    cv2.putText(bar,
                f"Imagen: {img_name}   |   "
                f"Recorte: {bw} x {bh} px  "
                f"(ratio={bw/bh:.2f})",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
                cv2.LINE_AA)

    canvas_with_bar = np.vstack([canvas, bar])

    # Ajustar recorte al alto del canvas_with_bar
    full_h = canvas_with_bar.shape[0]
    if crop_scaled.shape[0] != full_h:
        crop_scaled = cv2.resize(crop_scaled,
                                  (max(1, int(crop_scaled.shape[1] * full_h / crop_scaled.shape[0])),
                                   full_h),
                                  interpolation=cv2.INTER_AREA)
    sep_full = np.full((full_h, 6, 3), 80, dtype=np.uint8)

    combined = np.hstack([canvas_with_bar, sep_full, crop_scaled])

    win_name = f"Vista previa: {img_name}" if img_name else "Vista previa del recorte"

    try:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(combined.shape[1], 1600),
                          min(combined.shape[0], 900))
        cv2.imshow(win_name, combined)
        cv2.moveWindow(win_name, 50, 50)

        print(f"\n[PREVIEW] {win_name}")
        print(f"  Recorte: x={bx}, y={by}, w={bw}, h={bh}  |  ratio={bw/bh:.2f}")
        print("  Presiona ENTER o C para continuar, Q o ESC para cancelar...")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (13, ord('c'), ord('C')):   # ENTER o C
                cv2.destroyWindow(win_name)
                print("  [OK] Recorte confirmado. Continuando pipeline...")
                return True
            if key in (27, ord('q'), ord('Q')):   # ESC o Q
                cv2.destroyWindow(win_name)
                print("  [CANCELADO] Pipeline cancelado por el usuario.")
                return False
            # Ventana cerrada con el boton X
            try:
                vis = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                vis = -1
            if vis < 1:
                print("  [OK] Ventana cerrada. Continuando pipeline...")
                return True

    except cv2.error as exc:
        print(f"[PREVIEW] No se pudo abrir ventana ({exc}). Continuando automaticamente.")
        return True


# ── MEDIDAS DETALLADAS DE LA BOTELLA EN EL RECORTE ───────────────────────────

def plot_bottle_measurements(bottle_measures: dict,
                              silhouette_crop: np.ndarray = None,
                              img_name: str = "",
                              output_path: str = None) -> plt.Figure:
    """
    Visualiza las medidas detalladas de la botella dentro del recorte.

    Paneles:
      1. Silueta con zonas coloreadas (cuello/hombro/cuerpo/base) y lineas de
         medicion en fracciones clave  [solo si silhouette_crop se proporciona]
      2. Perfil de anchos vs fraccion de altura con zonas sombreadas
      3. Anchos min / promedio / max por zona (barras agrupadas)
      4. Tabla de metricas globales

    Parametros
    ----------
    bottle_measures : dict  retornado por measure_bottle_in_crop()
    silhouette_crop : np.ndarray uint8, mascara binaria recortada, opcional
    img_name        : str, nombre de la imagen para el titulo
    output_path     : str, si se provee guarda la figura en disco
    """
    from src.contour_measurement import _compute_width_profile

    zone_colors_mpl = {
        "cuello": "#e74c3c",
        "hombro": "#f39c12",
        "cuerpo": "#27ae60",
        "base":   "#2980b9",
    }
    zone_colors_rgb = {
        "cuello": (231,  76,  60),
        "hombro": (243, 156,  18),
        "cuerpo": ( 39, 174,  96),
        "base":   ( 41, 128, 185),
    }
    zone_fracs = {
        "cuello": (0.00, 0.20),
        "hombro": (0.20, 0.35),
        "cuerpo": (0.35, 0.80),
        "base":   (0.80, 1.00),
    }
    frac_keys = [
        (0.10, "ancho_10pct_px"), (0.20, "ancho_20pct_px"),
        (0.25, "ancho_25pct_px"), (0.30, "ancho_30pct_px"),
        (0.40, "ancho_40pct_px"), (0.50, "ancho_50pct_px"),
        (0.60, "ancho_60pct_px"), (0.70, "ancho_70pct_px"),
        (0.75, "ancho_75pct_px"), (0.80, "ancho_80pct_px"),
        (0.90, "ancho_90pct_px"),
    ]

    height_bottle = bottle_measures.get("height_bottle_px", 0)
    if height_bottle == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sin medidas disponibles",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        return fig

    # Pre-computar perfil completo si hay silueta disponible
    has_silhouette = silhouette_crop is not None and silhouette_crop.size > 0
    full_profile   = None
    active_rows    = []
    y_start_crop   = 0
    bot_height     = height_bottle

    if has_silhouette:
        full_profile = _compute_width_profile(silhouette_crop)
        active_rows  = [i for i, pw in enumerate(full_profile) if pw > 0]
        if active_rows:
            y_start_crop = active_rows[0]
            bot_height   = active_rows[-1] - y_start_crop + 1

    # ── Layout ───────────────────────────────────────────────────────────────
    if has_silhouette:
        fig = plt.figure(figsize=(20, 9))
        gs  = fig.add_gridspec(2, 3, width_ratios=[1.1, 2.0, 1.1],
                                hspace=0.5, wspace=0.42)
        ax_sil    = fig.add_subplot(gs[:, 0])
        ax_prof   = fig.add_subplot(gs[:, 1])
        ax_zones  = fig.add_subplot(gs[0, 2])
        ax_global = fig.add_subplot(gs[1, 2])
    else:
        fig = plt.figure(figsize=(14, 9))
        gs  = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.1],
                                hspace=0.5, wspace=0.42)
        ax_sil    = None
        ax_prof   = fig.add_subplot(gs[:, 0])
        ax_zones  = fig.add_subplot(gs[0, 1])
        ax_global = fig.add_subplot(gs[1, 1])

    # ── PANEL 1: Silueta con zonas y lineas de medicion ──────────────────────
    if ax_sil is not None and has_silhouette:
        h_c, w_c = silhouette_crop.shape[:2]
        sil_rgb  = np.zeros((h_c, w_c, 3), dtype=np.uint8)

        for zone_name, (f0, f1) in zone_fracs.items():
            r0    = y_start_crop + int(bot_height * f0)
            r1    = y_start_crop + int(bot_height * f1)
            color = zone_colors_rgb[zone_name]
            for r in range(r0, min(r1 + 1, h_c)):
                if full_profile[r] > 0:
                    act = np.where(silhouette_crop[r, :] > 0)[0]
                    if len(act) >= 2:
                        sil_rgb[r, act[0]:act[-1] + 1] = color

        # Lineas blancas en fracciones clave
        for frac in [0.10, 0.25, 0.50, 0.75, 0.90]:
            r = y_start_crop + int(bot_height * frac)
            r = max(0, min(r, h_c - 1))
            if full_profile[r] > 0:
                act = np.where(silhouette_crop[r, :] > 0)[0]
                if len(act) >= 2:
                    x1, x2 = int(act[0]), int(act[-1])
                    for dr in range(-1, 2):
                        rr = max(0, min(r + dr, h_c - 1))
                        sil_rgb[rr, x1:x2 + 1] = (255, 255, 255)

        ax_sil.imshow(sil_rgb)
        legend_patches = [
            mpatches.Patch(color=zone_colors_mpl[z],
                           label=f"{z.capitalize()} ({int(f0*100)}-{int(f1*100)}%)")
            for z, (f0, f1) in zone_fracs.items()
        ]
        legend_patches.append(mpatches.Patch(color="white", label="Medicion"))
        ax_sil.legend(handles=legend_patches, loc="lower right",
                       fontsize=6.5, framealpha=0.85)
        ax_sil.set_title("Silueta: zonas y\nmediciones clave", fontsize=9)
        ax_sil.axis("off")

    # ── PANEL 2: Perfil de anchos (curva horizontal) ──────────────────────────
    if full_profile is not None and active_rows:
        y_frac_full = [(r - y_start_crop) / bot_height for r in active_rows]
        w_full      = [full_profile[r] for r in active_rows]
    else:
        y_frac_full = [frac for frac, _ in frac_keys]
        w_full      = [bottle_measures.get(key, 0) for _, key in frac_keys]

    w_max_plot = max(w_full) if w_full else 1

    for zone_name, (f0, f1) in zone_fracs.items():
        ax_prof.axhspan(f0, f1, alpha=0.13, color=zone_colors_mpl[zone_name],
                        label=zone_name.capitalize())

    ax_prof.plot(w_full, y_frac_full, color="#2c3e50", linewidth=1.8, zorder=5)
    ax_prof.fill_betweenx(y_frac_full, 0, w_full, alpha=0.20, color="#2c3e50")

    for frac, key in frac_keys:
        w_px = bottle_measures.get(key, 0)
        if w_px > 0:
            ax_prof.plot(w_px, frac, "o", color="#c0392b", markersize=4.5, zorder=6)
            ax_prof.annotate(
                f"{int(frac * 100)}%: {w_px}px",
                xy=(w_px, frac),
                xytext=(w_px + w_max_plot * 0.04, frac),
                fontsize=6.5, va="center", color="#2c3e50",
            )

    ax_prof.set_ylim(0.0, 1.0)
    ax_prof.invert_yaxis()
    ax_prof.set_xlim(left=0)
    ax_prof.set_xlabel("Ancho (px)", fontsize=9)
    ax_prof.set_ylabel("Fraccion de altura  (0 = tapa, 1 = base)", fontsize=8)
    ax_prof.set_title("Perfil de anchos por zona", fontsize=10, fontweight="bold")
    ax_prof.legend(loc="lower right", fontsize=7.5, framealpha=0.85)
    ax_prof.grid(alpha=0.25)

    # ── PANEL 3: Anchos por zona (barras agrupadas) ───────────────────────────
    zone_list = ["cuello", "hombro", "cuerpo", "base"]
    xlabels   = ["Cuello\n(0-20%)", "Hombro\n(20-35%)",
                  "Cuerpo\n(35-80%)", "Base\n(80-100%)"]
    prom_vals = [bottle_measures.get(f"zona_{z}", {}).get("promedio_px", 0)
                 for z in zone_list]
    max_vals  = [bottle_measures.get(f"zona_{z}", {}).get("maximo_px",  0)
                 for z in zone_list]
    min_vals  = [bottle_measures.get(f"zona_{z}", {}).get("minimo_px",  0)
                 for z in zone_list]

    x_pos   = np.arange(len(zone_list))
    bw      = 0.25
    top_val = max(max_vals) if any(max_vals) else 1

    ax_zones.bar(x_pos - bw, min_vals,  bw, color="#5dade2", alpha=0.85, label="Min")
    ax_zones.bar(x_pos,      prom_vals, bw,
                 color=[zone_colors_mpl[z] for z in zone_list], alpha=0.85, label="Promedio")
    ax_zones.bar(x_pos + bw, max_vals,  bw, color="#e74c3c", alpha=0.65, label="Max")

    for xi, pv in zip(x_pos, prom_vals):
        if pv > 0:
            ax_zones.text(xi, pv + top_val * 0.02, f"{pv:.0f}",
                          ha="center", va="bottom", fontsize=7)

    ax_zones.set_xticks(x_pos)
    ax_zones.set_xticklabels(xlabels, fontsize=7.5)
    ax_zones.set_ylabel("Ancho (px)", fontsize=8)
    ax_zones.set_title("Anchos por zona\n(min / prom / max)", fontsize=9)
    ax_zones.legend(fontsize=7, loc="upper right")
    ax_zones.grid(axis="y", alpha=0.25)

    # ── PANEL 4: Metricas globales (tabla) ────────────────────────────────────
    global_fields = [
        ("height_bottle_px",    "Altura botella (px)"),
        ("width_max_px",        "Ancho maximo (px)"),
        ("width_min_px",        "Ancho minimo (px)"),
        ("width_mean_px",       "Ancho promedio (px)"),
        ("aspect_ratio",        "Aspect ratio"),
        ("ratio_cuello_cuerpo", "Ratio cuello/cuerpo"),
        ("simetria_vertical",   "Simetria vertical"),
    ]
    table_data = []
    for key, lbl in global_fields:
        val = bottle_measures.get(key, 0)
        fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
        table_data.append([lbl, fmt])

    ax_global.axis("off")
    tbl = ax_global.table(
        cellText=table_data,
        colLabels=["Metrica", "Valor"],
        cellLoc="left",
        loc="center",
        colWidths=[0.68, 0.32],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.45)
    ax_global.set_title("Metricas globales", fontsize=9, pad=4)

    # ── Supertitulo ───────────────────────────────────────────────────────────
    w_max = bottle_measures.get("width_max_px", 0)
    h_bot = bottle_measures.get("height_bottle_px", 0)
    ar    = bottle_measures.get("aspect_ratio", 0.0)
    sim   = bottle_measures.get("simetria_vertical", 0.0)
    ratio = bottle_measures.get("ratio_cuello_cuerpo", 0.0)

    sup = (
        f"{img_name}  |  "
        f"Alto={h_bot}px  Ancho_max={w_max}px  "
        f"AspectRatio={ar:.3f}  "
        f"Simetria={sim:.3f}  "
        f"Ratio_cuello/cuerpo={ratio:.3f}"
    )
    fig.suptitle(sup, fontsize=9, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig
