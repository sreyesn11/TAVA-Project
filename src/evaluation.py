"""
evaluation.py  --  v2.3
Comparacion cuantitativa entre pipelines baseline y mejorado.
Incluye analisis morfologico comparativo, metricas de fragmentacion
y comparacion contra imagen de referencia anotada manualmente.

v2.1: pipeline multi-metodo, analisis matricial, bbox desde bordes.
v2.2: Correccion de seleccion incorrecta de region.
  - run_full_pipeline_v2() incluye silueta externa (cierre+relleno).
  - El consensus del bbox ya no toma union con proyeccion.
  - Se agrega diagnostico que explica por que la version anterior fallaba.
  - Salida completa segun el nuevo prompt: x_min, y_min, x_max, y_max,
    area, aspect_ratio, score, tiempo.
v2.3: Integracion de crop_from_reference.
  - run_full_pipeline_v2() acepta ref_path para usar el recuadro rojo
    dibujado en una imagen de referencia como fuente primaria del bbox.
  - Acepta preview=True para mostrar ventana emergente de confirmacion.
  - find_red_rectangle() y scale_rect() integrados en este modulo.
"""

import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

from src.preprocessing import baseline_preprocess, improved_preprocess, to_uint8_gray
from src.edge_detection import (
    baseline_edges, baseline_mask,
    improved_edges, improved_mask,
    get_primary_contour_from_mask,
    get_bottle_contour,
    compare_morphology,
    _auto_canny,
    _scaled_kernel_size,
    detect_all_methods,
    combine_edge_maps,
    projection_analysis,
    score_contours,
    detect_vertical_lines_hough,
    estimate_bottle_bbox,
    crop_from_bbox,
    find_external_silhouette,  # v2.2
)
from src.contour_measurement import measure_structure, measure_bottle, measure_bottle_in_crop
from src.utils import preview_crop_window  # v2.3


# ── CROP DESDE IMAGEN DE REFERENCIA ──────────────────────────────────────────

def find_red_rectangle(ref_path: str) -> tuple:
    """
    Detecta el rectangulo rojo dibujado manualmente en la imagen de referencia.

    Usa deteccion HSV de rojo en dos rangos (0-10 y 170-180 grados de hue)
    seguido de cierre morfologico para limpiar la mascara.

    Retorna
    -------
    (x, y, w, h) del bounding box del recuadro rojo en coordenadas de la referencia
    (ref_w, ref_h) dimensiones de la imagen de referencia
    """
    img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen de referencia: {ref_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0,   80,  80], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170,  80,  80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                           cv2.inRange(hsv, lower2, upper2))

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(
            "No se detecto ningun recuadro rojo en la imagen de referencia. "
            "Verifica que la imagen tenga un rectangulo rojo bien definido."
        )

    contour  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    ref_h, ref_w = img.shape[:2]

    return (x, y, w, h), (ref_w, ref_h)


def scale_rect(rect: tuple, src_size: tuple, dst_size: tuple) -> tuple:
    """
    Escala un rectangulo (x, y, w, h) de las dimensiones de origen a destino.

    Parametros
    ----------
    rect     : (x, y, w, h) en coordenadas de la imagen fuente
    src_size : (src_w, src_h) dimensiones de la imagen fuente
    dst_size : (dst_w, dst_h) dimensiones de la imagen destino

    Retorna
    -------
    (x2, y2, w2, h2) en coordenadas de la imagen destino
    (sx, sy)         factores de escala aplicados
    """
    x, y, w, h     = rect
    src_w, src_h   = src_size
    dst_w, dst_h   = dst_size

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    x2 = int(round(x * sx));  y2 = int(round(y * sy))
    w2 = int(round(w * sx));  h2 = int(round(h * sy))

    x2 = max(0, min(x2, dst_w - 1))
    y2 = max(0, min(y2, dst_h - 1))
    x3 = max(x2 + 1, min(x2 + w2, dst_w))
    y3 = max(y2 + 1, min(y2 + h2, dst_h))

    return (x2, y2, x3 - x2, y3 - y2), (sx, sy)


# ── PIPELINES COMPLETOS ────────────────────────────────────────────────────────

def run_baseline(img: np.ndarray) -> dict:
    """
    Pipeline baseline. Acepta uint8 o uint16, gray o RGB.
    Retorna dict con resultados y tiempo.
    """
    t0 = time.time()
    prep    = baseline_preprocess(img)
    edges   = baseline_edges(prep)
    mask    = baseline_mask(edges)
    contour = get_primary_contour_from_mask(mask)
    measures = measure_structure(mask, contour)
    elapsed = time.time() - t0
    return {
        "preprocessed": prep,
        "edges":         edges,
        "mask":          mask,
        "contour":       contour,
        "measures":      measures,
        "time_s":        round(elapsed, 4),
    }


def run_improved(img: np.ndarray) -> dict:
    """
    Pipeline mejorado. Acepta uint8 o uint16, gray o RGB.
    """
    t0 = time.time()
    prep    = improved_preprocess(img)
    edges   = improved_edges(prep)
    mask    = improved_mask(edges)
    contour = get_primary_contour_from_mask(mask)
    measures = measure_structure(mask, contour)
    elapsed = time.time() - t0
    return {
        "preprocessed": prep,
        "edges":         edges,
        "mask":          mask,
        "contour":       contour,
        "measures":      measures,
        "time_s":        round(elapsed, 4),
    }


# ── ANALISIS MORFOLOGICO ───────────────────────────────────────────────────────

def run_morphology_analysis(img: np.ndarray, kernel_size: int = 5) -> dict:
    """
    Analiza el impacto de cada operacion morfologica sobre los bordes.

    Flujo:
      1. Preprocesamiento mejorado (CLAHE + bilateral)
      2. Canny puro SIN morfologia
      3. Aplicar closing, opening, dilation, closing_large
      4. Medir calidad + fragmentacion para cada resultado

    Retorna dict con clave = nombre_operacion, valor = {edges, metrics, time_s}
    """
    prep = improved_preprocess(img)
    edges_raw = _auto_canny(prep, sigma=0.33, l2=True)

    scaled_ks = _scaled_kernel_size(kernel_size, prep)
    morph_variants = compare_morphology(edges_raw, kernel_size=scaled_ks)

    results = {}
    for op_name, edges in morph_variants.items():
        t0 = time.time()
        metrics = edge_quality_metrics(edges)
        metrics.update(edge_fragmentation(edges))
        elapsed = time.time() - t0
        results[op_name] = {
            "edges":   edges,
            "metrics": metrics,
            "time_s":  round(elapsed, 6),
        }
    return results


# ── PIPELINE COMPLETO v2.2 ─────────────────────────────────────────────────────

def run_full_pipeline_v2(img_data: dict,
                          use_hough: bool = True,
                          combine_method: str = "weighted_vote",
                          ref_path: str = None,
                          preview: bool = True) -> dict:
    """
    Pipeline completo v2.3 para deteccion y recorte de la botella de vidrio.

    Parametros
    ----------
    img_data       : dict con 'image_rgb' (uint16), 'image_display' (uint8), 'path'
    use_hough      : usar Hough Lines para refinamiento lateral del bbox
    combine_method : metodo de combinacion de mapas de borde
    ref_path       : ruta a la imagen de referencia con el recuadro rojo dibujado.
                     Si se provee, el bbox se calcula escalando ese recuadro a las
                     dimensiones de la imagen RAW (fuente de maxima prioridad).
                     Si es None, se usa la deteccion automatica (silueta + scoring).
    preview        : si True, muestra ventana emergente con el recorte propuesto
                     y espera confirmacion del usuario antes de continuar.
                     Presionar Q o ESC cancela el pipeline y retorna
                     {'cancelled': True} como unico campo relevante.

    Fuentes del bbox (prioridad decreciente):
      1. Referencia (recuadro rojo escalado)   → bbox_source = 'reference'
      2. Silueta externa (cierre + relleno)    → bbox_source = 'silhouette'
      3. Scoring matricial                     → bbox_source = 'matrix_scoring'
      4. Sin deteccion                         → bbox_source = 'none'

    Retorna dict con todos los resultados intermedios, metricas y salida final.
    """
    t_start = time.time()

    img_rgb     = img_data["image_rgb"]
    img_display = img_data["image_display"]
    img_name    = os.path.basename(img_data.get("path", ""))

    # -- 0. Recorte desde imagen de referencia (maxima prioridad) -------------
    ref_bbox   = None   # (x, y, w, h) escalado a dimensiones RAW
    ref_rect   = None   # (x, y, w, h) en coordenadas de la referencia
    ref_size   = None   # (ref_w, ref_h)
    ref_scale  = None   # (sx, sy)

    if ref_path is not None:
        try:
            ref_rect, ref_size = find_red_rectangle(ref_path)
            # Necesitamos (raw_w, raw_h) — se calculan despues de la validacion,
            # pero aqui solo necesitamos las dimensiones del array:
            raw_h_pre, raw_w_pre = img_rgb.shape[:2]
            ref_bbox, ref_scale  = scale_rect(ref_rect, ref_size,
                                               (raw_w_pre, raw_h_pre))
            print(f"[REF] Recuadro rojo detectado en referencia: "
                  f"x={ref_rect[0]}, y={ref_rect[1]}, "
                  f"w={ref_rect[2]}, h={ref_rect[3]}")
            print(f"[REF] Escalado a RAW: x={ref_bbox[0]}, y={ref_bbox[1]}, "
                  f"w={ref_bbox[2]}, h={ref_bbox[3]}  "
                  f"(scale x={ref_scale[0]:.3f}, y={ref_scale[1]:.3f})")
        except Exception as exc:
            print(f"[REF] Advertencia: no se pudo usar la referencia ({exc}). "
                  "Continuando con deteccion automatica.")
            ref_bbox = None

    # -- 1. Validacion --------------------------------------------------------
    h, w = img_rgb.shape[:2]
    n_channels = img_rgb.shape[2] if img_rgb.ndim == 3 else 1
    bit_depth  = img_rgb.dtype.itemsize * 8

    validation = {
        "shape":          img_rgb.shape,
        "dtype":          str(img_rgb.dtype),
        "n_channels":     n_channels,
        "bit_depth":      bit_depth,
        "resolution_mpx": round(h * w / 1e6, 2),
        "path":           img_data.get("path", ""),
    }

    # -- 2-3. Copias auxiliares (img_rgb NUNCA se modifica) ------------------
    gray8 = to_uint8_gray(img_rgb)

    # -- 4. Preprocesamiento: filtro bilateral + CLAHE opcional ---------------
    t_prep = time.time()
    preprocessed   = improved_preprocess(img_rgb)
    time_preprocess = round(time.time() - t_prep, 4)

    # -- 5. Deteccion con 5 metodos ------------------------------------------
    t_edges = time.time()
    edge_maps = detect_all_methods(preprocessed)
    time_edges = round(time.time() - t_edges, 4)

    # -- 6. Comparacion de mapas de bordes -----------------------------------
    edge_metrics = {}
    for method_name, emap in edge_maps.items():
        m = edge_quality_metrics(emap)
        m.update(edge_fragmentation(emap))
        m["total_edge_length"] = int(np.sum(emap > 0))
        edge_metrics[method_name] = m

    # -- 7. Combinacion de mapas (voto mayoritario) --------------------------
    combined_edges = combine_edge_maps(edge_maps, method=combine_method)
    combined_metrics = edge_quality_metrics(combined_edges)
    combined_metrics.update(edge_fragmentation(combined_edges))
    combined_metrics["total_edge_length"] = int(np.sum(combined_edges > 0))

    # -- 8. Cierre morfologico moderado (experimento) ------------------------
    ks     = _scaled_kernel_size(5, preprocessed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

    edges_no_closing   = combined_edges.copy()
    edges_with_closing = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)

    metrics_no_closing = edge_quality_metrics(edges_no_closing)
    metrics_no_closing.update(edge_fragmentation(edges_no_closing))
    metrics_no_closing["total_edge_length"] = int(np.sum(edges_no_closing > 0))

    metrics_with_closing = edge_quality_metrics(edges_with_closing)
    metrics_with_closing.update(edge_fragmentation(edges_with_closing))
    metrics_with_closing["total_edge_length"] = int(np.sum(edges_with_closing > 0))

    edges_final = edges_with_closing

    # -- 9. Silueta externa (cierre fuerte + relleno) -------------------------
    # Detecta la FORMA GLOBAL del objeto ignorando detalles internos.
    # Usa cierre agresivo para unir bordes del contorno externo → rellena
    # la region → extrae el contorno externo de la mascara rellena.
    t_sil = time.time()
    silhouette_contour, silhouette_mask = find_external_silhouette(
        edges_final, img_rgb.shape
    )
    time_silhouette = round(time.time() - t_sil, 4)

    silhouette_bbox = None
    if silhouette_contour is not None:
        sx, sy, sw_, sh_ = cv2.boundingRect(silhouette_contour)
        silhouette_bbox = (sx, sy, sw_, sh_)

    # -- 10. Analisis matricial completo (fuente de apoyo) -------------------
    t_bbox = time.time()
    bbox_result = estimate_bottle_bbox(
        edges_final, img_rgb.shape, use_hough=use_hough
    )
    time_bbox = round(time.time() - t_bbox, 4)

    # -- 11. Seleccion del bbox final y diagnostico --------------------------
    # Prioridad:
    #   1. Referencia (recuadro rojo escalado): fuente externa, maxima confianza
    #   2. Silueta externa (cierre+relleno): forma global del objeto
    #   3. Contorno puntuado del analisis matricial
    #   4. Sin deteccion

    bbox_from_matrix = bbox_result.get("bbox_consensus")

    if ref_bbox is not None:
        bbox_consensus = ref_bbox
        bbox_source    = "reference"
    elif silhouette_bbox is not None:
        bbox_consensus = silhouette_bbox
        bbox_source    = "silhouette"
    elif bbox_from_matrix is not None:
        bbox_consensus = bbox_from_matrix
        bbox_source    = "matrix_scoring"
    else:
        bbox_consensus = None
        bbox_source    = "none"

    # Metricas del bbox final
    bbox_metrics = {}
    if bbox_consensus is not None:
        bx, by, bw_, bh_ = bbox_consensus
        bbox_metrics = {
            "x_min":        bx,
            "y_min":        by,
            "x_max":        bx + bw_,
            "y_max":        by + bh_,
            "width":        bw_,
            "height":       bh_,
            "area":         bw_ * bh_,
            "aspect_ratio": round(bw_ / bh_, 4) if bh_ > 0 else 0.0,
            "coverage_frac":round((bw_ * bh_) / (h * w), 4),
            "source":       bbox_source,
            "contour_score":bbox_result.get("contour_score", 0.0),
        }

    # Diagnostico: explica la seleccion y potenciales problemas
    diagnosis = _diagnose_bbox_selection(
        bbox_consensus, bbox_result, silhouette_bbox,
        img_shape=(h, w)
    )

    print(f"[BBOX] Fuente: {bbox_source.upper()}  |  "
          f"{bbox_metrics.get('width', 0)} x {bbox_metrics.get('height', 0)} px  "
          f"aspect={bbox_metrics.get('aspect_ratio', 0):.3f}  "
          f"score={bbox_metrics.get('contour_score', 0):.3f}")

    # -- 11b. Preview: ventana emergente para confirmar el recorte -----------
    # Muestra la imagen con el bbox dibujado y el recorte al lado.
    # Espera que el usuario presione ENTER/C (continuar) o Q/ESC (cancelar).
    if preview and bbox_consensus is not None:
        confirmed = preview_crop_window(img_display, bbox_consensus,
                                         img_name=img_name)
        if not confirmed:
            return {"cancelled": True, "bbox_consensus": bbox_consensus,
                    "bbox_source": bbox_source, "bbox_metrics": bbox_metrics,
                    "validation": validation}

    # -- 12. Recorte posterior sobre la imagen ORIGINAL ----------------------
    # La imagen original (img_rgb) NO se modifica.
    # crop_rgb y crop_disp son copias independientes obtenidas con slicing.
    crop_rgb  = crop_from_bbox(img_rgb,     bbox_consensus)
    crop_disp = crop_from_bbox(img_display, bbox_consensus)

    # -- 13. Medidas dentro del recorte (silueta recortada) ------------------
    # Recortar la silueta al mismo bbox para tener coordenadas relativas al crop
    silhouette_mask_crop = crop_from_bbox(silhouette_mask, bbox_consensus, padding_frac=0.02)
    bottle_measures = measure_bottle_in_crop(silhouette_mask_crop)

    # -- 14. JSON de recorte -------------------------------------------------
    # Si se uso la referencia, incluir sus coordenadas y escala en el JSON.
    ref_shape_for_json = (ref_size[1], ref_size[0]) if ref_size else None  # (H, W)
    crop_json = generate_crop_json(
        img_shape        = (h, w),
        bbox_consensus   = bbox_consensus,
        bbox_metrics     = bbox_metrics,
        img_path         = img_data.get("path", ""),
        time_total_s     = 0,           # se completa abajo
        score            = bbox_metrics.get("contour_score", 0.0),
        reference_shape  = ref_shape_for_json,
        reference_bbox   = ref_rect,
    )

    t_total = round(time.time() - t_start, 4)

    # Completar el tiempo en el JSON ahora que esta calculado
    if crop_json:
        crop_json["metadata"]["time_total_s"] = t_total

    return {
        # Metadatos y validacion
        "validation":           validation,
        "time_total_s":         t_total,
        "time_preprocess_s":    time_preprocess,
        "time_edges_s":         time_edges,
        "time_silhouette_s":    time_silhouette,
        "time_bbox_s":          time_bbox,

        # Imagenes intermedias
        "img_display":          img_display,
        "gray8":                gray8,
        "preprocessed":         preprocessed,

        # Mapas de bordes
        "edge_maps":            edge_maps,
        "edge_metrics":         edge_metrics,
        "combined_edges":       combined_edges,
        "combined_metrics":     combined_metrics,
        "edges_no_closing":     edges_no_closing,
        "edges_with_closing":   edges_with_closing,
        "edges_final":          edges_final,
        "metrics_no_closing":   metrics_no_closing,
        "metrics_with_closing": metrics_with_closing,

        # Silueta externa (fuente principal del bbox)
        "silhouette_contour":   silhouette_contour,
        "silhouette_mask":      silhouette_mask,
        "silhouette_bbox":      silhouette_bbox,

        # Analisis matricial (fuente de apoyo)
        "bbox_result":          bbox_result,

        # Bbox final y metricas de salida
        "bbox_consensus":       bbox_consensus,
        "bbox_source":          bbox_source,
        "bbox_metrics":         bbox_metrics,
        "diagnosis":            diagnosis,

        # Recorte final (calculado post-procesamiento, aplicado sobre original)
        "crop_rgb":             crop_rgb,
        "crop_display":         crop_disp,

        # Medidas dentro del recorte (v2.2)
        "bottle_measures":      bottle_measures,

        # JSON de recorte (v2.2)
        "crop_json":            crop_json,
    }


def generate_crop_json(img_shape: tuple,
                        bbox_consensus: tuple,
                        bbox_metrics: dict,
                        img_path: str = "",
                        time_total_s: float = 0.0,
                        score: float = 0.0,
                        reference_shape: tuple = None,
                        reference_bbox: tuple = None) -> dict:
    """
    Genera el JSON de recorte obligatorio segun el prompt.

    Formato de salida:
    {
      "instruction": "...",
      "crop_optional": true,
      "raw": {
        "width": W,
        "height": H,
        "rect": {"x": x, "y": y, "w": w, "h": h},
        "rect_xyxy": {"x_min": x, "y_min": y, "x_max": x+w, "y_max": y+h}
      },
      "reference": { ... },  // solo si se provee reference_shape y reference_bbox
      "scale": {"x": sx, "y": sy},  // solo si hay referencia
      "metrics": {
        "area_px": ...,
        "aspect_ratio": ...,
        "coverage_frac": ...,
        "score": ...
      },
      "metadata": {"path": ..., "time_total_s": ...}
    }

    Parametros
    ----------
    img_shape        : (H, W) de la imagen RAW completa
    bbox_consensus   : (x, y, w, h) del bbox detectado en coordenadas RAW
    bbox_metrics     : dict con metricas del bbox (del pipeline)
    img_path         : ruta del archivo RAW
    time_total_s     : tiempo de procesamiento total
    score            : score del candidato elegido
    reference_shape  : (H_ref, W_ref) de una imagen de referencia (JPEG preview).
                       Si None, la seccion 'reference' y 'scale' se omiten.
    reference_bbox   : (x_ref, y_ref, w_ref, h_ref) del bbox en la referencia.
    """
    import os

    if bbox_consensus is None:
        return {
            "instruction": "No se pudo detectar la botella. Revisar parametros.",
            "crop_optional": True,
            "raw": None,
            "metrics": {},
            "metadata": {"path": img_path, "time_total_s": time_total_s},
        }

    h_raw, w_raw = img_shape
    bx, by, bw, bh = [int(v) for v in bbox_consensus]
    x_min = bx
    y_min = by
    x_max = bx + bw
    y_max = by + bh

    result = {
        "instruction": (
            "Recortar la imagen RAW usando el rectangulo detectado en la silueta de la botella. "
            "Aplicar raw.rect_xyxy sobre la imagen original preservada. "
            "El recorte es opcional: la imagen original NO debe ser modificada."
        ),
        "crop_optional": True,
        "raw": {
            "width":  w_raw,
            "height": h_raw,
            "path":   os.path.basename(img_path) if img_path else "",
            "rect": {"x": x_min, "y": y_min, "w": bw, "h": bh},
            "rect_xyxy": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            },
        },
    }

    # Seccion de referencia (JPEG preview o imagen de baja resolucion)
    if reference_shape is not None and reference_bbox is not None:
        h_ref, w_ref = reference_shape
        rx, ry, rw, rh = [int(v) for v in reference_bbox]

        scale_x = w_raw / w_ref if w_ref > 0 else 1.0
        scale_y = h_raw / h_ref if h_ref > 0 else 1.0

        result["reference"] = {
            "width":  w_ref,
            "height": h_ref,
            "rect": {"x": rx, "y": ry, "w": rw, "h": rh},
            "rect_xyxy": {
                "x_min": rx,
                "y_min": ry,
                "x_max": rx + rw,
                "y_max": ry + rh,
            },
        }
        result["scale"] = {"x": round(scale_x, 8), "y": round(scale_y, 8)}
    else:
        result["scale"] = {"x": 1.0, "y": 1.0}

    result["metrics"] = {
        "area_px":       bbox_metrics.get("area", bw * bh),
        "aspect_ratio":  bbox_metrics.get("aspect_ratio", round(bw / bh, 4) if bh > 0 else 0),
        "coverage_frac": bbox_metrics.get("coverage_frac", round(bw * bh / (w_raw * h_raw), 4)),
        "score":         round(float(score), 4),
    }

    result["metadata"] = {
        "path":         img_path,
        "time_total_s": time_total_s,
    }

    return result


def save_crop_json(crop_json: dict, output_path: str):
    """Guarda el JSON de recorte en disco."""
    import json
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(crop_json, f, ensure_ascii=False, indent=2)
    print(f"  JSON guardado: {output_path}")


def _diagnose_bbox_selection(bbox_final: tuple, bbox_result: dict,
                               silhouette_bbox: tuple,
                               img_shape: tuple) -> dict:
    """
    Genera un diagnostico de la seleccion del bounding box.

    Explica:
      - Que fuente fue usada (silueta, contorno puntuado, componentes)
      - Si el bbox parece razonable (vertical, centrado, tamano apropiado)
      - Posibles riesgos de seleccion incorrecta
      - Por que la version anterior podia fallar con esta imagen
    """
    h, w = img_shape
    warnings  = []
    ok_checks = []

    if bbox_final is None:
        return {
            "ok": False,
            "warnings": ["No se pudo estimar ningún bounding box"],
            "checks": [],
        }

    bx, by, bw_, bh_ = bbox_final
    cx_box = bx + bw_ / 2
    aspect = bw_ / bh_ if bh_ > 0 else 999

    # Check 1: verticalidad
    if aspect < 0.7:
        ok_checks.append("Bbox vertical (aspect ratio correcto para botella)")
    else:
        warnings.append(
            f"Bbox no es vertical (aspect={aspect:.2f}). "
            "Posible seleccion de fondo o plataforma."
        )

    # Check 2: centralidad
    cx_img = w / 2
    offset_frac = abs(cx_box - cx_img) / cx_img
    if offset_frac < 0.30:
        ok_checks.append(f"Bbox centrado (offset={offset_frac:.2%})")
    else:
        warnings.append(
            f"Bbox descentrado (offset={offset_frac:.2%}). "
            "Posible seleccion de estructura lateral del fondo."
        )

    # Check 3: altura significativa
    height_frac = bh_ / h
    if height_frac > 0.30:
        ok_checks.append(f"Bbox de altura significativa ({height_frac:.1%} de la imagen)")
    else:
        warnings.append(
            f"Bbox demasiado bajo ({height_frac:.1%}). "
            "Posible seleccion de plataforma o detalle pequeño."
        )

    # Check 4: no abarca toda la imagen
    width_frac = bw_ / w
    if width_frac < 0.80:
        ok_checks.append(f"Bbox no abarca el fondo completo ({width_frac:.1%} del ancho)")
    else:
        warnings.append(
            f"Bbox muy ancho ({width_frac:.1%} del ancho). "
            "Posible inclusion del fondo. En v2.1 esto ocurria por la union "
            "con la proyeccion. En v2.2 se usa solo el contorno puntuado."
        )

    # Check 5: coherencia silhouette vs matrix
    if silhouette_bbox is not None and bbox_result.get("bbox_contour") is not None:
        sc = silhouette_bbox
        mc = bbox_result["bbox_contour"]
        overlap_x = min(sc[0] + sc[2], mc[0] + mc[2]) - max(sc[0], mc[0])
        overlap_y = min(sc[1] + sc[3], mc[1] + mc[3]) - max(sc[1], mc[1])
        if overlap_x > 0 and overlap_y > 0:
            ok_checks.append("Silueta y contorno puntuado son coherentes entre si")
        else:
            warnings.append(
                "Silueta y contorno puntuado no se solapan. "
                "Posible inconsistencia en la deteccion."
            )

    return {
        "ok":       len(warnings) == 0,
        "warnings": warnings,
        "checks":   ok_checks,
        "bbox_source_used": "silhouette" if silhouette_bbox == bbox_final else "matrix",
    }


# ── METRICAS DE CALIDAD DE BORDE ──────────────────────────────────────────────

def edge_quality_metrics(edges: np.ndarray) -> dict:
    """
    Metricas proxy de calidad de borde (sin referencia de verdad absoluta):
      edge_count    : numero total de pixeles de borde
      edge_density  : proporcion de pixeles de borde sobre total
      edge_continuity: fraccion de pixeles de borde que tienen al menos un vecino
    """
    h, w = edges.shape
    total = h * w
    edge_count = int(np.sum(edges > 0))
    density = edge_count / total

    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D((edges > 0).astype(np.uint8), -1, kernel)
    connected = int(np.sum((edges > 0) & (neighbor_count > 1)))
    continuity = connected / edge_count if edge_count > 0 else 0.0

    return {
        "edge_count":      edge_count,
        "edge_density":    round(density, 6),
        "edge_continuity": round(continuity, 4),
    }


def edge_fragmentation(edges: np.ndarray) -> dict:
    """
    Analiza la fragmentacion de bordes contando componentes conectadas.

    n_components  : numero de componentes conectadas (mas alto = mas fragmentado)
    mean_comp_size: tamano medio de componente en pixeles (mas alto = bordes mas continuos)
    fragmentation : n_components / edge_count (indice normalizado de fragmentacion)

    Interpretacion:
      - Pocas componentes grandes -> bordes continuos, bien cerrados
      - Muchas componentes pequenas -> bordes fragmentados, discontinuos
      - Morfologia closing reduce n_components al unir extremos de borde
      - Morfologia opening puede aumentar n_components al romper bordes finos
    """
    edge_bin = (edges > 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edge_bin, connectivity=8)

    n_components = n_labels - 1  # excluir fondo (label 0)
    edge_count = int(np.sum(edge_bin))

    if n_components > 0:
        comp_sizes = stats[1:, cv2.CC_STAT_AREA]
        mean_size = float(np.mean(comp_sizes))
        fragmentation = n_components / edge_count if edge_count > 0 else 0.0
    else:
        mean_size = 0.0
        fragmentation = 0.0

    return {
        "n_components":   n_components,
        "mean_comp_size": round(mean_size, 2),
        "fragmentation":  round(fragmentation, 6),
    }


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU entre dos mascaras binarias."""
    a = mask_a > 0
    b = mask_b > 0
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return round(float(intersection / union), 4) if union > 0 else 0.0


def iou_with_reference(detection_mask: np.ndarray, reference_mask: np.ndarray) -> dict:
    """
    Compara mascara de deteccion contra imagen de referencia anotada manualmente.

    NOTA: La referencia NO es verdad absoluta -- es una aproximacion visual
    hecha manualmente. Los resultados son indicativos, no definitivos.

    Retorna:
      iou       : Intersection over Union
      precision : TP / (TP + FP) -- fraccion de lo detectado que coincide con referencia
      recall    : TP / (TP + FN) -- fraccion de la referencia que fue detectada
    """
    det = detection_mask > 0
    ref = reference_mask > 0

    tp = int(np.sum(det & ref))
    fp = int(np.sum(det & ~ref))
    fn = int(np.sum(~det & ref))

    iou_val   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp)      if (tp + fp)      > 0 else 0.0
    recall    = tp / (tp + fn)      if (tp + fn)      > 0 else 0.0

    return {
        "iou":       round(float(iou_val), 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
    }


# ── TABLA COMPARATIVA DE METODOS DE BORDE ─────────────────────────────────────

def compare_edge_methods_table(edge_metrics: dict) -> pd.DataFrame:
    """
    Construye tabla comparativa de los 5 metodos de deteccion de bordes.
    edge_metrics: dict retornado por run_full_pipeline_v2()["edge_metrics"].

    Columnas: metodo, edge_count, edge_density, edge_continuity,
              n_components, mean_comp_size, fragmentation, total_edge_length
    """
    rows = []
    for method, m in edge_metrics.items():
        rows.append({
            "metodo":            method,
            "edge_count":        m.get("edge_count", 0),
            "edge_density":      m.get("edge_density", 0),
            "edge_continuity":   m.get("edge_continuity", 0),
            "n_components":      m.get("n_components", 0),
            "mean_comp_size":    m.get("mean_comp_size", 0),
            "fragmentation":     m.get("fragmentation", 0),
            "total_edge_length": m.get("total_edge_length", 0),
        })
    return pd.DataFrame(rows)


def bbox_stability_metrics(pipeline_results: list) -> pd.DataFrame:
    """
    Calcula la estabilidad del bounding box entre multiples imagenes del mismo objeto.

    Una botella rotada deberia producir bboxes similares en area y alto.
    Alta varianza en width puede ser normal (rotacion cambia el perfil lateral).

    pipeline_results: lista de dicts retornados por run_full_pipeline_v2().
                      Cada elemento debe tener 'bbox_consensus' y
                      img_data['name'] (se usa el validation['path'] si no).

    Retorna DataFrame con estadisticas: mean, std, cv (coef. variacion) por metrica.
    """
    rows = []
    for r in pipeline_results:
        bm = r.get("bbox_metrics", {})
        rows.append({
            "nombre":    os.path.basename(r["validation"].get("path", "")),
            "bbox_x":    bm.get("x", 0),
            "bbox_y":    bm.get("y", 0),
            "bbox_w":    bm.get("width", 0),
            "bbox_h":    bm.get("height", 0),
            "bbox_area": bm.get("area", 0),
            "coverage":  bm.get("coverage_frac", 0),
            "aspect":    bm.get("aspect_ratio", 0),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    numeric_cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_area", "coverage", "aspect"]
    stats = []
    for col in numeric_cols:
        if col in df.columns:
            mean_ = df[col].mean()
            std_  = df[col].std()
            cv_   = (std_ / mean_ * 100) if mean_ != 0 else 0.0
            stats.append({
                "metrica": col,
                "mean":    round(mean_, 2),
                "std":     round(std_, 2),
                "cv_pct":  round(cv_, 2),
                "min":     round(df[col].min(), 2),
                "max":     round(df[col].max(), 2),
            })

    return pd.DataFrame(stats)


# ── TABLA RESUMEN ─────────────────────────────────────────────────────────────

def build_summary_table(results: list) -> pd.DataFrame:
    """
    Construye DataFrame resumen con metricas genericas (v2).
    results: lista de dicts con claves 'name', 'baseline', 'improved'.
    """
    rows = []
    metric_keys = [
        "height_px", "width_max_px", "width_25_px", "width_50_px", "width_75_px",
        "area_px", "perimeter_px", "compactness", "solidity", "extent", "aspect_ratio",
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
        bf = edge_fragmentation(r["baseline"]["edges"])
        if_frag = edge_fragmentation(r["improved"]["edges"])

        row["base_edge_density"]    = bq["edge_density"]
        row["impr_edge_density"]    = iq["edge_density"]
        row["base_edge_continuity"] = bq["edge_continuity"]
        row["impr_edge_continuity"] = iq["edge_continuity"]
        row["base_fragmentation"]   = bf["fragmentation"]
        row["impr_fragmentation"]   = if_frag["fragmentation"]
        row["base_n_components"]    = bf["n_components"]
        row["impr_n_components"]    = if_frag["n_components"]

        # IoU entre mascaras baseline vs mejorado
        row["iou_masks"] = mask_iou(r["baseline"]["mask"], r["improved"]["mask"])

        rows.append(row)

    return pd.DataFrame(rows)


def print_stats(df: pd.DataFrame):
    """Estadisticas descriptivas comparativas con metricas genericas v2."""
    measure_cols = [
        "height_px", "width_max_px", "width_50_px",
        "compactness", "solidity", "aspect_ratio",
    ]
    print("\n=== ESTADISTICAS COMPARATIVAS (media +/- std) ===\n")
    print(f"{'Medida':<22} {'Baseline':>18} {'Mejorado':>18} {'Delta%':>10}")
    print("-" * 72)
    for col in measure_cols:
        bc = f"base_{col}"
        ic = f"impr_{col}"
        if bc in df.columns and ic in df.columns:
            bm  = df[bc].mean()
            bs  = df[bc].std()
            im  = df[ic].mean()
            is_ = df[ic].std()
            delta = ((im - bm) / bm * 100) if bm != 0 else 0.0
            print(f"{col:<22} {bm:>8.3f} +/- {bs:<6.3f}   {im:>8.3f} +/- {is_:<6.3f}  {delta:>+8.1f}%")
    print()


# ── GRAFICAS ──────────────────────────────────────────────────────────────────

def plot_comparison_bar(df: pd.DataFrame, output_path: str = None):
    """Barras comparativas de medidas estructurales genericas."""
    metrics = ["height_px", "width_max_px", "width_50_px", "compactness", "solidity"]
    labels  = ["Altura", "Ancho Max", "Ancho 50%", "Compacidad", "Solidez"]

    base_means = [df[f"base_{m}"].mean() for m in metrics]
    impr_means = [df[f"impr_{m}"].mean() for m in metrics]
    base_stds  = [df[f"base_{m}"].std()  for m in metrics]
    impr_stds  = [df[f"impr_{m}"].std()  for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, base_means, width, yerr=base_stds,
           label="Baseline", color="#4c72b0", capsize=5, alpha=0.85)
    ax.bar(x + width / 2, impr_means, width, yerr=impr_stds,
           label="Mejorado", color="#dd8452", capsize=5, alpha=0.85)
    ax.set_xlabel("Medida")
    ax.set_ylabel("Valor")
    ax.set_title("Medidas estructurales: Baseline vs Mejorado\n(media +/- std entre imagenes)")
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
    """Densidad, continuidad, fragmentacion y tiempo de procesamiento."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    specs = [
        (axes[0], "base_edge_density",    "impr_edge_density",    "Densidad de borde",    "px borde / total"),
        (axes[1], "base_edge_continuity", "impr_edge_continuity", "Continuidad de borde", "Fraccion conectados"),
        (axes[2], "base_fragmentation",   "impr_fragmentation",   "Fragmentacion",        "n_comp / edge_count"),
    ]
    for ax, bc, ic, title, ylabel in specs:
        ax.bar(df["imagen"], df[bc], label="Baseline", alpha=0.8, color="#4c72b0")
        ax.bar(df["imagen"], df[ic], label="Mejorado",  alpha=0.6, color="#dd8452")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    ax = axes[3]
    ax.plot(df["imagen"], df["base_time_s"], "o-", label="Baseline", color="#4c72b0")
    ax.plot(df["imagen"], df["impr_time_s"], "s-", label="Mejorado",  color="#dd8452")
    ax.set_title("Tiempo de procesamiento")
    ax.set_ylabel("Segundos")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.suptitle("Metricas de calidad de borde y rendimiento (v2)", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig


def plot_morphology_comparison(morph_results: dict, output_path: str = None):
    """
    Visualiza bordes y metricas para cada operacion morfologica.
    morph_results: retorno de run_morphology_analysis().
    """
    ops = list(morph_results.keys())
    n = len(ops)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, op_name in zip(axes, ops):
        edges = morph_results[op_name]["edges"]
        m = morph_results[op_name]["metrics"]
        ax.imshow(edges, cmap="gray")
        ax.set_title(
            f"{op_name}\n"
            f"dens={m['edge_density']:.4f}  cont={m['edge_continuity']:.3f}\n"
            f"frag={m['fragmentation']:.5f}  comp={m['n_components']}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle("Comparacion de operaciones morfologicas", fontsize=12)
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
    """Overlay de contornos: baseline (azul) y mejorado (naranja)."""
    if gray.ndim == 2:
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        display = (gray / 257).astype(np.uint8) if gray.dtype == np.uint16 else gray.copy()

    if baseline_result["contour"] is not None:
        cv2.drawContours(display, [baseline_result["contour"]], -1, (66, 114, 176), 3)
    if improved_result["contour"] is not None:
        cv2.drawContours(display, [improved_result["contour"]], -1, (221, 132, 82), 3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original (display)")
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


def plot_reference_overlay(display_img: np.ndarray,
                            detection_mask: np.ndarray,
                            reference_mask: np.ndarray,
                            title: str = "",
                            output_path: str = None):
    """
    Visualiza overlay de deteccion vs imagen de referencia anotada.
    Colores:
      Azul  = solo deteccion (FP respecto a referencia)
      Verde = solo referencia (FN de la deteccion)
      Blanco = overlap (TP)
    """
    if display_img.ndim == 2:
        canvas = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB).copy()
    else:
        canvas = display_img.copy()

    canvas[detection_mask > 0] = [66, 114, 176]           # azul: deteccion
    canvas[reference_mask > 0] = [44, 160, 44]            # verde: referencia
    canvas[(detection_mask > 0) & (reference_mask > 0)] = [255, 255, 255]  # blanco: overlap

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_img, cmap="gray")
    axes[0].set_title("Imagen original (display)")
    axes[0].axis("off")
    axes[1].imshow(reference_mask, cmap="gray")
    axes[1].set_title("Referencia anotada")
    axes[1].axis("off")
    axes[2].imshow(canvas)
    axes[2].set_title("Overlay\nAzul=Det  Verde=Ref  Blanco=Overlap")
    axes[2].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"  Guardado: {output_path}")
    return fig
