"""
contour_measurement.py  --  v2.2
Metricas geometricas de estructuras detectadas.

v2.0: Eliminadas metricas especificas de botella (width_neck, width_base).
v2.2: Agrega measure_bottle_in_crop() con perfil completo de anchos por zona
      (cuello, hombro, cuerpo, base) calculado desde la silueta dentro del recorte.
"""

import cv2
import numpy as np


# ── MEDIDAS GENERICAS (mask + contour) ────────────────────────────────────────

def measure_structure(mask: np.ndarray, contour) -> dict:
    """
    Calcula metricas geometricas genericas de una estructura detectada.

    Retorna dict con:
      height_px    : altura del bounding box (px)
      width_max_px : ancho del bounding box (px)
      width_25_px  : ancho de la mascara al 25% de la altura
      width_50_px  : ancho de la mascara al 50% de la altura
      width_75_px  : ancho de la mascara al 75% de la altura
      area_px      : area del contorno (px^2)
      perimeter_px : perimetro del contorno (px)
      compactness  : 4*pi*area / perimetro^2  (1.0 = circulo)
      solidity     : area / area_convex_hull
      extent       : area / area_bounding_box
      aspect_ratio : ancho / alto
      bbox         : (x, y, w, h) en coordenadas de la mascara completa
    """
    if contour is None or mask is None:
        return _empty_measures()

    x, y, w, h = cv2.boundingRect(contour)
    if h == 0 or w == 0:
        return _empty_measures()

    area_px      = float(cv2.contourArea(contour))
    perimeter_px = float(cv2.arcLength(contour, closed=True))

    compactness = (4.0 * np.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0

    hull      = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity  = (area_px / hull_area) if hull_area > 0 else 0.0

    bbox_area = float(w * h)
    extent    = (area_px / bbox_area) if bbox_area > 0 else 0.0

    width_25 = _width_at_relative_height(mask, y, h, 0.25)
    width_50 = _width_at_relative_height(mask, y, h, 0.50)
    width_75 = _width_at_relative_height(mask, y, h, 0.75)

    return {
        "height_px":    h,
        "width_max_px": w,
        "width_25_px":  width_25,
        "width_50_px":  width_50,
        "width_75_px":  width_75,
        "area_px":      round(area_px, 1),
        "perimeter_px": round(perimeter_px, 2),
        "compactness":  round(compactness, 4),
        "solidity":     round(solidity, 4),
        "extent":       round(extent, 4),
        "aspect_ratio": round(w / h, 4) if h > 0 else 0.0,
        "bbox":         (x, y, w, h),
    }


def measure_bottle(mask: np.ndarray, contour) -> dict:
    """Alias para compatibilidad con codigo v1."""
    return measure_structure(mask, contour)


# ── MEDIDAS DETALLADAS DENTRO DEL RECORTE ─────────────────────────────────────

def measure_bottle_in_crop(silhouette_mask_crop: np.ndarray) -> dict:
    """
    Calcula medidas detalladas de la botella operando DENTRO del recorte.

    Recibe la silueta binaria ya recortada al bbox de la botella.
    Todas las coordenadas y dimensiones son relativas al recorte.

    ZONAS DE MEDICION:
      cuello  (top   0 – 20% de la altura): zona mas estrecha superior
      hombro  (top  20 – 35%):              transicion cuello-cuerpo
      cuerpo  (top  35 – 80%):              zona central mas ancha
      base    (top  80 – 100%):             zona inferior de la botella

    Para cada zona se calcula el ancho promedio, maximo y minimo.

    Adicionalmente:
      - perfil_anchos: lista de anchos en cada fila de la silueta
      - ancho_max, ancho_min, ancho_medio: sobre toda la botella
      - simetria_vertical: similitud entre mitad izquierda y derecha del perfil

    Parametros
    ----------
    silhouette_mask_crop : np.ndarray uint8 (H_crop, W_crop) con la mascara
                           binaria de la silueta YA recortada al bbox.
                           Coordenadas relativas al recorte (origen en [0,0]).

    Retorna dict con medidas en pixeles relativas al recorte.
    """
    if silhouette_mask_crop is None or silhouette_mask_crop.size == 0:
        return _empty_crop_measures()

    h_crop, w_crop = silhouette_mask_crop.shape[:2]
    if h_crop == 0 or w_crop == 0:
        return _empty_crop_measures()

    # Perfil de anchos: ancho real de la silueta en cada fila
    profile = _compute_width_profile(silhouette_mask_crop)

    # Zona activa: filas donde la silueta tiene pixeles
    active_rows = [i for i, pw in enumerate(profile) if pw > 0]
    if not active_rows:
        return _empty_crop_measures()

    y_start = active_rows[0]
    y_end   = active_rows[-1]
    height  = y_end - y_start + 1

    if height == 0:
        return _empty_crop_measures()

    active_widths = [profile[r] for r in active_rows]

    # Metricas globales sobre toda la botella
    ancho_max   = int(max(active_widths))
    ancho_min   = int(min(w for w in active_widths if w > 0))
    ancho_medio = round(float(np.mean(active_widths)), 2)

    # Maximo ancho global (desde el bbox del contorno)
    contours, _ = cv2.findContours(
        silhouette_mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return _empty_crop_measures()

    main_contour = max(contours, key=cv2.contourArea)
    cx, cy, cw_, ch_ = cv2.boundingRect(main_contour)

    # Medidas por zona (relativas a la extension vertical activa de la botella)
    def zone_stats(frac_start: float, frac_end: float) -> dict:
        r0 = y_start + int(height * frac_start)
        r1 = y_start + int(height * frac_end)
        zone_widths = [profile[r] for r in range(r0, min(r1 + 1, h_crop)) if profile[r] > 0]
        if not zone_widths:
            return {"promedio_px": 0, "maximo_px": 0, "minimo_px": 0}
        return {
            "promedio_px": round(float(np.mean(zone_widths)), 2),
            "maximo_px":   int(max(zone_widths)),
            "minimo_px":   int(min(zone_widths)),
        }

    zona_cuello  = zone_stats(0.00, 0.20)  # cuello / tapa
    zona_hombro  = zone_stats(0.20, 0.35)  # hombro (transicion)
    zona_cuerpo  = zone_stats(0.35, 0.80)  # cuerpo principal
    zona_base    = zone_stats(0.80, 1.00)  # base

    # Anchos en fracciones exactas (para comparacion entre imagenes)
    anchos_frac = {}
    for frac in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]:
        r = y_start + int(height * frac)
        r = max(0, min(r, h_crop - 1))
        anchos_frac[f"ancho_{int(frac*100):02d}pct_px"] = profile[r]

    # Simetria vertical: comparar perfil izquierdo vs derecho respecto al eje central
    simetria = _symmetry_from_profile(silhouette_mask_crop, active_rows)

    # Relacion cuello/cuerpo (indicador de forma de botella)
    ratio_cuello_cuerpo = (
        round(zona_cuello["maximo_px"] / zona_cuerpo["maximo_px"], 4)
        if zona_cuerpo["maximo_px"] > 0 else 0.0
    )

    return {
        # Dimensiones globales del recorte
        "height_crop_px":    h_crop,
        "width_crop_px":     w_crop,
        # Altura y ancho de la botella dentro del recorte
        "height_bottle_px":  height,
        "width_max_px":      ancho_max,
        "width_min_px":      ancho_min,
        "width_mean_px":     ancho_medio,
        "aspect_ratio":      round(ancho_max / height, 4) if height > 0 else 0.0,
        # Medidas por zona
        "zona_cuello":       zona_cuello,
        "zona_hombro":       zona_hombro,
        "zona_cuerpo":       zona_cuerpo,
        "zona_base":         zona_base,
        # Anchos en fracciones clave
        **anchos_frac,
        # Metricas de forma
        "ratio_cuello_cuerpo": ratio_cuello_cuerpo,
        "simetria_vertical":   round(simetria, 4),
        # Referencia del contorno en el recorte
        "contour_bbox_in_crop": (int(cx), int(cy), int(cw_), int(ch_)),
    }


def _compute_width_profile(mask: np.ndarray) -> list:
    """
    Calcula el ancho de la silueta en cada fila de la mascara.
    Ancho = distancia entre el primer y ultimo pixel activo en esa fila.
    Retorna lista de longitud H con el ancho en px de cada fila.
    """
    h = mask.shape[0]
    profile = []
    for row_idx in range(h):
        row = mask[row_idx, :]
        active = np.where(row > 0)[0]
        if len(active) >= 2:
            profile.append(int(active[-1] - active[0]) + 1)
        elif len(active) == 1:
            profile.append(1)
        else:
            profile.append(0)
    return profile


def _symmetry_from_profile(mask: np.ndarray, active_rows: list) -> float:
    """
    Mide la simetria bilateral de la silueta respecto al eje vertical central.
    Compara las columnas activas izquierda y derecha del centro en cada fila.
    """
    if not active_rows:
        return 0.5

    h, w = mask.shape[:2]
    mid   = w // 2
    total = 0.0
    match = 0.0

    for r in active_rows:
        row = mask[r, :]
        active = np.where(row > 0)[0]
        if len(active) < 2:
            continue

        x_left  = active[0]
        x_right = active[-1]

        dist_left  = mid - x_left
        dist_right = x_right - mid

        if dist_left <= 0 or dist_right <= 0:
            continue

        pair_min = min(dist_left, dist_right)
        pair_max = max(dist_left, dist_right)

        total += 1
        match += pair_min / pair_max

    return float(match / total) if total > 0 else 0.5


def _width_at_relative_height(mask: np.ndarray, y_top: int, height: int,
                               fraction: float) -> int:
    row = int(y_top + fraction * height)
    row = max(0, min(row, mask.shape[0] - 1))
    row_pixels = np.where(mask[row, :] > 0)[0]
    if len(row_pixels) < 2:
        return 0
    return int(row_pixels[-1] - row_pixels[0])


def _empty_measures() -> dict:
    return {
        "height_px":    0, "width_max_px": 0,
        "width_25_px":  0, "width_50_px":  0, "width_75_px": 0,
        "area_px":      0.0, "perimeter_px": 0.0,
        "compactness":  0.0, "solidity":     0.0,
        "extent":       0.0, "aspect_ratio": 0.0,
        "bbox":         (0, 0, 0, 0),
    }


def _empty_crop_measures() -> dict:
    return {
        "height_crop_px":       0, "width_crop_px":    0,
        "height_bottle_px":     0, "width_max_px":     0,
        "width_min_px":         0, "width_mean_px":    0.0,
        "aspect_ratio":         0.0,
        "zona_cuello":          {"promedio_px": 0, "maximo_px": 0, "minimo_px": 0},
        "zona_hombro":          {"promedio_px": 0, "maximo_px": 0, "minimo_px": 0},
        "zona_cuerpo":          {"promedio_px": 0, "maximo_px": 0, "minimo_px": 0},
        "zona_base":            {"promedio_px": 0, "maximo_px": 0, "minimo_px": 0},
        "ratio_cuello_cuerpo":  0.0,
        "simetria_vertical":    0.0,
        "contour_bbox_in_crop": (0, 0, 0, 0),
    }
