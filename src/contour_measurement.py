"""
contour_measurement.py
Extraccion de medidas geometricas de la botella desde su contorno / mascara.
"""

import cv2
import numpy as np


def measure_bottle(mask: np.ndarray, contour) -> dict:
    """
    Calcula medidas geometricas de la botella.

    Parametros
    ----------
    mask     : mascara binaria (H, W) uint8
    contour  : contorno principal obtenido con cv2.findContours

    Retorna dict con:
      - height_px        : altura en pixeles (bounding box)
      - width_max_px     : ancho maximo en pixeles
      - width_25_px      : ancho al 25% de la altura
      - width_50_px      : ancho al 50% de la altura (cintura)
      - width_75_px      : ancho al 75% de la altura
      - ratio_w_h        : relacion ancho_max / altura
      - ratio_neck_base  : relacion ancho superior / ancho inferior
      - area_px          : area del contorno en pixeles^2
      - perimeter_px     : perimetro del contorno en pixeles
      - bbox             : (x, y, w, h) del bounding box
    """
    if contour is None or mask is None:
        return _empty_measures()

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0 or w == 0:
        return _empty_measures()

    height_px = h
    width_max_px = w

    # Anchos en alturas relativas usando la mascara real (mas preciso que bounding box)
    width_25 = _width_at_relative_height(mask, y, h, fraction=0.25)
    width_50 = _width_at_relative_height(mask, y, h, fraction=0.50)
    width_75 = _width_at_relative_height(mask, y, h, fraction=0.75)

    # Cuello (zona superior 10%) vs base (zona inferior 10%)
    width_neck = _average_width_in_zone(mask, y, h, frac_start=0.0, frac_end=0.10)
    width_base = _average_width_in_zone(mask, y, h, frac_start=0.85, frac_end=1.00)

    ratio_neck_base = (width_neck / width_base) if width_base > 0 else 0.0
    ratio_w_h = width_max_px / height_px if height_px > 0 else 0.0

    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, closed=True)

    return {
        "height_px": height_px,
        "width_max_px": width_max_px,
        "width_25_px": width_25,
        "width_50_px": width_50,
        "width_75_px": width_75,
        "width_neck_px": width_neck,
        "width_base_px": width_base,
        "ratio_w_h": round(ratio_w_h, 4),
        "ratio_neck_base": round(ratio_neck_base, 4),
        "area_px": area_px,
        "perimeter_px": round(perimeter_px, 2),
        "bbox": (x, y, w, h),
    }


def _width_at_relative_height(mask: np.ndarray, y_top: int, height: int, fraction: float) -> int:
    """
    Mide el ancho de la mascara en una fila especifica (fraccion de la altura total).
    """
    row = int(y_top + fraction * height)
    row = max(0, min(row, mask.shape[0] - 1))
    row_pixels = np.where(mask[row, :] > 0)[0]
    if len(row_pixels) < 2:
        return 0
    return int(row_pixels[-1] - row_pixels[0])


def _average_width_in_zone(mask: np.ndarray, y_top: int, height: int,
                            frac_start: float, frac_end: float) -> float:
    """
    Promedia los anchos de la mascara en una zona vertical (entre frac_start y frac_end).
    """
    row_start = int(y_top + frac_start * height)
    row_end = int(y_top + frac_end * height)
    row_start = max(0, min(row_start, mask.shape[0] - 1))
    row_end = max(0, min(row_end, mask.shape[0] - 1))

    widths = []
    for row in range(row_start, row_end + 1):
        cols = np.where(mask[row, :] > 0)[0]
        if len(cols) >= 2:
            widths.append(int(cols[-1] - cols[0]))

    return round(float(np.mean(widths)), 2) if widths else 0.0


def _empty_measures() -> dict:
    """Retorna medidas vacias cuando no se detecto contorno."""
    return {
        "height_px": 0,
        "width_max_px": 0,
        "width_25_px": 0,
        "width_50_px": 0,
        "width_75_px": 0,
        "width_neck_px": 0.0,
        "width_base_px": 0.0,
        "ratio_w_h": 0.0,
        "ratio_neck_base": 0.0,
        "area_px": 0.0,
        "perimeter_px": 0.0,
        "bbox": (0, 0, 0, 0),
    }
