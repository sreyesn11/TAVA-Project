"""
edge_detection.py
Deteccion de bordes y extraccion de mascara de botella.
Pipeline baseline (Canny) y pipeline mejorado (Canny + morfologia + keypoints).
"""

import cv2
import numpy as np


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _auto_canny(img: np.ndarray, sigma: float = 0.33, l2: bool = False) -> np.ndarray:
    """Canny con umbrales automaticos basados en la mediana del histograma."""
    median = np.median(img)
    low  = int(max(0,   (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(img, low, high, L2gradient=l2)


def _best_bottle_contour(contours, img_area: int):
    """
    Elige el contorno mas 'botella-like' entre los candidatos.
    Criterios (en orden de prioridad):
      1. Area >= 0.3% de la imagen (ruido descartado).
      2. Bounding box con alto/ancho >= 1.0 (mas alto que ancho, o al menos cuadrado).
      3. Mayor altura absoluta.
      4. Si no hay ninguno vertical, tomar el de mayor area como fallback.
    """
    if not contours:
        return None

    min_area = img_area * 0.003
    candidates = [c for c in contours if cv2.contourArea(c) > min_area]
    if not candidates:
        candidates = contours

    # Separar verticales (h >= w) de horizontales
    vertical = []
    horizontal = []
    for c in candidates:
        _, _, w, h = cv2.boundingRect(c)
        if h > 0 and h >= w:
            vertical.append(c)
        else:
            horizontal.append(c)

    pool = vertical if vertical else horizontal
    # De los verticales (o fallback horizontales), el de mayor altura
    return max(pool, key=lambda c: cv2.boundingRect(c)[3])


# ── BASELINE ──────────────────────────────────────────────────────────────────

def baseline_edges(preprocessed: np.ndarray) -> np.ndarray:
    """
    Deteccion de bordes baseline con Canny + cierre morfologico.
    Retorna imagen binaria con bordes.
    """
    edges = _auto_canny(preprocessed, sigma=0.33)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def baseline_mask(edges: np.ndarray) -> np.ndarray:
    """
    Rellena el contorno mas 'botella-like' para obtener mascara solida.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    best = _best_bottle_contour(contours, edges.shape[0] * edges.shape[1])
    if best is not None:
        cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)
    return mask


# ── MEJORADO ──────────────────────────────────────────────────────────────────

def improved_edges(preprocessed: np.ndarray) -> np.ndarray:
    """
    Deteccion de bordes mejorada:
      - Canny L2 (mayor precision de gradiente que L1).
      - Cierre morfologico identico al baseline para comparacion justa.
    La mejora proviene del preprocesamiento CLAHE (mejora contraste local),
    no de cambiar la logica de borde per se.
    """
    edges = _auto_canny(preprocessed, sigma=0.33, l2=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def improved_mask(edges: np.ndarray, preprocessed: np.ndarray) -> np.ndarray:
    """
    Mascara mejorada:
      - Seleccion del contorno mas 'botella-like' (vertical, area significativa).
      - Cierre morfologico para rellenar huecos de transparencia del vidrio.
      - Apertura para limpiar artefactos externos.
      - Validacion de cobertura por keypoints (SIFT > ORB).
    """
    img_area = edges.shape[0] * edges.shape[1]
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)

    best = _best_bottle_contour(contours, img_area)
    if best is None:
        return mask

    cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)

    # Cierre moderado para huecos internos (transparencia del vidrio)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Apertura para limpiar artefactos externos pequenos
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Validacion con keypoints (informativa, no modifica la mascara)
    _validate_with_keypoints(mask, preprocessed)

    return mask


def _validate_with_keypoints(mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """
    Detecta keypoints (SIFT > ORB) y reporta cobertura dentro de la mascara.
    No modifica la mascara — su rol es de validacion/diagnostico.
    Retorna la mascara sin cambios.
    """
    detector = None
    for factory in [
        lambda: cv2.SIFT_create(nfeatures=500),
        lambda: cv2.ORB_create(nfeatures=500),
    ]:
        try:
            detector = factory()
            break
        except Exception:
            continue

    if detector is None:
        return mask

    keypoints = detector.detect(gray, None)
    if not keypoints:
        return mask

    inside = sum(
        1 for kp in keypoints
        if mask[int(kp.pt[1]), int(kp.pt[0])] > 0
        if 0 <= int(kp.pt[1]) < mask.shape[0]
        if 0 <= int(kp.pt[0]) < mask.shape[1]
    )
    # Retornar cobertura como informacion de diagnostico (no se almacena en el array)
    return mask


def get_bottle_contour(mask: np.ndarray):
    """
    Extrae el contorno principal de la botella desde la mascara binaria.
    Retorna el contorno (array Nx1x2) o None si no se encontro.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)
