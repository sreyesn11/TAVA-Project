"""
preprocessing.py
Preprocesamiento de imagenes para deteccion de botellas.
Pipeline baseline y pipeline mejorado.
"""

import cv2
import numpy as np


# ── BASELINE ──────────────────────────────────────────────────────────────────

def baseline_preprocess(gray: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento baseline:
      1. Suavizado Gaussiano estandar.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return blurred


# ── MEJORADO ──────────────────────────────────────────────────────────────────

def improved_preprocess(gray: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento mejorado:
      1. CLAHE moderado para contraste local adaptativo.
      2. Supresion de reflejos brillantes (hot-spots).
      3. Suavizado Gaussiano (mantiene coherencia global para Canny).
      4. Retorna tambien la version bilateral (para analisis de bordes finos).
    """
    # CLAHE con clip moderado (no excesivo para no amplificar ruido de fondo)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # Supresion de reflejos: limitar pixeles muy brillantes
    p99 = np.percentile(enhanced, 99)
    p95 = np.percentile(enhanced, 95)
    if p99 > p95:
        enhanced = np.clip(enhanced, 0, int(p99))
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Suavizado Gaussiano robusto para que Canny encuentre bordes estructurales
    # (no texturas finas del fondo)
    smoothed = cv2.GaussianBlur(enhanced, (7, 7), 2.0)

    return smoothed


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normaliza la imagen al rango [0, 255] uint8."""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img
