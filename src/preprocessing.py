"""
preprocessing.py  --  v2
Preprocesamiento generico para deteccion de bordes y analisis estructural en imagenes RAW.
Soporta uint8 y uint16. Normalizacion a uint8 SOLO cuando la requiere un algoritmo especifico.
Sin supuestos de objeto especifico (eliminado sesgo de botella).
"""

import cv2
import numpy as np


# ── CONVERSION SEGURA (no destructiva para la imagen original) ─────────────────

def to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """
    Convierte imagen a uint8 escala de grises UNICAMENTE para uso en algoritmos
    que no soportan uint16 (e.g. cv2.Canny, cv2.GaussianBlur con uint16).
    La imagen original NO se modifica -- esta funcion devuelve una copia.

    Mapeo de profundidad:
      uint16 -> /257 -> uint8   (65535/257=255, sin recorte de rango)
      float  -> min-max lineal -> uint8
      uint8  -> copia directa
    """
    if img.dtype == np.uint16:
        img8 = (img / 257).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        mn, mx = float(img.min()), float(img.max())
        img8 = ((img - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(img, dtype=np.uint8)
    else:
        img8 = img.astype(np.uint8) if img.dtype != np.uint8 else img.copy()

    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
    return img8


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convierte a escala de grises PRESERVANDO el dtype original (uint8 o uint16).

    Para uint16 usa pesos de luminosidad Rec.601:
      Y = 0.299*R + 0.587*G + 0.114*B
    """
    if img.ndim == 2:
        return img.copy()
    if img.dtype == np.uint16:
        gray = (
            0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)
        ).astype(np.uint16)
        return gray
    # uint8 u otros: usar cvtColor
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _scaled_kernel(base: int, img: np.ndarray, ref_dim: int = 2000) -> tuple:
    """
    Escala un tamano de kernel base segun la resolucion de la imagen.
    Referencia: 2000px de dimension maxima.
    Devuelve siempre un numero impar.
    """
    scale = max(img.shape[:2]) / ref_dim
    size = max(base, int(round(base * scale)))
    if size % 2 == 0:
        size += 1
    return (size, size)


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normaliza imagen al rango [0, 255] uint8 SOLO para visualizacion.
    No usar en el pipeline de procesamiento cuantitativo.
    """
    if img.dtype == np.uint8:
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (
        cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        .astype(np.uint8)
    )


# Alias para compatibilidad con v1
def normalize_image(img: np.ndarray) -> np.ndarray:
    return normalize_for_display(img)


# ── PIPELINE BASELINE ──────────────────────────────────────────────────────────

def baseline_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento baseline: suavizado Gaussiano estandar con kernel fijo (5x5).

    Kernel fijo intencional: el baseline representa el pipeline simple de v1.
    En imagenes de alta resolucion (>2MP) este suavizado es muy local y puede
    ser insuficiente para eliminar ruido de texturas finas -- esto es una limitacion
    documentada del baseline que el pipeline mejorado corrige.

    Acepta uint8 o uint16. Retorna uint8 gray.
    """
    gray8 = to_uint8_gray(img)
    return cv2.GaussianBlur(gray8, (5, 5), 1.0)


# ── PIPELINE MEJORADO ──────────────────────────────────────────────────────────

def improved_preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento mejorado:
      1. Conversion a gray uint8 (solo para algoritmos -- no destructiva)
      2. CLAHE moderado: mejora contraste local adaptativo sin amplificar ruido global
      3. Filtro bilateral: suaviza preservando bordes estructurales

    Cambios vs v1:
      - Eliminada la supresion de reflejos (sesgada para vidrio/botellas)
      - Filtro bilateral en lugar de Gaussiano (preserva mejor los bordes)

    Retorna uint8 gray.
    """
    gray8 = to_uint8_gray(img)

    # CLAHE: ecualizacion local de histograma con limite de amplificacion
    # tileGridSize se escala con la resolucion: ventanas del ~0.8% de la dimension
    tile = max(8, int(max(img.shape[:2]) / 250))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
    enhanced = clahe.apply(gray8)

    # Filtro bilateral: preserva bordes al suavizar
    # d se escala con la resolucion para mantener el mismo radio perceptual
    d = max(9, int(max(img.shape[:2]) / 700))
    if d % 2 == 0:
        d += 1
    smoothed = cv2.bilateralFilter(enhanced, d=d, sigmaColor=75, sigmaSpace=75)

    return smoothed


# ── CLAHE NATIVO uint16 ────────────────────────────────────────────────────────

def preprocess_uint16_clahe(img_gray16: np.ndarray,
                             clip_limit: float = 2.0,
                             tile_size: int = 16) -> np.ndarray:
    """
    CLAHE aplicado directamente sobre imagen uint16, sin conversion a uint8.
    cv2.createCLAHE soporta uint16 nativo.

    Util para analisis cuantitativo donde se requiere conservar la profundidad
    de 16 bits durante el procesamiento (e.g. comparativa uint8 vs uint16).

    Parametros
    ----------
    img_gray16 : array uint16 monocanal (H, W)
    clip_limit : limite de amplificacion del contraste
    tile_size  : tamano de ventana adaptativa (NxN)

    Retorna uint16.
    """
    if img_gray16.ndim == 3:
        raise ValueError("preprocess_uint16_clahe requiere imagen monocanal (H, W).")
    if img_gray16.dtype != np.uint16:
        raise TypeError(f"Se esperaba uint16, se recibio {img_gray16.dtype}.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_gray16)
