"""
edge_detection.py  --  v2.2
Deteccion de bordes generica y analisis de operaciones morfologicas.

Historial de cambios:
  v2.0: Eliminado sesgo de objeto especifico (_best_bottle_contour).
  v2.1: Agregado multi-metodo (Sobel, Scharr, Laplaciano, grad. morfologico),
        analisis de proyeccion, scoring de contornos, Hough Lines, bbox estimado.
  v2.2: Correccion critica -- la version anterior selecionaba regiones incorrectas
        (fondo, plataforma, barras verticales) por dos razones:
          1. El consenso tomaba la UNION de bbox_contour + bbox_projection,
             lo que expandia el bbox a incluir bordes del fondo.
          2. El scoring no tenia criterio de simetria bilateral, por lo que
             las barras del fondo podian puntuar alto por aspect ratio.
        Cambios en v2.2:
          - Consenso usa SOLO el contorno puntuado (no union con proyeccion).
          - score_contours() agrega simetria bilateral y penalizaciones especificas.
          - Nueva funcion find_external_silhouette() para detectar la silueta
            externa mediante cierre fuerte + relleno.
          - estimate_bottle_bbox() acepta edges para pasarlos al scoring.
"""

import cv2
import numpy as np


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _auto_canny(img: np.ndarray, sigma: float = 0.33, l2: bool = False) -> np.ndarray:
    """
    Canny con umbrales automaticos basados en la mediana del histograma.
    Requiere uint8 monocanal.
    """
    if img.dtype != np.uint8:
        raise TypeError(
            "_auto_canny requiere uint8. "
            "Convierte con preprocessing.to_uint8_gray() antes de llamar esta funcion."
        )
    median = np.median(img)
    low  = int(max(0,   (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(img, low, high, L2gradient=l2)


def get_significant_contours(edges: np.ndarray, min_bbox_frac: float = 0.0001) -> list:
    """
    Extrae todos los contornos significativos de una imagen de bordes binaria.
    No asume ningun tipo de objeto.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    img_area = edges.shape[0] * edges.shape[1]
    min_area = img_area * min_bbox_frac

    def bbox_area(c):
        _, _, w, h = cv2.boundingRect(c)
        return w * h

    filtered = [c for c in contours if bbox_area(c) >= min_area]
    if not filtered:
        filtered = list(contours)

    return sorted(filtered, key=bbox_area, reverse=True)


def get_primary_contour(contours: list):
    """Retorna el contorno de mayor area. Solo como fallback generico."""
    return contours[0] if contours else None


# ── PIPELINE BASELINE ──────────────────────────────────────────────────────────

def _scaled_kernel_size(base: int, img: np.ndarray, ref_dim: int = 2000) -> int:
    """Escala un tamano de kernel base segun la resolucion de la imagen. Devuelve entero impar."""
    scale = max(img.shape[:2]) / ref_dim
    size = max(base, int(round(base * scale)))
    return size if size % 2 != 0 else size + 1


def baseline_edges(preprocessed: np.ndarray) -> np.ndarray:
    edges = _auto_canny(preprocessed, sigma=0.33)
    ks = _scaled_kernel_size(5, preprocessed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def baseline_mask(edges: np.ndarray) -> np.ndarray:
    contours = get_significant_contours(edges)
    mask = np.zeros_like(edges)
    primary = get_primary_contour(contours)
    if primary is not None:
        cv2.drawContours(mask, [primary], -1, 255, thickness=cv2.FILLED)
    return mask


# ── PIPELINE MEJORADO ──────────────────────────────────────────────────────────

def improved_edges(preprocessed: np.ndarray) -> np.ndarray:
    edges = _auto_canny(preprocessed, sigma=0.33, l2=True)
    ks = _scaled_kernel_size(5, preprocessed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def improved_mask(edges: np.ndarray, preprocessed: np.ndarray = None) -> np.ndarray:
    contours = get_significant_contours(edges)
    mask = np.zeros_like(edges)
    primary = get_primary_contour(contours)
    if primary is None:
        return mask

    cv2.drawContours(mask, [primary], -1, 255, thickness=cv2.FILLED)

    ks_close = _scaled_kernel_size(7, edges)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_close, ks_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    ks_open = _scaled_kernel_size(3, edges)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_open, ks_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return mask


# ── ANALISIS MORFOLOGICO COMPARATIVO ─────────────────────────────────────────

def compare_morphology(edges_raw: np.ndarray, kernel_size: int = 5) -> dict:
    """
    Compara el efecto de diferentes operaciones morfologicas sobre bordes Canny.
    Retorna dict con clave=nombre_operacion, valor=imagen_bordes_resultante.
    """
    k_base  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    k_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size * 2 - 1, kernel_size * 2 - 1)
    )

    return {
        "sin_morfologia": edges_raw.copy(),
        "closing":        cv2.morphologyEx(edges_raw, cv2.MORPH_CLOSE, k_base),
        "opening":        cv2.morphologyEx(edges_raw, cv2.MORPH_OPEN,  k_base),
        "dilation":       cv2.dilate(edges_raw, k_base, iterations=1),
        "closing_large":  cv2.morphologyEx(edges_raw, cv2.MORPH_CLOSE, k_large),
    }


# ── EXTRACCION DE CONTORNO PRINCIPAL ──────────────────────────────────────────

def get_primary_contour_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def get_bottle_contour(mask: np.ndarray):
    """Alias backward-compat v1."""
    return get_primary_contour_from_mask(mask)


# =============================================================================
# v2.1 – DETECCION MULTI-METODO
# =============================================================================

def detect_all_methods(img8: np.ndarray, ksize: int = 3) -> dict:
    """
    Aplica los 5 metodos de deteccion de bordes sobre uint8 gray.
    Retorna dict: metodo -> mapa binario uint8 (0 o 255).
    """
    if img8.dtype != np.uint8:
        raise TypeError("detect_all_methods requiere uint8 monocanal.")
    if img8.ndim != 2:
        raise ValueError("detect_all_methods requiere imagen monocanal (H, W).")

    results = {}

    results["canny"] = _auto_canny(img8, sigma=0.33, l2=True)

    sx = cv2.Sobel(img8, cv2.CV_64F, 1, 0, ksize=ksize)
    sy = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_mag = np.sqrt(sx ** 2 + sy ** 2)
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, results["sobel"] = cv2.threshold(sobel_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scx = cv2.Scharr(img8, cv2.CV_64F, 1, 0)
    scy = cv2.Scharr(img8, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scx ** 2 + scy ** 2)
    scharr_norm = cv2.normalize(scharr_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, results["scharr"] = cv2.threshold(scharr_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lap = cv2.Laplacian(img8, cv2.CV_64F, ksize=ksize)
    lap_norm = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, results["laplacian"] = cv2.threshold(lap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    morph = cv2.morphologyEx(img8, cv2.MORPH_GRADIENT, kernel_m)
    _, results["morph_gradient"] = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return results


def combine_edge_maps(edge_maps: dict, method: str = "weighted_vote") -> np.ndarray:
    """
    Combina multiples mapas de bordes.
    method: 'union' | 'intersection' | 'weighted_vote'
    """
    binary_maps = [(m > 0).astype(np.float32) for m in edge_maps.values()]
    if not binary_maps:
        raise ValueError("edge_maps vacio.")

    if method == "union":
        combined = np.zeros_like(binary_maps[0])
        for b in binary_maps:
            combined = np.maximum(combined, b)
    elif method == "intersection":
        combined = np.ones_like(binary_maps[0])
        for b in binary_maps:
            combined = np.minimum(combined, b)
    elif method == "weighted_vote":
        vote = sum(binary_maps)
        combined = (vote >= len(binary_maps) / 2.0).astype(np.float32)
    else:
        raise ValueError(f"Metodo desconocido: '{method}'.")

    return (combined * 255).astype(np.uint8)


# ── ANALISIS DE PROYECCION ────────────────────────────────────────────────────

def projection_analysis(edges: np.ndarray, threshold_frac: float = 0.05) -> dict:
    """
    Proyecciones ortogonales del mapa de bordes para localizar el objeto.
    Retorna col_projection, row_projection, bbox (informativo -- NO usar como
    fuente primaria del bbox final porque incluye bordes del fondo).
    """
    edge_bin = (edges > 0).astype(np.uint8)
    h, w = edges.shape

    col_proj = edge_bin.sum(axis=0).astype(np.float32)
    row_proj = edge_bin.sum(axis=1).astype(np.float32)

    col_max = float(col_proj.max())
    row_max = float(row_proj.max())

    if col_max == 0 or row_max == 0:
        return {"col_projection": col_proj, "row_projection": row_proj, "bbox": None}

    col_active = np.where(col_proj >= col_max * threshold_frac)[0]
    row_active = np.where(row_proj >= row_max * threshold_frac)[0]

    if len(col_active) == 0 or len(row_active) == 0:
        return {"col_projection": col_proj, "row_projection": row_proj, "bbox": None}

    x1, x2 = int(col_active[0]), int(col_active[-1])
    y1, y2 = int(row_active[0]), int(row_active[-1])

    return {
        "col_projection": col_proj,
        "row_projection": row_proj,
        "bbox": (x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)),
    }


# =============================================================================
# v2.2 – SCORING MEJORADO CON SIMETRIA Y PENALIZACIONES
# =============================================================================

def _measure_symmetry(contour, edges: np.ndarray) -> float:
    """
    Mide la simetria bilateral aproximada de un contorno respecto a su eje vertical.

    Compara la distribucion de bordes activos por fila en la mitad izquierda y
    derecha del bounding box del contorno.

    Objetos simetricos (botellas, envases) tienen distribuciones similares en ambas
    mitades. Las estructuras del fondo (barras verticales laterales, zonas de
    iluminacion asimetrica) tendran diferencias mayores.

    Justificacion: la simetria del contorno es un discriminador potente para objetos
    de manufactura (botella) vs. estructuras de entorno (soporte, fondo iluminado).

    Retorna: 0.0 (muy asimetrico) a 1.0 (perfectamente simetrico).
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w < 6 or h < 6:
        return 0.5

    # Extraer region del bbox en el mapa de bordes
    h_img, w_img = edges.shape[:2]
    y2 = min(y + h, h_img)
    x2 = min(x + w, w_img)
    region = (edges[y:y2, x:x2] > 0).astype(np.float32)

    actual_w = region.shape[1]
    if actual_w < 4:
        return 0.5

    mid = actual_w // 2
    left  = region[:, :mid]
    right = region[:, actual_w - mid:][:, ::-1]  # flip horizontal

    min_w = min(left.shape[1], right.shape[1])
    if min_w == 0:
        return 0.5

    left  = left[:, :min_w]
    right = right[:, :min_w]

    left_rows  = left.sum(axis=1)
    right_rows = right.sum(axis=1)

    total = left_rows.sum() + right_rows.sum()
    if total == 0:
        return 0.5

    diff = np.abs(left_rows - right_rows).sum()
    symmetry = 1.0 - (diff / total)

    return float(max(0.0, min(1.0, symmetry)))


def score_contours(contours: list, img_shape: tuple,
                   edges: np.ndarray = None,
                   weights: dict = None) -> list:
    """
    Puntua cada contorno con criterios geometricos orientados a la botella.

    CAMBIOS v2.2 vs v2.1:
      - Nuevo criterio: simetria bilateral (peso 0.15).
        Las botellas son simetricas; el fondo o las barras laterales no lo son
        respecto al eje central de la botella.
      - Penalizaciones explicitas:
          a) Plataforma: contorno cuya base cae en el 15% inferior de la imagen
             Y es mas ancho que alto → muy probablemente es la mesa/base circular.
          b) Fondo estructural: ancho > 70% del ancho de imagen Y alto < 35%
             de la imagen → franjas de fondo o iluminacion.
          c) Muy pequeño: alto < 5% de la imagen → detalles internos, texto.
      - Pesos ajustados: centralidad sube a 0.25 (refuerza que la botella
        esta en el centro), simetria = 0.15.

    Criterios finales con pesos por defecto:
      height      : 0.25  ocupa parte significativa de la altura de imagen
      centrality  : 0.25  esta aproximadamente en el centro del encuadre
      aspect_ratio: 0.20  mas alto que ancho (vertical)
      area        : 0.10  tamano relativo normalizado
      continuity  : 0.05  continuidad del borde respecto al bbox
      symmetry    : 0.15  simetria bilateral del contorno (NUEVO)

    edges: mapa de bordes uint8. Requerido para calcular simetria.
           Si None, la simetria se fija en 0.5 (neutral).
    """
    if not contours:
        return []

    h_img, w_img = img_shape[:2]
    img_cx   = w_img / 2.0
    img_cy   = h_img / 2.0
    img_area = float(h_img * w_img)
    max_dist = np.sqrt(img_cx ** 2 + img_cy ** 2)

    if weights is None:
        weights = {
            "height":       0.25,
            "centrality":   0.25,
            "aspect_ratio": 0.20,
            "area":         0.10,
            "continuity":   0.05,
            "symmetry":     0.15,
        }

    scored = []
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)

        if cw == 0 or ch == 0:
            scored.append((0.0, c))
            continue

        area      = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))

        # 1. Altura normalizada
        height_score = min(ch / h_img, 1.0)

        # 2. Centralidad respecto al centro de la imagen
        cx_center = cx + cw / 2.0
        cy_center = cy + ch / 2.0
        dist = np.sqrt((cx_center - img_cx) ** 2 + (cy_center - img_cy) ** 2)
        centrality_score = 1.0 - (dist / max_dist) if max_dist > 0 else 0.0

        # 3. Aspect ratio: favorece h > w (botella vertical)
        aspect_score = 1.0 / (1.0 + (cw / ch))

        # 4. Area relativa con raiz cuadrada para atenuar sesgo
        area_score = min(np.sqrt(area / img_area) * 5.0, 1.0)

        # 5. Continuidad: perimetro del contorno vs perimetro del bbox
        bbox_perimeter = 2.0 * (cw + ch)
        continuity_score = min(perimeter / bbox_perimeter, 1.0) if bbox_perimeter > 0 else 0.0

        # 6. Simetria bilateral (NUEVO v2.2)
        symmetry_score = 0.5  # neutral si no hay mapa de bordes
        if edges is not None:
            symmetry_score = _measure_symmetry(c, edges)

        # Score base ponderado
        score = (
            weights["height"]       * height_score +
            weights["centrality"]   * centrality_score +
            weights["aspect_ratio"] * aspect_score +
            weights["area"]         * area_score +
            weights["continuity"]   * continuity_score +
            weights["symmetry"]     * symmetry_score
        )

        # --- PENALIZACIONES (v2.2) ---

        # a) Plataforma / mesa: base en el 15% inferior Y mas ancho que alto
        c_bottom = cy + ch
        if c_bottom >= h_img * 0.85 and cw >= ch:
            score *= 0.25

        # b) Franja de fondo: cubre >70% del ancho pero <35% del alto
        if cw >= w_img * 0.70 and ch < h_img * 0.35:
            score *= 0.30

        # c) Componente muy pequeño: alto <5% de la imagen
        if ch < h_img * 0.05:
            score *= 0.10

        scored.append((score, c))

    return sorted(scored, key=lambda x: x[0], reverse=True)


# ── HOUGH LINES VERTICALES ────────────────────────────────────────────────────

def detect_vertical_lines_hough(edges: np.ndarray,
                                  angle_tolerance_deg: float = 15.0,
                                  hough_threshold: int = None) -> dict:
    """
    Detecta lineas verticales via Hough. Se usa para REFINAR los bordes
    laterales del bbox, no para definirlo. Solo se aplica si las lineas
    detectadas estan dentro del rango horizontal del contorno principal.
    """
    h, w = edges.shape

    if hough_threshold is None:
        hough_threshold = max(50, int(h * 0.05))

    lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=hough_threshold)

    if lines is None:
        return {"lines": None, "x_left": None, "x_right": None, "bbox": None}

    tol_rad = np.radians(angle_tolerance_deg)
    vertical_xs = []

    for line in lines:
        rho, theta = line[0]
        if theta <= tol_rad or theta >= (np.pi - tol_rad):
            cos_t = np.cos(theta)
            if abs(cos_t) > 1e-4:
                x = int(rho / cos_t)
                if 0 <= x < w:
                    vertical_xs.append(x)

    if not vertical_xs:
        return {"lines": lines, "x_left": None, "x_right": None, "bbox": None}

    x_left  = int(np.percentile(vertical_xs, 10))
    x_right = int(np.percentile(vertical_xs, 90))

    if x_right <= x_left:
        x_left  = min(vertical_xs)
        x_right = max(vertical_xs)

    return {
        "lines":   lines,
        "x_left":  x_left,
        "x_right": x_right,
        "bbox":    (x_left, 0, max(1, x_right - x_left), h),
    }


# =============================================================================
# v2.2 – SILUETA EXTERNA Y BBOX CORREGIDO
# =============================================================================

def find_external_silhouette(edges: np.ndarray, img_shape: tuple,
                               close_factor: float = 0.012) -> tuple:
    """
    Detecta la silueta externa del objeto principal mediante cierre fuerte + relleno.

    Justificacion:
      Los bordes detectados por Canny/Sobel incluyen tanto el contorno externo de la
      botella como bordes internos (texto, relieve, reflejos). Al aplicar un cierre
      morfologico con kernel grande, se unen los fragmentos del contorno externo
      formando una region cerrada. Al rellenarla, se obtiene la mascara del objeto
      completo. La silueta externa de esa mascara es el contorno que buscamos.

      Este enfoque prioriza la forma global del objeto por sobre los detalles internos,
      que quedan "tragados" por el relleno.

    close_factor: tamano del kernel de cierre como fraccion de la dimension maxima.
                  Valores tipicos: 0.01 - 0.02. Mayor → mas conexion de fragmentos.

    Retorna:
      silhouette_contour : contorno externo de la silueta (o None)
      silhouette_mask    : mascara binaria (H, W) uint8 del objeto relleno
    """
    h, w = img_shape[:2]

    # Kernel de cierre escala con la resolucion
    ks = max(5, int(max(h, w) * close_factor))
    if ks % 2 == 0:
        ks += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

    # Cierre agresivo: une bordes fragmentados del contorno externo
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Buscar contornos externos de las regiones cerradas
    contours_closed, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours_closed:
        return None, np.zeros((h, w), dtype=np.uint8)

    # Puntuar con el scoring mejorado (incluye simetria y penalizaciones)
    scored = score_contours(contours_closed, img_shape, edges=edges)
    if not scored:
        return None, np.zeros((h, w), dtype=np.uint8)

    best_contour = scored[0][1]

    # Rellenar la region del mejor contorno → mascara solida del objeto
    silhouette_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(silhouette_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    # Extraer el contorno externo de la mascara rellena → silueta limpia
    ext_contours, _ = cv2.findContours(
        silhouette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not ext_contours:
        return best_contour, silhouette_mask

    silhouette_contour = max(ext_contours, key=cv2.contourArea)
    return silhouette_contour, silhouette_mask


def estimate_bottle_bbox(edges: np.ndarray, img_shape: tuple,
                          use_hough: bool = True,
                          min_height_frac: float = 0.15,
                          min_area_frac: float = 0.001) -> dict:
    """
    Estima el bounding box del objeto principal combinando 4 fuentes matriciales.

    CAMBIO CRITICO v2.2:
      La version anterior tomaba la UNION de bbox_contour + bbox_projection como
      consensus. Esto era incorrecto porque la proyeccion acumula todos los bordes
      de la imagen (incluyendo fondo, barras verticales, plataforma), lo que
      expandia el bbox mas alla de la botella.

      Ahora el consensus usa SOLO el bbox del contorno puntuado:
        - bbox_contour es el resultado de score_contours() con 6 criterios
        - Se puede refinar LATERALMENTE con Hough Lines, pero solo si las
          lineas detectadas estan dentro del rango horizontal del contorno
          (evita que barras del fondo expandan el bbox)
        - La proyeccion se conserva como informacion de diagnostico

    La jerarquia de fallback:
      bbox_contour > bbox_components > None

    Parametros
    ----------
    edges           : mapa de bordes binario uint8 (imagen completa, post-cierre)
    img_shape       : (H, W, ...) de la imagen original
    use_hough       : si True, usa Hough para refinar bordes laterales (conservador)
    min_height_frac : altura minima de componentes candidatos
    min_area_frac   : area minima de componentes candidatos
    """
    h, w = img_shape[:2]
    img_area = float(h * w)
    results = {}

    # -- 1. Proyeccion (solo diagnostico, NO define el consensus) ------------
    proj = projection_analysis(edges)
    results["bbox_projection"] = proj["bbox"]
    results["col_projection"]  = proj["col_projection"]
    results["row_projection"]  = proj["row_projection"]

    # -- 2. Componentes conectados (filtro por altura y area minima) ---------
    edge_bin = (edges > 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(edge_bin, connectivity=8)

    candidates = []
    for i in range(1, n_labels):
        cx_   = int(stats[i, cv2.CC_STAT_LEFT])
        cy_   = int(stats[i, cv2.CC_STAT_TOP])
        cw_   = int(stats[i, cv2.CC_STAT_WIDTH])
        ch_   = int(stats[i, cv2.CC_STAT_HEIGHT])
        area_ = int(stats[i, cv2.CC_STAT_AREA])

        if area_ >= img_area * min_area_frac and ch_ >= h * min_height_frac:
            candidates.append((cx_, cy_, cw_, ch_, area_))

    if candidates:
        x1_all = [c[0]         for c in candidates]
        y1_all = [c[1]         for c in candidates]
        x2_all = [c[0] + c[2]  for c in candidates]
        y2_all = [c[1] + c[3]  for c in candidates]
        results["bbox_components"] = (
            min(x1_all), min(y1_all),
            max(x2_all) - min(x1_all),
            max(y2_all) - min(y1_all),
        )
        results["n_candidates"] = len(candidates)
    else:
        results["bbox_components"] = None
        results["n_candidates"]    = 0

    # -- 3. Contornos puntuados (con simetria y penalizaciones v2.2) ---------
    contours = get_significant_contours(edges)
    scored   = score_contours(contours, img_shape, edges=edges)  # pasa edges para simetria

    results["scored_contours"] = scored

    if scored:
        best_c = scored[0][1]
        bx, by, bw_, bh_ = cv2.boundingRect(best_c)
        results["bbox_contour"]  = (bx, by, bw_, bh_)
        results["best_contour"]  = best_c
        results["contour_score"] = scored[0][0]
    else:
        results["bbox_contour"]  = None
        results["best_contour"]  = None
        results["contour_score"] = 0.0

    # -- 4. Hough Lines (refinamiento conservador, no definicion) ------------
    if use_hough:
        hough = detect_vertical_lines_hough(edges)
        results["hough"]      = hough
        results["bbox_hough"] = hough.get("bbox")
    else:
        results["hough"]      = None
        results["bbox_hough"] = None

    # -- 5. Consensus CORREGIDO: usar SOLO contorno puntuado ----------------
    # NO tomar union con proyeccion → eso expandia a bordes del fondo.
    bbox_c = results["bbox_contour"]
    bbox_k = results["bbox_components"]

    if bbox_c is None and bbox_k is None:
        results["bbox_consensus"] = None

    elif bbox_c is not None:
        consensus = bbox_c

        # Refinamiento CONSERVADOR con Hough: solo si las lineas estan
        # dentro del rango horizontal del contorno (+/- 15% del ancho).
        if use_hough and results["bbox_hough"] is not None:
            hx, _, hw, _ = results["bbox_hough"]
            bx, by, bw_, bh_ = bbox_c
            margin = w * 0.15

            h_xl = hx
            h_xr = hx + hw
            c_xl = bx
            c_xr = bx + bw_

            if (c_xl - margin) <= h_xl and h_xr <= (c_xr + margin):
                new_x1 = min(c_xl, h_xl)
                new_x2 = max(c_xr, h_xr)
                consensus = (int(new_x1), by, max(1, int(new_x2 - new_x1)), bh_)

        results["bbox_consensus"] = consensus

    else:
        # Fallback: usar componentes si no hay contorno puntuado
        results["bbox_consensus"] = bbox_k

    return results


# ── RECORTE POST-PROCESAMIENTO ─────────────────────────────────────────────────

def crop_from_bbox(img: np.ndarray, bbox: tuple,
                   padding_frac: float = 0.02) -> np.ndarray:
    """
    Recorta la imagen usando coordenadas calculadas DESDE la matriz de bordes.
    Solo debe llamarse DESPUES de estimate_bottle_bbox().
    """
    if bbox is None:
        return img.copy()

    h, w = img.shape[:2]
    x, y, bw, bh = bbox

    pad_x = int(w * padding_frac)
    pad_y = int(h * padding_frac)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    return img[y1:y2, x1:x2].copy()
