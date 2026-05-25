"""
signature.py
Extraccion automatica de firma geometrica radial desde mascara de botella.
Comparacion contra firma de referencia (firma_botella/firma_completa_0_360.csv).

Convencion de angulos:
  0   deg -> derecha  (+X)
  90  deg -> arriba   (-Y en coordenadas imagen, eje Y invertido)
  180 deg -> izquierda(-X)
  270 deg -> abajo    (+Y en coordenadas imagen)

Normalizacion px vs mm:
  La firma detectada esta en pixeles y la de referencia en milimetros.
  Para comparar la FORMA (no la escala absoluta), ambas se dividen por su
  valor maximo antes de calcular las metricas de error.
  Si se conoce un factor de conversion px->mm, puede pasarse como argumento
  a compare_with_reference() para obtener metricas en mm reales.
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# ── RELLENO DE MASCARA ────────────────────────────────────────────────────────

def _fill_mask(mask: np.ndarray) -> np.ndarray:
    """
    Rellena huecos internos en una mascara binaria.
    Aplica cierre morfologico + flood-fill desde el exterior para obtener
    una mascara solida robusta a bordes discontinuos.
    """
    bw = (mask > 0).astype(np.uint8)

    # Cierre morfologico para unir bordes cercanos antes de rellenar
    ks = max(5, min(bw.shape) // 40)
    if ks % 2 == 0:
        ks += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=3)

    # FloodFill desde el borde de la imagen para marcar el exterior
    h, w = closed.shape
    canvas = np.zeros((h + 2, w + 2), dtype=np.uint8)
    canvas[1:h+1, 1:w+1] = closed
    flood = canvas.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    exterior = flood[1:h+1, 1:w+1] == 1

    # Interior = lo que no es exterior, unido con el cierre
    filled = (~exterior).astype(np.uint8) | closed
    return (filled * 255).astype(np.uint8)


# ── EXTRACCION DE FIRMA RADIAL ────────────────────────────────────────────────

def extract_radial_signature(mask: np.ndarray,
                              step_deg: int = 1) -> tuple:
    """
    Extrae la firma radial de la botella desde una mascara binaria.

    Metodo: lanzamiento de rayos desde el centroide del contorno.
    Para cada angulo, avanza desde el centro hacia el exterior y registra
    la distancia al ultimo pixel perteneciente a la mascara rellenada.
    Usar una mascara rellenada hace el metodo robusto a huecos de borde.

    Parametros
    ----------
    mask     : mascara binaria (uint8 o bool), puede tener huecos internos
    step_deg : paso angular en grados (default 1 -> 360 valores)

    Retorna
    -------
    angles_deg : np.ndarray  angulos muestreados [0, step, 2*step, ...]
    distances  : np.ndarray  distancias en pixeles desde el centroide al borde
    centroid   : (cx, cy)    centroide de la mascara (flotante)
    """
    filled = _fill_mask(mask)

    moments = cv2.moments(filled)
    if moments["m00"] == 0:
        raise ValueError("La mascara esta vacia: no se puede extraer la firma.")
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    h, w = filled.shape
    max_r = int(np.sqrt(h**2 + w**2)) + 2

    angles = np.arange(0, 360, step_deg, dtype=float)
    distances = np.zeros(len(angles), dtype=np.float64)
    r_arr = np.arange(1, max_r, dtype=float)

    for i, ang in enumerate(angles):
        rad = np.deg2rad(ang)
        dx = np.cos(rad)
        dy = -np.sin(rad)   # Y invertido: 90 deg apunta hacia arriba en imagen

        pxs = (cx + dx * r_arr).astype(np.int32)
        pys = (cy + dy * r_arr).astype(np.int32)

        in_bounds = (pxs >= 0) & (pxs < w) & (pys >= 0) & (pys < h)

        # Tomar solo los pasos contiguos validos desde el inicio del rayo
        oob = np.where(~in_bounds)[0]
        valid_count = int(oob[0]) if len(oob) > 0 else len(r_arr)
        if valid_count == 0:
            distances[i] = 0.0
            continue

        px_v = pxs[:valid_count]
        py_v = pys[:valid_count]
        r_v  = r_arr[:valid_count]

        in_mask = filled[py_v, px_v] > 0
        if in_mask.any():
            last_idx = int(np.where(in_mask)[0][-1])
            distances[i] = r_v[last_idx]
        else:
            distances[i] = 0.0

    return angles, distances, (cx, cy)


# ── PERSISTENCIA ──────────────────────────────────────────────────────────────

def save_signature_csv(angles: np.ndarray, distances: np.ndarray,
                        out_path: str) -> None:
    """Guarda la firma como CSV con columnas Angle_deg, Distance_px."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df = pd.DataFrame({"Angle_deg": angles.astype(float),
                        "Distance_px": distances})
    df.to_csv(out_path, index=False, encoding="utf-8")


# ── VISUALIZACION ─────────────────────────────────────────────────────────────

def plot_signature(angles: np.ndarray, distances: np.ndarray,
                   ref_df: pd.DataFrame = None,
                   img_name: str = "",
                   out_path: str = None) -> plt.Figure:
    """
    Grafica la firma radial detectada y, si se provee ref_df,
    la firma de referencia normalizada en el mismo plot.

    Ambas curvas se normalizan por su maximo para que sean comparables
    en forma independientemente de la diferencia de unidades (px vs mm).
    """
    dist_norm = distances / distances.max() if distances.max() > 0 else distances.copy()

    if ref_df is not None:
        ref_angles  = ref_df["Angle_deg"].values
        ref_lengths = ref_df["Length_mm"].values
        ref_norm    = ref_lengths / ref_lengths.max() if ref_lengths.max() > 0 else ref_lengths.copy()
        # Interpolar referencia a los angulos de la firma detectada
        ref_interp  = np.interp(angles, ref_angles, ref_norm, period=360)

        fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                                  gridspec_kw={"height_ratios": [3, 1]})
        ax_main, ax_err = axes
    else:
        fig, ax_main = plt.subplots(figsize=(13, 5))
        ax_err = None
        ref_interp = None

    # Curva detectada
    ax_main.plot(angles, dist_norm, color="#4472C4", linewidth=2,
                  label="Detectada (px, norm.)")

    if ref_interp is not None:
        ax_main.plot(angles, ref_interp, color="#ED7D31", linewidth=2,
                      linestyle="--", label="Referencia (mm, norm.)")

    ax_main.set_ylabel("Distancia normalizada [0, 1]")
    ax_main.set_title(
        f"Firma radial  --  {img_name}\n"
        "Normalizacion por maximo (comparacion de forma, independiente de escala)"
    )
    ax_main.set_xticks(np.arange(0, 361, 30))
    ax_main.grid(True, alpha=0.35)
    ax_main.legend()

    if ax_err is not None and ref_interp is not None:
        error = dist_norm - ref_interp
        ax_err.bar(angles, error, width=max(1, angles[1] - angles[0]) * 0.9,
                   color=np.where(error >= 0, "#4472C4", "#ED7D31"),
                   alpha=0.7, label="Error (detectada - referencia)")
        ax_err.axhline(0, color="black", linewidth=0.8)
        ax_err.set_xlabel("Angulo (grados)")
        ax_err.set_ylabel("Error norm.")
        ax_err.set_xticks(np.arange(0, 361, 30))
        ax_err.grid(True, alpha=0.25)
        ax_err.legend(fontsize=8)
    else:
        ax_main.set_xlabel("Angulo (grados)")

    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


# ── COMPARACION CON REFERENCIA ────────────────────────────────────────────────

def compare_with_reference(sig_angles: np.ndarray,
                             sig_distances: np.ndarray,
                             ref_df: pd.DataFrame,
                             px_per_mm: float = None) -> dict:
    """
    Compara la firma detectada contra la firma de referencia.

    Normalizacion (siempre aplicada):
      Ambas firmas se dividen por su maximo. Las metricas miden diferencia
      de FORMA radial sin depender de escala absoluta (px vs mm).

    Si se provee px_per_mm, se calcula adicionalmente el MAE en milimetros
    reales (util si existe una calibracion fisico-espacial del sistema).

    Parametros
    ----------
    sig_angles    : angulos de la firma detectada (grados)
    sig_distances : distancias en pixeles
    ref_df        : DataFrame con columnas Angle_deg, Length_mm
    px_per_mm     : factor de conversion (pixeles por mm). Si None, se omite.

    Retorna dict con MAE, RMSE, MAPE_pct, max_error (todos sobre escala norm.)
    y opcionalmente MAE_mm si se da px_per_mm.
    """
    sig_norm = sig_distances / sig_distances.max() if sig_distances.max() > 0 else sig_distances.copy()

    ref_angles  = ref_df["Angle_deg"].values
    ref_lengths = ref_df["Length_mm"].values
    ref_norm    = ref_lengths / ref_lengths.max() if ref_lengths.max() > 0 else ref_lengths.copy()

    # Interpolar referencia a los angulos de la firma detectada
    ref_interp = np.interp(sig_angles, ref_angles, ref_norm, period=360)

    diff = sig_norm - ref_interp
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    # MAPE: solo donde la referencia normalizada > 1% para evitar division por cero
    valid = ref_interp > 0.01
    mape = float(np.mean(np.abs(diff[valid] / ref_interp[valid])) * 100) if valid.any() else float("nan")
    max_err = float(np.max(np.abs(diff)))

    result = {
        "MAE":       round(mae, 6),
        "RMSE":      round(rmse, 6),
        "MAPE_pct":  round(mape, 4),
        "max_error": round(max_err, 6),
        "nota":      "Metricas sobre firma normalizada [0,1]. Para mm reales, proveer px_per_mm.",
    }

    if px_per_mm is not None and px_per_mm > 0:
        ref_mm_interp = np.interp(sig_angles, ref_angles, ref_lengths, period=360)
        sig_mm = sig_distances / px_per_mm
        diff_mm = sig_mm - ref_mm_interp
        result["MAE_mm"]      = round(float(np.mean(np.abs(diff_mm))), 4)
        result["RMSE_mm"]     = round(float(np.sqrt(np.mean(diff_mm**2))), 4)
        result["max_error_mm"]= round(float(np.max(np.abs(diff_mm))), 4)

    return result


# ── RESUMEN CSV ───────────────────────────────────────────────────────────────

def save_comparison_summary(rows: list, out_path: str) -> None:
    """
    Guarda el resumen de comparacion de firmas como CSV.
    rows: lista de dicts con claves imagen, MAE, RMSE, MAPE_pct, max_error, ...
    Sobrescribe el archivo con todos los datos (incluye ejecuciones acumuladas).
    """
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Resumen de firmas: {out_path}")
