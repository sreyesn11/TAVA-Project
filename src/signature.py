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


# ── VISUALIZACION PASO A PASO ─────────────────────────────────────────────────

def plot_signature_extraction(mask: np.ndarray,
                               angles: np.ndarray,
                               distances: np.ndarray,
                               centroid: tuple,
                               img_name: str = "",
                               out_path: str = None,
                               ray_step: int = 15,
                               max_display_px: int = 700) -> plt.Figure:
    """
    Visualizacion paso a paso de la extraccion de firma radial.

    Paneles:
      1  Silueta original (mascara de entrada, con posibles huecos)
      2  Mascara rellena + centroide geometrico marcado
      3  Rayos lanzados desde el centroide (cada ray_step grados)
      4  Firma radial cartesiana (distancia al borde en px vs angulo)
      5  Firma radial polar (contorno de la botella visto desde el centroide)

    Parametros
    ----------
    mask          : mascara binaria original ANTES de rellenar
    angles        : angulos en grados (360 valores a paso 1, de extract_radial_signature)
    distances     : distancias en pixeles al borde (de extract_radial_signature)
    centroid      : (cx, cy) calculado sobre la mascara rellena
    ray_step      : cada cuantos grados dibujar un rayo en el panel 3 (default 15)
    max_display_px: lado maximo para display — evita figuras de 6000 px
    """
    filled = _fill_mask(mask)
    cx, cy = centroid

    # ── Downsampling para display ─────────────────────────────────────────────
    h_f, w_f = filled.shape
    scale = min(1.0, max_display_px / max(h_f, w_f, 1))
    if scale < 1.0:
        new_h = max(1, int(h_f * scale))
        new_w = max(1, int(w_f * scale))
        filled_d = cv2.resize(filled, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_d   = cv2.resize(
            (mask > 0).astype(np.uint8) * 255, (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )
        cxd, cyd = cx * scale, cy * scale
        dist_d   = distances * scale
    else:
        filled_d = filled
        mask_d   = (mask > 0).astype(np.uint8) * 255
        cxd, cyd = cx, cy
        dist_d   = distances

    dh, dw = filled_d.shape

    # ── Layout 2×3 con GridSpec ───────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 11))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32,
                           left=0.04, right=0.97, top=0.89, bottom=0.07)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2], projection='polar')

    # ── Panel 1: Silueta original ─────────────────────────────────────────────
    ax1.imshow(mask_d, cmap='gray', interpolation='nearest')
    ax1.set_title("① Silueta detectada\n(máscara de entrada)", fontweight='bold', fontsize=10)
    ax1.axis('off')

    # ── Panel 2: Mascara rellena + centroide ──────────────────────────────────
    ax2.imshow(filled_d, cmap='gray', interpolation='nearest')
    ax2.plot(cxd, cyd, 'r+', markersize=20, markeredgewidth=2.5)
    ax2.plot(cxd, cyd, 'ro', markersize=8,
             markerfacecolor='red', markeredgecolor='white', markeredgewidth=1.5,
             label=f'Centroide ({cx:.0f}, {cy:.0f}) px')
    ax2.legend(loc='lower right', fontsize=7.5, framealpha=0.75)
    ax2.set_title("② Máscara rellena\n+ centroide geométrico (•)", fontweight='bold', fontsize=10)
    ax2.axis('off')

    # ── Panel 3: Rayos desde el centroide ─────────────────────────────────────
    # Fondo azul oscuro para que los rayos multicolor destaquen sobre la silueta
    overlay = np.zeros((dh, dw, 3), dtype=np.uint8)
    overlay[filled_d > 0]  = [25, 45, 80]
    overlay[filled_d == 0] = [10, 10, 13]
    ax3.imshow(overlay, interpolation='nearest')

    cmap_rays   = plt.cm.hsv
    sampled_idx = np.arange(0, len(angles), ray_step)

    for idx in sampled_idx:
        ang = float(angles[idx])
        d   = dist_d[idx]
        if d == 0:
            continue
        rad = np.deg2rad(ang)
        ex  = cxd + np.cos(rad) * d
        ey  = cyd + (-np.sin(rad)) * d          # eje Y invertido en imagen
        col = cmap_rays(ang / 360.0)
        ax3.plot([cxd, ex], [cyd, ey], color=col, linewidth=1.1, alpha=0.88)
        ax3.plot(ex, ey, 'o', color=col, markersize=4, alpha=0.95,
                 markeredgecolor='white', markeredgewidth=0.4)

    # Centroide encima de los rayos
    ax3.plot(cxd, cyd, 'w+', markersize=18, markeredgewidth=2.5)
    ax3.plot(cxd, cyd, 'wo', markersize=7,
             markerfacecolor='white', markeredgecolor='#aaa', markeredgewidth=1.5)

    # Etiquetas de angulos cardinales
    ang_step = float(angles[1] - angles[0]) if len(angles) > 1 else 1.0
    for lang, ltxt in zip([0, 90, 180, 270],
                          ['0°\n(der)', '90°\n(arr)', '180°\n(izq)', '270°\n(abj)']):
        idx_l = int(round(lang / ang_step)) % len(angles)
        r_lbl = dist_d[idx_l] * 1.22 if dist_d[idx_l] > 0 else float(max(dist_d)) * 0.5
        rad_l = np.deg2rad(lang)
        lx = float(np.clip(cxd + np.cos(rad_l) * r_lbl, 4, dw - 4))
        ly = float(np.clip(cyd + (-np.sin(rad_l)) * r_lbl, 4, dh - 4))
        ax3.text(lx, ly, ltxt, ha='center', va='center',
                 fontsize=7.5, color='yellow', fontweight='bold')

    ax3.set_xlim(0, dw)
    ax3.set_ylim(dh, 0)          # mantener orientacion de imagen (Y hacia abajo)
    ax3.set_title(f"③ Lanzamiento de rayos\n(desde centroide, cada {ray_step}°)",
                  fontweight='bold', fontsize=10)
    ax3.axis('off')

    # ── Panel 4: Firma cartesiana ─────────────────────────────────────────────
    ax4.plot(angles, distances, color='#4472C4', linewidth=2, label='Firma detectada')
    ax4.fill_between(angles, 0, distances, alpha=0.14, color='#4472C4')
    ax4.set_xlabel("Ángulo (grados)", fontsize=10)
    ax4.set_ylabel("Distancia al borde (píxeles)", fontsize=10)
    ax4.set_title("④ Firma radial resultante  (distancia al borde por ángulo)",
                  fontweight='bold', fontsize=10)
    ax4.set_xticks(np.arange(0, 361, 45))
    ax4.set_xlim(0, 359)
    ax4.grid(True, alpha=0.35)

    ymax = float(distances.max())
    for lang, ltxt in zip(
        [0, 90, 180, 270],
        ['0°\n(derecha)', '90°\n(arriba / cuello)', '180°\n(izquierda)', '270°\n(abajo / base)']
    ):
        ax4.axvline(lang, color='#bbb', linewidth=0.85, linestyle='--', alpha=0.7)
        ax4.text(lang + 3, ymax * 0.97, ltxt, fontsize=7.5, color='#666', va='top')

    ax4.legend(fontsize=9, loc='upper right')

    # ── Panel 5: Firma polar ──────────────────────────────────────────────────
    theta  = np.deg2rad(np.append(angles, angles[0]))   # cerrar poligono
    r_plot = np.append(distances, distances[0])
    ax5.plot(theta, r_plot, color='#4472C4', linewidth=1.6)
    ax5.fill(theta, r_plot, alpha=0.22, color='#4472C4')
    ax5.set_theta_zero_location('E')     # 0 deg a la derecha, igual que convencion del codigo
    ax5.set_theta_direction(1)           # counter-clockwise: 90 deg apunta arriba
    ax5.set_title("⑤ Vista polar\n(contorno desde centroide)",
                  fontweight='bold', fontsize=10, pad=13)
    ax5.tick_params(labelsize=7.5)

    # ── Titulo global ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"Extracción de firma radial  ·  {img_name}\n"
        "Lanzamiento de rayos desde centroide geométrico sobre máscara de silueta rellena",
        fontsize=12, fontweight='bold'
    )

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')

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
