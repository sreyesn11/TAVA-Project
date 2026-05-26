"""
run_pipeline.py  --  v2.3
Pipeline completo para deteccion y medicion de botellas en imagenes RAW.

MODOS DE USO
------------
1. Detectar recortes con imagen de referencia y guardar JSONs:
       python run_pipeline.py --ref referencia.png [--preview]

2. Analizar recortes ya calculados (recortar RAWs, bordes, medidas, overlays):
       python run_pipeline.py --analyze-crops [--image IMG_0293]

ESTRUCTURA DE SALIDA
--------------------
outputs/
  full/    → todo lo relacionado con la imagen completa (pipeline, overlays, CSV)
  crops/   → recortes TIFF, pipeline sobre crop, medidas, overlay de bordes
"""

import argparse
import json
import os
import sys
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.raw_loader    import load_config, load_images_from_folder
from src.evaluation    import (run_full_pipeline_v2, run_baseline, run_improved,
                                save_crop_json, plot_contour_overlay)
from src.utils         import plot_full_pipeline_v2, plot_bottle_measurements
from src.signature     import (extract_radial_signature, save_signature_csv,
                                plot_signature, plot_signature_extraction,
                                compare_with_reference,
                                save_comparison_summary)

OUTPUT_ROOT   = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_FULL   = os.path.join(OUTPUT_ROOT, "full")
OUTPUT_CROPS  = os.path.join(OUTPUT_ROOT, "crops")
OUTPUT_FIRMA  = os.path.join(OUTPUT_ROOT, "firma")
CONFIG_PATH   = os.path.join(PROJECT_ROOT, "config", "raw_config.json")
IMAGE_FOLDER  = os.path.join(PROJECT_ROOT, "fotos_raw")
REF_FIRMA_PATH = os.path.join(PROJECT_ROOT, "firma_botella", "firma_completa_0_360.csv")

for d in (OUTPUT_ROOT, OUTPUT_FULL, OUTPUT_CROPS, OUTPUT_FIRMA):
    os.makedirs(d, exist_ok=True)


# ── MIGRAR ARCHIVOS EXISTENTES EN outputs/ RAIZ ───────────────────────────────

def migrate_existing_outputs():
    """
    Mueve archivos existentes en outputs/ al subfolder correcto.
    crops/ → recorte_*.json, crop_*.tiff, pipeline_crop_*, medidas_crop_*, analisis_*
    full/  → todo lo demas
    """
    # Solo archivos que contengan el sufijo de imagen (IMG_XXXX) van a crops/
    crop_prefixes = ("recorte_IMG_", "crop_IMG_", "pipeline_crop_IMG_",
                     "medidas_crop_IMG_", "analisis_IMG_",
                     "overlay_bordes_")
    moved = 0
    for fname in os.listdir(OUTPUT_ROOT):
        src = os.path.join(OUTPUT_ROOT, fname)
        if not os.path.isfile(src):
            continue
        if any(fname.startswith(p) for p in crop_prefixes):
            dst = os.path.join(OUTPUT_CROPS, fname)
        else:
            dst = os.path.join(OUTPUT_FULL, fname)
        if src != dst:
            shutil.move(src, dst)
            moved += 1
    if moved:
        print(f"  [migrate] {moved} archivos movidos a full/ o crops/")


# ── MODO 1: detectar recortes con referencia ──────────────────────────────────

def run_all(ref_path=None, preview=False, single=None):
    config = load_config(CONFIG_PATH)
    images = load_images_from_folder(IMAGE_FOLDER, config)

    if not images:
        print("[ERROR] No se encontraron imagenes RAW en fotos_raw/"); return
    if single:
        images = [d for d in images if single.lower() in d["name"].lower()]
        if not images:
            print(f"[ERROR] No se encontro imagen con '{single}'"); return

    print(f"\n{'='*60}")
    print(f"  PIPELINE v2.3 — DETECCION DE RECORTE  ({len(images)} imagen/es)")
    print(f"  Referencia : {ref_path or 'ninguna (deteccion automatica)'}")
    print(f"  Preview    : {'SI' if preview else 'NO'}")
    print(f"  Salida     : {OUTPUT_FULL}")
    print(f"{'='*60}\n")

    for i, img_data in enumerate(images, 1):
        name = img_data["name"]
        stem = os.path.splitext(name)[0]
        print(f"[{i}/{len(images)}] {name}")
        t0 = time.time()

        result = run_full_pipeline_v2(img_data, ref_path=ref_path, preview=preview)
        if result.get("cancelled"):
            print("  [CANCELADO]\n"); continue

        bm   = result.get("bbox_metrics", {})
        src  = result.get("bbox_source", "?")
        diag = result.get("diagnosis", {})
        print(f"  fuente: {src.upper()}  {bm.get('width','?')}x{bm.get('height','?')}px  "
              f"aspect={bm.get('aspect_ratio','?')}  "
              f"{'OK' if diag.get('ok') else 'WARN'}")

        crop_json = result.get("crop_json")
        if crop_json:
            save_crop_json(crop_json,
                           os.path.join(OUTPUT_CROPS, f"recorte_{stem}.json"))

        fig = plot_full_pipeline_v2(result, img_name=name,
                                     output_path=os.path.join(OUTPUT_FULL,
                                                               f"pipeline_{stem}.png"))
        plt.close(fig)
        print(f"  {time.time()-t0:.1f}s\n")

    print(f"Recortes guardados en {OUTPUT_CROPS}/\n")


# ── MODO 2: analizar recortes desde JSONs ─────────────────────────────────────

def _make_img_data(crop_rgb: np.ndarray, path: str) -> dict:
    """Construye img_data para run_full_pipeline_v2 desde un crop uint16."""
    disp = (crop_rgb / 257).astype(np.uint8) if crop_rgb.dtype == np.uint16 else crop_rgb.copy()
    disp_gray = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY) if disp.ndim == 3 else disp
    return {
        "image_rgb":     crop_rgb,
        "image_display": disp_gray,
        "name":          os.path.basename(path),
        "path":          path,
        "dtype":         str(crop_rgb.dtype),
        "shape":         crop_rgb.shape,
    }


def _overlay_borders(display_gray: np.ndarray,
                      baseline_res: dict,
                      improved_res: dict,
                      title: str,
                      output_path: str) -> None:
    """
    Genera imagen con los bordes detectados calcados:
      Azul   (66,114,176) → contorno baseline (Gaussiano + Canny)
      Naranja(221,132, 82) → contorno pipeline mejorado (CLAHE + Bilateral + multi-metodo)

    Si el contorno es None pero hay edges, dibuja los edges directamente.
    """
    if display_gray.ndim == 2:
        canvas = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
    else:
        canvas = cv2.cvtColor(display_gray, cv2.COLOR_RGB2BGR)

    # Azul: contorno baseline
    c_base = baseline_res.get("contour")
    if c_base is not None:
        cv2.drawContours(canvas, [c_base], -1, (176, 114, 66), 3)   # BGR azul
    else:
        edges_b = baseline_res.get("edges")
        if edges_b is not None:
            canvas[edges_b > 0] = [176, 114, 66]

    # Naranja: contorno mejorado
    c_impr = improved_res.get("contour")
    if c_impr is not None:
        cv2.drawContours(canvas, [c_impr], -1, (82, 132, 221), 3)   # BGR naranja
    else:
        edges_i = improved_res.get("edges")
        if edges_i is not None:
            canvas[edges_i > 0] = [82, 132, 221]

    # Figura: original | overlay | bordes por separado
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    axes[0].imshow(display_gray, cmap="gray")
    axes[0].set_title("Recorte original (display)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        "Bordes detectados\nAzul = Baseline  |  Naranja = Mejorado",
        fontsize=10
    )
    axes[1].axis("off")

    # Panel 3: mapa de bordes combinado
    edges_base = baseline_res.get("edges", np.zeros_like(display_gray))
    edges_impr = improved_res.get("edges", np.zeros_like(display_gray))
    combined_vis = np.zeros((*display_gray.shape[:2], 3), dtype=np.uint8)
    combined_vis[edges_base > 0] = [176, 114, 66]    # azul (baseline)
    combined_vis[edges_impr > 0] = [82,  132, 221]   # naranja (mejorado)
    # Overlap en blanco
    overlap = (edges_base > 0) & (edges_impr > 0)
    combined_vis[overlap] = [255, 255, 255]

    axes[2].imshow(combined_vis)
    axes[2].set_title(
        "Mapa de bordes\nAzul=Base  Naranja=Mejor  Blanco=Coincidencia",
        fontsize=10
    )
    axes[2].axis("off")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _run_signature(silhouette_mask: np.ndarray,
                    stem: str,
                    summary_rows: list) -> None:
    """
    Extrae la firma radial de la silueta, la guarda como CSV y grafica,
    la compara con la firma de referencia y acumula el resultado en summary_rows.
    """
    if silhouette_mask is None or silhouette_mask.max() == 0:
        print("  [FIRMA] Mascara vacia, firma no extraida.")
        return

    print("  Extrayendo firma radial...")
    try:
        angles, distances, centroid = extract_radial_signature(silhouette_mask)
    except ValueError as exc:
        print(f"  [FIRMA] Error: {exc}")
        return

    # Guardar CSV de la firma detectada
    csv_path = os.path.join(OUTPUT_FIRMA, f"firma_modelo_{stem}.csv")
    save_signature_csv(angles, distances, csv_path)
    print(f"  Firma CSV: {os.path.basename(csv_path)}")

    # Visualizacion paso a paso de la extraccion (silueta → relleno → rayos → firma)
    ext_png_path = os.path.join(OUTPUT_FIRMA, f"extraccion_firma_{stem}.png")
    fig = plot_signature_extraction(
        silhouette_mask, angles, distances, centroid,
        img_name=stem, out_path=ext_png_path
    )
    plt.close(fig)
    print(f"  Extraccion PNG: {os.path.basename(ext_png_path)}")

    # Cargar firma de referencia (si existe)
    ref_df = None
    if os.path.isfile(REF_FIRMA_PATH):
        ref_df = pd.read_csv(REF_FIRMA_PATH)
    else:
        print(f"  [FIRMA] Referencia no encontrada: {REF_FIRMA_PATH}")

    # Grafica de la firma (con o sin referencia)
    png_path = os.path.join(OUTPUT_FIRMA, f"grafica_firma_{stem}.png")
    fig = plot_signature(angles, distances, ref_df=ref_df,
                          img_name=stem, out_path=png_path)
    plt.close(fig)
    print(f"  Firma PNG: {os.path.basename(png_path)}")

    # Comparacion con referencia
    if ref_df is not None:
        metrics = compare_with_reference(angles, distances, ref_df)
        print(f"  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}"
              f"  MAPE={metrics['MAPE_pct']:.2f}%  max_err={metrics['max_error']:.4f}"
              "  (escala norm. [0,1])")
        row = {"imagen": stem}
        row.update(metrics)
        summary_rows.append(row)
    else:
        summary_rows.append({"imagen": stem,
                               "MAE": None, "RMSE": None,
                               "MAPE_pct": None, "max_error": None,
                               "nota": "sin referencia"})


def analyze_crops(single=None, preview=False, signature=False):
    """
    Lee los JSONs de recorte (outputs/crops/recorte_*.json), aplica el recorte
    sobre cada RAW y ejecuta:
      - Pipeline completo (bordes + silueta + medidas)
      - Overlay de bordes baseline (azul) vs mejorado (naranja) sobre el recorte
      - (opcional, --signature) Firma radial y comparacion con referencia
    Guarda todo en outputs/crops/ y, si signature=True, en outputs/firma/.
    """
    import rawpy
    import imageio.v3 as iio
    firma_summary_rows = []

    json_files = sorted([
        f for f in os.listdir(OUTPUT_CROPS)
        if f.startswith("recorte_") and f.endswith(".json")
    ])
    if not json_files:
        print("[ERROR] No se encontraron recorte_*.json en outputs/crops/")
        print("        Corre primero: python run_pipeline.py --ref referencia.png")
        return

    if single:
        json_files = [f for f in json_files if single.lower() in f.lower()]
        if not json_files:
            print(f"[ERROR] No se encontro JSON para '{single}'"); return

    print(f"\n{'='*60}")
    print(f"  PIPELINE v2.3 — ANALISIS DE RECORTES  ({len(json_files)} imagen/es)")
    print(f"  Preview : {'SI' if preview else 'NO'}")
    print(f"  Salida  : {OUTPUT_CROPS}")
    print(f"{'='*60}\n")

    for i, jname in enumerate(json_files, 1):
        jpath = os.path.join(OUTPUT_CROPS, jname)
        with open(jpath, encoding="utf-8") as fp:
            jdata = json.load(fp)

        raw_path = jdata["metadata"]["path"]
        rect     = jdata["raw"]["rect"]
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        stem = os.path.splitext(os.path.basename(raw_path))[0]

        print(f"[{i}/{len(json_files)}] {os.path.basename(raw_path)}")
        print(f"  Recorte : x={x} y={y} w={w} h={h}")

        # Buscar el RAW
        if not os.path.isfile(raw_path):
            alt = os.path.join(IMAGE_FOLDER, os.path.basename(raw_path))
            if os.path.isfile(alt):
                raw_path = alt
            else:
                print(f"  [ERROR] RAW no encontrado: {raw_path}\n"); continue

        t0 = time.time()

        # -- Cargar RAW y recortar ----------------------------------------
        print("  Cargando RAW y aplicando recorte...")
        with rawpy.imread(raw_path) as raw:
            rgb_full = raw.postprocess(
                output_bps=16, no_auto_bright=True, use_camera_wb=True,
            )
        crop_rgb = rgb_full[y : y + h, x : x + w].copy()
        del rgb_full

        # Guardar TIFF (uint16, sin perdida)
        tiff_path = os.path.join(OUTPUT_CROPS, f"crop_{stem}.tiff")
        iio.imwrite(tiff_path, crop_rgb)
        print(f"  TIFF    : {os.path.basename(tiff_path)}")

        # -- Imagen de display para overlays ------------------------------
        disp_bgr  = cv2.cvtColor(
            (crop_rgb / 257).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        disp_gray = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2GRAY)

        img_data = _make_img_data(crop_rgb, raw_path)

        # -- Pipeline completo (bordes, silueta, medidas) -----------------
        print("  Pipeline v2.3 sobre recorte...")
        result = run_full_pipeline_v2(img_data, ref_path=None, preview=preview)
        if result.get("cancelled"):
            print("  [CANCELADO]\n"); continue

        bm   = result.get("bbox_metrics", {})
        diag = result.get("diagnosis", {})
        m    = result.get("bottle_measures", {})
        print(f"  bbox    : {result.get('bbox_consensus')}  "
              f"aspect={bm.get('aspect_ratio','?')}  "
              f"{'OK' if diag.get('ok') else 'WARN'}")
        for w_ in diag.get("warnings", []):
            print(f"    ! {w_}")
        print(f"  altura  : {m.get('height_bottle_px','?')} px  "
              f"ancho_max: {m.get('width_max_px','?')} px")
        print(f"  simetria: {m.get('simetria_vertical',0):.3f}  "
              f"cuello/cuerpo: {m.get('ratio_cuello_cuerpo',0):.3f}")

        # -- Baseline y mejorado para el overlay de bordes ----------------
        print("  Generando overlay de bordes (baseline + mejorado)...")
        base_res = run_baseline(crop_rgb)
        impr_res = run_improved(crop_rgb)

        # Usar tambien el contorno de silueta del pipeline v2.3 como alternativa
        sil_contour = result.get("silhouette_contour")
        impr_res_v23 = {
            "contour": sil_contour,
            "edges":   result.get("combined_edges"),
        }

        # Overlay clasico: baseline (azul) vs mejorado CLAHE+bilateral (naranja)
        out_overlay = os.path.join(OUTPUT_CROPS, f"overlay_bordes_{stem}.png")
        _overlay_borders(
            disp_gray,
            baseline_res=base_res,
            improved_res=impr_res,
            title=f"{stem}  —  Bordes detectados sobre recorte\n"
                  f"Azul = Baseline  |  Naranja = Mejorado (CLAHE+Bilateral)",
            output_path=out_overlay,
        )

        # Overlay v2.3: baseline (azul) vs silueta multi-metodo (naranja)
        out_overlay_v23 = os.path.join(OUTPUT_CROPS,
                                        f"overlay_bordes_v23_{stem}.png")
        _overlay_borders(
            disp_gray,
            baseline_res=base_res,
            improved_res=impr_res_v23,
            title=f"{stem}  —  Bordes detectados sobre recorte (v2.3)\n"
                  f"Azul = Baseline  |  Naranja = Multi-metodo combinado",
            output_path=out_overlay_v23,
        )

        # Pipeline visual (6 paneles)
        out_pipe = os.path.join(OUTPUT_CROPS, f"pipeline_crop_{stem}.png")
        fig = plot_full_pipeline_v2(result, img_name=f"{stem} (crop)",
                                     output_path=out_pipe)
        plt.close(fig)

        # Medidas por zona
        out_med = os.path.join(OUTPUT_CROPS, f"medidas_crop_{stem}.png")
        fig = plot_bottle_measurements(
            m,
            silhouette_crop=result.get("silhouette_mask"),
            img_name=f"{stem} (crop)",
            output_path=out_med,
        )
        plt.close(fig)

        # JSON con medidas
        crop_json = result.get("crop_json", {})
        if crop_json:
            crop_json["bottle_measures"] = m
            save_crop_json(crop_json,
                           os.path.join(OUTPUT_CROPS, f"analisis_{stem}.json"))

        # -- Firma radial (opcional, activada con --signature) ----------------
        if signature:
            sil_mask = result.get("silhouette_mask")
            # Recortar la silueta al bbox detectado para aislar solo la botella
            bbox = result.get("bbox_consensus")
            if sil_mask is not None and bbox is not None:
                bx, by, bw_, bh_ = bbox
                sil_for_sig = sil_mask[by:by+bh_, bx:bx+bw_]
            else:
                sil_for_sig = sil_mask
            _run_signature(sil_for_sig, stem, firma_summary_rows)

        t_total = time.time() - t0
        print(f"  tiempo  : {t_total:.1f}s")
        print(f"  outputs : overlay_bordes_{stem}.png")
        print(f"            overlay_bordes_v23_{stem}.png")
        print(f"            pipeline_crop_{stem}.png")
        print(f"            medidas_crop_{stem}.png")
        if signature:
            print(f"            firma_modelo_{stem}.csv      (outputs/firma/)")
            print(f"            extraccion_firma_{stem}.png  (outputs/firma/)")
            print(f"            grafica_firma_{stem}.png     (outputs/firma/)")
        print()

    # -- Resumen de comparacion de firmas (si se pidio --signature) -----------
    if signature and firma_summary_rows:
        resumen_path = os.path.join(OUTPUT_FIRMA, "resumen_firma.csv")
        save_comparison_summary(firma_summary_rows, resumen_path)
        print(f"  Resumen firmas: {resumen_path}")

    print(f"{'='*60}")
    print(f"  Listo. Outputs en: {OUTPUT_CROPS}")
    if signature:
        print(f"  Firmas en      : {OUTPUT_FIRMA}")
    print(f"{'='*60}\n")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline v2.3 para botellas CR3")
    parser.add_argument("--ref",           default=None,
                        help="Imagen de referencia con recuadro rojo")
    parser.add_argument("--preview",       action="store_true",
                        help="Mostrar ventana emergente de confirmacion")
    parser.add_argument("--image",         default=None,
                        help="Procesar solo una imagen (ej: IMG_0293)")
    parser.add_argument("--analyze-crops", action="store_true",
                        help="Leer JSONs y analizar recortes")
    parser.add_argument("--signature",    action="store_true",
                        help="Extraer firma radial y comparar con referencia "
                             "(requiere --analyze-crops; salida en outputs/firma/)")
    args = parser.parse_args()

    if args.signature and not args.analyze_crops:
        print("[INFO] --signature implica --analyze-crops. Activando automaticamente.")
        args.analyze_crops = True

    # Migrar archivos existentes al subfolder correcto
    migrate_existing_outputs()

    if args.analyze_crops:
        analyze_crops(single=args.image, preview=args.preview,
                      signature=args.signature)
    else:
        ref = None
        if args.ref:
            ref = (args.ref if os.path.isabs(args.ref)
                   else os.path.join(PROJECT_ROOT, args.ref))
            if not os.path.isfile(ref):
                print(f"[ERROR] Referencia no encontrada: {ref}")
                sys.exit(1)
        run_all(ref_path=ref, preview=args.preview, single=args.image)
