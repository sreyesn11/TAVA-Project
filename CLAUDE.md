# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Classical computer vision pipeline for automated geometric measurement of glass bottles from Canon RAW (CR3) images. Built as a technical demo for Licorera de Caldas to validate automated dimension extraction as an alternative to manual inspection. No deep learning — uses classical methods (Canny, Sobel, Scharr, CLAHE, morphology) for interpretability and zero-shot operation.

## Commands

### Run the pipeline

**The workflow is two-step: Mode 1 must run first to produce the crop JSONs that Mode 2 reads.**

**Mode 1 — detect and crop using a reference image with red rectangle annotation:**
```
python run_pipeline.py --ref referencia.png [--preview] [--image IMG_0293]
```

**Mode 2 — analyze crops from previously saved JSONs:**
```
python run_pipeline.py --analyze-crops [--image IMG_0293] [--preview]
```

**Mode 2 + radial signature comparison:**
```
python run_pipeline.py --analyze-crops --signature [--image IMG_0293]
```

- `--preview`: opens an interactive OpenCV window for manual confirmation of the detected crop. **Requires a display — do not use in headless environments.**
- `--image NAME`: restrict processing to a single image (e.g. `IMG_0293`, no extension)
- `--signature` implies `--analyze-crops` automatically

### Install dependencies
```
pip install -r requirements.txt
```

`imageio` is also required by Mode 2 (TIFF export) but not listed in `requirements.txt`. It ships with `scikit-image`, so it is usually available. If not: `pip install imageio`.

### Launch the demo notebook
```
jupyter notebook notebooks/demo_botellas.ipynb
```

### Regenerate the reference signature
If new physical measurements of the reference bottle are available (exported from ImageJ/Fiji as `Results_0_90.csv` and `Results_0_m90.csv`):
```
cd firma_botella
python generar_firma_botella.py
```
This overwrites `firma_botella/firma_completa_0_360.csv`, which is the ground truth used by `--signature`.

## Architecture

### Data flow

```
fotos_raw/*.CR3
    └─ raw_loader.py        → uint16 RGB dict {image_rgb, image_display, dtype, shape, path}
        └─ preprocessing.py → uint8 grayscale (CLAHE + bilateral, or simple Gaussian)
            └─ edge_detection.py → 5-method voting edges + primary contour
                └─ contour_measurement.py → pixel-space geometric metrics
                    └─ evaluation.py     → pipeline orchestration, IoU, crop export
                        └─ utils.py      → matplotlib/cv2 visualizations, CSV/JSON/TIFF export
```

### Module responsibilities

| File | Role |
|------|------|
| `src/raw_loader.py` | Decodes CR3/CR2 via rawpy (LibRaw); preserves uint16; no resize by default |
| `src/preprocessing.py` | Two pipelines: baseline (Gaussian→uint8) and improved (CLAHE + bilateral→uint8); kernel sizes scale with resolution |
| `src/edge_detection.py` | Baseline Canny + improved Canny; `detect_all_methods()` runs all 5 detectors (Canny, Sobel, Scharr, Laplacian, Morphological Gradient) |
| `src/contour_measurement.py` | `measure_structure()` for generic shapes; `measure_bottle_in_crop()` for zone-based metrics (neck 0-20%, shoulder 20-35%, body 35-80%, base 80-100%) |
| `src/evaluation.py` | `run_full_pipeline_v2()` — main entry point; handles reference red-rectangle detection, Hough-line crop scoring, silhouette masking, metric export |
| `src/signature.py` | Radial signature extraction from `silhouette_mask`; comparison against `firma_botella/firma_completa_0_360.csv` |
| `src/utils.py` | All visualization (matplotlib + cv2); `plot_full_pipeline_v2()` produces the canonical 6-panel output |
| `run_pipeline.py` | CLI entry point (v2.3); orchestrates both modes |

### Configuration

`config/raw_config.json` controls rawpy decode params (white balance, 16-bit output), resize policy (disabled), image folder (`fotos_raw/`), and extensions. Change `output_bps` here to switch between 8 and 16-bit RAW decoding.

### Reference signature data

`firma_botella/` holds the physical reference measurements of the bottle model:
- `firma_completa_0_360.csv` — ground-truth radial signature (columns: `Angle_deg`, `Length_mm`), assembled from ImageJ/Fiji radial profile exports.
- `generar_firma_botella.py` — reads `Results_0_90.csv` and `Results_0_m90.csv` (upper/lower half exports from ImageJ) and stitches them into the complete 0-360° CSV.

### Outputs

```
outputs/
├── full/    — 6-panel PNG + crop JSON per image (from mode 1)
├── crops/   — uint16 TIFF crop, pipeline PNG, edge overlay PNG,
│              measurement PNG, analysis JSON (from mode 2)
└── firma/   — radial signature CSVs, comparison plots, resumen_firma.csv (from --signature)
```

### Key design decisions

- **uint16 preserved end-to-end** until OpenCV functions that require uint8 (Canny, etc.) force conversion — preserves tonal range for glass/transparency.
- **No resizing** — full 32 MP images (≈6984×4660 px) ensure geometric accuracy in pixel-space measurements.
- **Multi-method voting** — no single edge detector is reliable on reflective glass; 5 methods are combined in `detect_all_methods()`.
- **Red-rectangle reference** — manual annotation on a sample PNG provides ground truth bounding box; `find_red_rectangle()` detects it in HSV, then `scale_rect()` maps it to RAW dimensions.
- **Bbox priority cascade** — `run_full_pipeline_v2()` selects the crop region from: (1) red-rectangle reference, (2) external silhouette, (3) matrix scoring, (4) none. The source is recorded as `bbox_source` in the result dict and the crop JSON.
- **Zone-based measurement** — neck/shoulder/body/base zones in `measure_bottle_in_crop()` enable bottle model classification without a labeled dataset.
- **No metric calibration yet** — all measurements are in pixels; mm conversion requires a physical reference target. For signature comparison, both curves are normalized by their maximum before computing error metrics (shape comparison, scale-independent).
- **Mode 2 re-decodes RAWs** — `analyze_crops()` in `run_pipeline.py` re-opens the original CR3 via rawpy directly (not through `raw_loader`) to apply the saved crop rectangle at full resolution.
