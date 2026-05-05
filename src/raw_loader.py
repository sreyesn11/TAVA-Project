"""
raw_loader.py  --  v2
Carga imagenes RAW de camara (CR3, CR2, etc.) usando rawpy.
Preserva profundidad de bits original (uint16) y canales RGB.
NO redimensiona. Normalizacion solo para visualizacion.
"""

import os
import json
import numpy as np
import cv2

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False
    print("[raw_loader] rawpy no disponible. Solo se soportaran archivos .raw planos.")


def load_config(config_path: str) -> dict:
    """Carga la configuracion desde un JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_image(path: str, config: dict) -> dict:
    """
    Carga una imagen RAW desde disco preservando informacion original.

    v2: retorna dict en lugar de array simple:
      - image_rgb    : np.ndarray (H, W, 3) uint16 o uint8 segun output_bps
      - image_display: np.ndarray (H, W) uint8 normalizado SOLO para visualizacion
      - dtype        : str  'uint16' o 'uint8'
      - shape        : tuple (H, W, C)
      - path         : str  ruta del archivo

    La imagen_display se obtiene sin modificar image_rgb -- la conversion es
    una copia independiente para uso exclusivo en visualizacion o en algoritmos
    que requieren uint8 (ej. cv2.Canny).
    """
    ext = os.path.splitext(path)[1].upper()
    if ext in (".CR3", ".CR2", ".NEF", ".ARW", ".DNG") or config.get("use_rawpy", True):
        return _load_with_rawpy(path, config)
    else:
        return _load_plain_raw(path, config)


def _load_with_rawpy(path: str, config: dict) -> dict:
    """
    Decodifica un archivo RAW de camara con rawpy.

    v2 cambios criticos:
    - output_bps default = 16 (preserva profundidad de bits original)
    - resize_for_processing disabled por defecto
    - Retorna imagen RGB completa SIN convertir a gris
    """
    if not RAWPY_AVAILABLE:
        raise ImportError("rawpy no esta instalado. Ejecuta: pip install rawpy")

    params = config.get("rawpy_params", {})
    output_bps = params.get("output_bps", 16)  # v2: 16 por defecto

    pp_kwargs = {
        "use_camera_wb": params.get("use_camera_wb", True),
        "no_auto_bright": params.get("no_auto_bright", False),
        "output_bps": output_bps,
    }

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(**pp_kwargs)  # uint16 (H,W,3) con output_bps=16

    # v2: NO redimensionar por defecto
    resize_cfg = config.get("resize_for_processing", {})
    if resize_cfg.get("enabled", False):  # v2: disabled=False por defecto
        max_dim = resize_cfg.get("max_dimension", 2000)
        h, w = rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Normalizacion SOLO para visualizacion (copia independiente)
    display = _normalize_for_display(rgb)
    dtype_name = "uint16" if rgb.dtype == np.uint16 else "uint8"

    return {
        "image_rgb":     rgb,
        "image_display": display,
        "dtype":         dtype_name,
        "shape":         rgb.shape,
        "path":          path,
    }


def _normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Convierte imagen a uint8 escala de grises SOLO para visualizacion.
    La imagen original NO se modifica.

    Mapeo:
      uint16  -> /257 -> uint8  (65535 / 257 = 255, sin recorte)
      float   -> min-max -> uint8
      uint8   -> copia directa
    """
    if img.dtype == np.uint8:
        out = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return out.copy()
    elif img.dtype == np.uint16:
        display = (img / 257).astype(np.uint8)
        if display.ndim == 3:
            display = cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
        return display
    else:
        mn, mx = float(img.min()), float(img.max())
        if mx > mn:
            display = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            display = np.zeros(img.shape[:2], dtype=np.uint8)
        if display.ndim == 3:
            display = cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
        return display


def _load_plain_raw(path: str, config: dict) -> dict:
    """
    Lee un archivo .raw binario plano con numpy.
    v2: preserva dtype original sin normalizar a uint8.
    """
    width    = config.get("width")
    height   = config.get("height")
    channels = config.get("channels", 1)
    dtype_str = config.get("dtype", "uint8")
    byte_order = config.get("byte_order", "little")

    if width is None or height is None:
        raise ValueError(
            f"Para archivos .raw planos se requieren 'width' y 'height' en la configuracion. "
            f"Archivo: {path}"
        )

    dtype_map = {
        "uint8":   np.uint8,
        "uint16":  np.uint16,
        "float32": np.float32,
        "float64": np.float64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"dtype no soportado: {dtype_str}. Opciones: {list(dtype_map.keys())}")

    dtype = dtype_map[dtype_str]
    expected_bytes = width * height * channels * np.dtype(dtype).itemsize
    actual_bytes = os.path.getsize(path)

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Tamano de archivo inesperado para {os.path.basename(path)}.\n"
            f"  Esperado: {expected_bytes} bytes ({width}x{height}x{channels} {dtype_str})\n"
            f"  Real:     {actual_bytes} bytes\n"
            f"  Ajusta width/height/dtype en raw_config.json."
        )

    data = np.fromfile(path, dtype=dtype)

    if byte_order == "big" and dtype != np.uint8:
        data = data.byteswap().newbyteorder()

    if channels == 1:
        img = data.reshape((height, width))
    else:
        img = data.reshape((height, width, channels))

    # v2: NO normalizar a uint8 -- preservar dtype original
    display = _normalize_for_display(img)

    return {
        "image_rgb":     img,
        "image_display": display,
        "dtype":         dtype_str,
        "shape":         img.shape,
        "path":          path,
    }


def load_images_from_folder(folder: str, config: dict, extensions: tuple = None) -> list:
    """
    Carga todas las imagenes RAW de una carpeta.

    Retorna lista de dicts con claves:
      'name', 'path', 'image_rgb', 'image_display', 'dtype', 'shape'
      'image' (alias de image_display para compatibilidad con v1)
    """
    if extensions is None:
        extensions = tuple(
            e.upper() for e in config.get("extensions", [".CR3", ".CR2", ".raw", ".RAW"])
        )
    else:
        extensions = tuple(e.upper() for e in extensions)

    if not os.path.isdir(folder):
        print(f"[raw_loader] La carpeta '{folder}' no existe.")
        return []

    files = [
        f for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].upper() in extensions
    ]

    if not files:
        print(f"[raw_loader] No se encontraron archivos RAW en '{folder}'.")
        print(f"  Extensiones buscadas: {extensions}")
        return []

    results = []
    for fname in files:
        fpath = os.path.join(folder, fname)
        print(f"  Cargando: {fname} ...", end=" ")
        try:
            img_data = load_raw_image(fpath, config)
            h, w = img_data["image_rgb"].shape[:2]
            results.append({
                "name":          fname,
                "path":          fpath,
                "image_rgb":     img_data["image_rgb"],
                "image_display": img_data["image_display"],
                "dtype":         img_data["dtype"],
                "shape":         img_data["shape"],
                # Alias backward-compat: 'image' apunta a display (uint8 gray)
                "image":         img_data["image_display"],
            })
            print(f"OK  ({w}x{h} px, {img_data['dtype']})")
        except Exception as e:
            print(f"ERROR: {e}")

    return results
