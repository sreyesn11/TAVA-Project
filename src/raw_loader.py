"""
raw_loader.py
Carga imagenes RAW de camara (CR3, CR2, etc.) usando rawpy,
con fallback a lectura binaria para archivos .raw planos.
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


def load_raw_image(path: str, config: dict) -> np.ndarray:
    """
    Carga una imagen RAW desde disco.

    Para CR3/CR2 usa rawpy (LibRaw).
    Para .raw binario plano usa lectura directa con numpy.

    Retorna array uint8 en escala de grises (H, W).
    """
    ext = os.path.splitext(path)[1].upper()

    if ext in (".CR3", ".CR2", ".NEF", ".ARW", ".DNG") or config.get("use_rawpy", True):
        return _load_with_rawpy(path, config)
    else:
        return _load_plain_raw(path, config)


def _load_with_rawpy(path: str, config: dict) -> np.ndarray:
    """Decodifica un archivo RAW de camara con rawpy."""
    if not RAWPY_AVAILABLE:
        raise ImportError("rawpy no esta instalado. Ejecuta: pip install rawpy")

    params = config.get("rawpy_params", {})

    # Construir parametros de postprocesado
    pp_kwargs = {
        "use_camera_wb": params.get("use_camera_wb", True),
        "no_auto_bright": params.get("no_auto_bright", False),
        "output_bps": params.get("output_bps", 8),
    }

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(**pp_kwargs)  # uint8 RGB (H, W, 3)

    # Reducir resolucion si la imagen es muy grande para procesamiento
    resize_cfg = config.get("resize_for_processing", {})
    if resize_cfg.get("enabled", True):
        max_dim = resize_cfg.get("max_dimension", 2000)
        h, w = rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convertir a escala de grises
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray


def _load_plain_raw(path: str, config: dict) -> np.ndarray:
    """Lee un archivo .raw binario plano con numpy."""
    width = config.get("width")
    height = config.get("height")
    channels = config.get("channels", 1)
    dtype_str = config.get("dtype", "uint8")
    byte_order = config.get("byte_order", "little")

    if width is None or height is None:
        raise ValueError(
            f"Para archivos .raw planos se requieren 'width' y 'height' en la configuracion. "
            f"Archivo: {path}"
        )

    dtype_map = {
        "uint8": np.uint8,
        "uint16": np.uint16,
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

    # Ajustar orden de bytes si es big-endian
    if byte_order == "big" and dtype != np.uint8:
        data = data.byteswap().newbyteorder()

    if channels == 1:
        img = data.reshape((height, width))
    else:
        img = data.reshape((height, width, channels))

    # Normalizar uint16 a uint8 para procesamiento
    if dtype == np.uint16:
        img = (img / 65535.0 * 255).astype(np.uint8)
    elif dtype in (np.float32, np.float64):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    # Si es color, convertir a gris
    if channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def load_images_from_folder(folder: str, config: dict, extensions: tuple = None) -> list:
    """
    Carga todas las imagenes RAW de una carpeta.

    Retorna lista de dicts: {'name': str, 'path': str, 'image': np.ndarray}
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
            img = load_raw_image(fpath, config)
            results.append({"name": fname, "path": fpath, "image": img})
            print(f"OK  ({img.shape[1]}x{img.shape[0]} px)")
        except Exception as e:
            print(f"ERROR: {e}")

    return results
