"""Image writers: dependency-free PNG and (Pillow-based) GeoTIFF."""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Optional

import numpy as np


def write_png(path: str | Path, img: np.ndarray) -> Path:
    """Write an 8-bit grayscale (H, W) or RGB (H, W, 3) array as PNG using only zlib."""
    path = Path(path)
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise TypeError("write_png expects uint8 data; use processing.stretch() first")
    if a.ndim == 2:
        color_type, channels = 0, 1
    elif a.ndim == 3 and a.shape[2] == 3:
        color_type, channels = 2, 3
    else:
        raise ValueError("image must be (H, W) or (H, W, 3)")
    h, w = a.shape[:2]
    rows = a.reshape(h, w * channels)
    raw = np.empty((h, w * channels + 1), dtype=np.uint8)
    raw[:, 0] = 0  # filter type "none"
    raw[:, 1:] = rows

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
    return path


def write_geotiff(
    path: str | Path,
    data: np.ndarray,
    x0: float,
    y0: float,
    cell_size: float,
    epsg: Optional[int] = None,
    nodata: Optional[float] = None,
) -> Path:
    """Write a single-band north-up GeoTIFF (upper-left corner at x0, y0)."""
    try:
        from PIL import Image, TiffImagePlugin
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("GeoTIFF output requires Pillow: pip install 'xtfjsf[images]'") from exc

    path = Path(path)
    a = np.asarray(data)
    # uint8 -> mode "L", float32 -> mode "F"
    im = Image.fromarray(np.ascontiguousarray(a if a.dtype == np.uint8 else a.astype(np.float32)))

    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (float(cell_size), float(cell_size), 0.0)  # ModelPixelScale
    ifd.tagtype[33550] = 12
    ifd[33922] = (0.0, 0.0, 0.0, float(x0), float(y0), 0.0)  # ModelTiepoint
    ifd.tagtype[33922] = 12
    keys = [1, 1, 0, 0]
    keys += [1024, 0, 1, 1]  # GTModelType = projected
    keys += [1025, 0, 1, 1]  # GTRasterType = PixelIsArea
    if epsg:
        keys += [3072, 0, 1, int(epsg)]  # ProjectedCSType
    keys[3] = (len(keys) - 4) // 4
    ifd[34735] = tuple(keys)  # GeoKeyDirectory
    ifd.tagtype[34735] = 3
    if nodata is not None:
        ifd[42113] = "nan" if isinstance(nodata, float) and math.isnan(nodata) else str(nodata)
        ifd.tagtype[42113] = 2
    im.save(path, format="TIFF", tiffinfo=ifd)
    return path
