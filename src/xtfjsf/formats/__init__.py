"""Format registry and auto-detection.

New formats can be plugged in with :func:`register_format`::

    register_format("sdf", MySDFReader, detect=lambda head: head[:4] == b"\\xff\\xff\\xff\\xff",
                    extensions=(".sdf",))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Type

from ..core import SonarFile
from .jsf import JSFFile, JSFWriter
from .xtf import XTFFile, XTFWriter


@dataclass
class FormatSpec:
    name: str
    reader: Type[SonarFile]
    detect: Callable[[bytes], bool]
    extensions: Tuple[str, ...] = ()
    writer: Optional[type] = None


_REGISTRY: Dict[str, FormatSpec] = {}


def register_format(
    name: str,
    reader: Type[SonarFile],
    detect: Callable[[bytes], bool],
    extensions: Tuple[str, ...] = (),
    writer: Optional[type] = None,
) -> None:
    """Register a reader.  ``detect`` receives the first 1024 bytes of the file."""
    _REGISTRY[name] = FormatSpec(name, reader, detect, tuple(e.lower() for e in extensions), writer)


def formats() -> Dict[str, FormatSpec]:
    return dict(_REGISTRY)


def detect_format(path: str | Path) -> str:
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(1024)
    for spec in _REGISTRY.values():
        try:
            if spec.detect(head):
                return spec.name
        except Exception:
            continue
    ext = path.suffix.lower()
    for spec in _REGISTRY.values():
        if ext in spec.extensions:
            return spec.name
    raise ValueError(f"{path}: unrecognised sonar file format")


def open_sonar(path: str | Path, format: Optional[str] = None) -> SonarFile:
    """Open any supported sonar file, detecting the format from its content."""
    name = format or detect_format(path)
    if name not in _REGISTRY:
        raise ValueError(f"unknown format {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name].reader(path)


def writer_for(path: str | Path, format: Optional[str] = None, **kwargs):
    """Create a writer, choosing the format from ``format`` or the extension."""
    path = Path(path)
    if format is None:
        ext = path.suffix.lower()
        format = next((s.name for s in _REGISTRY.values() if ext in s.extensions and s.writer), None)
    spec = _REGISTRY.get(format or "")
    if spec is None or spec.writer is None:
        raise ValueError(f"no writer for {path} (format={format!r})")
    return spec.writer(path, **kwargs)


def _looks_like_xtf(h: bytes) -> bool:
    # FileFormat byte 0x7B ("{") alone would also match JSON, so sanity-check
    # the channel counts in the file header as well.
    if len(h) < 256 or h[0] != 0x7B:
        return False
    sonar = int.from_bytes(h[166:168], "little")
    bathy = int.from_bytes(h[168:170], "little")
    return sonar < 256 and bathy < 256


register_format("xtf", XTFFile, _looks_like_xtf, (".xtf",), XTFWriter)
register_format("jsf", JSFFile, lambda h: h[:2] == b"\x01\x16", (".jsf",), JSFWriter)

__all__ = [
    "JSFFile",
    "JSFWriter",
    "XTFFile",
    "XTFWriter",
    "detect_format",
    "formats",
    "open_sonar",
    "register_format",
    "writer_for",
]
