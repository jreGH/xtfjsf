"""Georeferencing of side-scan pixels and mosaicking to GeoTIFF / PNG.

Everything here is pure numpy: UTM (WGS84) conversions use the standard
Krüger series (sub-millimetre inside a zone), so no GDAL/pyproj is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .processing import _moving_average
from .waterfall import Waterfall

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_K0 = 0.9996


def utm_zone(longitude: float) -> int:
    return int(((longitude + 180.0) % 360.0) // 6.0) + 1


def utm_epsg(zone: int, north: bool) -> int:
    return (32600 if north else 32700) + zone


def latlon_to_utm(lat, lon, zone: Optional[int] = None, north: Optional[bool] = None):
    """Convert WGS84 lat/lon (degrees) to UTM.  Returns ``(easting, northing, zone, north)``."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if zone is None:
        zone = utm_zone(float(np.nanmedian(lon)))
    if north is None:
        north = bool(np.nanmedian(lat) >= 0)
    phi = np.radians(lat)
    lam = np.radians(lon) - math.radians((zone - 1) * 6 - 180 + 3)
    lam = (lam + math.pi) % (2 * math.pi) - math.pi
    s, c = np.sin(phi), np.cos(phi)
    n = _A / np.sqrt(1 - _E2 * s**2)
    t = np.tan(phi) ** 2
    cc = _EP2 * c**2
    a = c * lam
    e4, e6 = _E2**2, _E2**3
    m = _A * (
        (1 - _E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
        - (3 * _E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
        + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi)
        - (35 * e6 / 3072) * np.sin(6 * phi)
    )
    x = _K0 * n * (a + (1 - t + cc) * a**3 / 6 + (5 - 18 * t + t**2 + 72 * cc - 58 * _EP2) * a**5 / 120) + 500000.0
    y = _K0 * (
        m
        + n
        * np.tan(phi)
        * (a**2 / 2 + (5 - t + 9 * cc + 4 * cc**2) * a**4 / 24 + (61 - 58 * t + t**2 + 600 * cc - 330 * _EP2) * a**6 / 720)
    )
    if not north:
        y = y + 10_000_000.0
    return x, y, zone, north


def utm_to_latlon(easting, northing, zone: int, north: bool = True):
    """Inverse of :func:`latlon_to_utm`.  Returns ``(lat, lon)`` in degrees."""
    x = np.asarray(easting, dtype=np.float64) - 500000.0
    y = np.asarray(northing, dtype=np.float64)
    if not north:
        y = y - 10_000_000.0
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    m = y / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * np.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * np.sin(4 * mu)
        + (151 * e1**3 / 96) * np.sin(6 * mu)
        + (1097 * e1**4 / 512) * np.sin(8 * mu)
    )
    s, c = np.sin(phi1), np.cos(phi1)
    c1 = _EP2 * c**2
    t1 = np.tan(phi1) ** 2
    n1 = _A / np.sqrt(1 - _E2 * s**2)
    r1 = _A * (1 - _E2) / (1 - _E2 * s**2) ** 1.5
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * np.tan(phi1) / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * _EP2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * _EP2 - 3 * c1**2) * d**6 / 720
    )
    lon = (d - (1 + 2 * t1 + c1) * d**3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * _EP2 + 24 * t1**2) * d**5 / 120) / c
    return np.degrees(lat), np.degrees(lon) + (zone - 1) * 6 - 180 + 3


# ---------------------------------------------------------------------------
# track geometry
# ---------------------------------------------------------------------------


@dataclass
class Track:
    """Projected sensor track in metres."""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray  # degrees true
    epsg: Optional[int] = None
    zone: Optional[int] = None
    north: bool = True


def _interp_nans(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=np.float64)
    good = np.isfinite(a)
    if good.all() or not good.any():
        return a
    idx = np.arange(a.size)
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def course_made_good(x: np.ndarray, y: np.ndarray, span: int = 5) -> np.ndarray:
    """Heading (deg) from successive positions using a centred difference over ``span`` pings."""
    n = x.size
    if n < 2:
        return np.zeros(n)
    i0 = np.clip(np.arange(n) - span, 0, n - 1)
    i1 = np.clip(np.arange(n) + span, 0, n - 1)
    hd = np.degrees(np.arctan2(x[i1] - x[i0], y[i1] - y[i0])) % 360.0
    still = (x[i1] == x[i0]) & (y[i1] == y[i0])
    if still.any() and (~still).any():
        hd[still] = np.nan
        hd = _interp_nans(hd)
    return hd


def track(
    wf: Waterfall,
    use_heading: str = "auto",
    smooth_pings: int = 5,
    apply_layback: bool = False,
    zone: Optional[int] = None,
    epsg: Optional[int] = None,
) -> Track:
    """Project and smooth the sensor track of a waterfall.

    ``use_heading``: ``"sensor"`` uses the recorded heading, ``"cog"`` derives
    it from the track, ``"auto"`` uses the sensor heading when it is present
    and non-constant.  ``apply_layback`` moves positions astern by the
    recorded layback (only needed when positions are ship, not fish, fixes).
    """
    north = True
    if wf.geographic:
        x, y, zone, north = latlon_to_utm(_interp_nans(wf.latitude), _interp_nans(wf.longitude), zone=zone)
        epsg = utm_epsg(zone, north)
    else:
        x, y = _interp_nans(wf.easting), _interp_nans(wf.northing)
    if not (np.isfinite(x).any() and np.isfinite(y).any()):
        raise ValueError("waterfall has no navigation")
    if smooth_pings > 1:
        x = _moving_average(x, smooth_pings, 0)
        y = _moving_average(y, smooth_pings, 0)
    sensor = wf.heading
    have_sensor = np.isfinite(sensor).mean() > 0.9 and np.nanstd(sensor) > 1e-6
    if use_heading == "sensor" or (use_heading == "auto" and have_sensor):
        hd = _interp_nans(sensor % 360.0)
        if smooth_pings > 1:
            rad = np.radians(hd)
            hd = np.degrees(
                np.arctan2(_moving_average(np.sin(rad), smooth_pings, 0), _moving_average(np.cos(rad), smooth_pings, 0))
            ) % 360.0
    else:
        hd = course_made_good(x, y)
    if apply_layback:
        lb = np.nan_to_num(wf.layback)
        rad = np.radians(hd)
        x = x - lb * np.sin(rad)
        y = y - lb * np.cos(rad)
    return Track(x=x, y=y, heading=hd, epsg=epsg, zone=zone, north=north)


def pixel_coordinates(wf: Waterfall, trk: Optional[Track] = None, **track_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Projected (x, y) of every pixel of ``wf.image()``, each shape (n_pings, 2*n_bins).

    Use a ground-range (slant-corrected) waterfall for correct positions.
    """
    trk = trk or track(wf, **track_kwargs)
    d = wf.across_track()[None, :]  # starboard positive
    rad = np.radians(trk.heading)[:, None]
    x = trk.x[:, None] + d * np.cos(rad)
    y = trk.y[:, None] - d * np.sin(rad)
    return x, y


# ---------------------------------------------------------------------------
# mosaicking
# ---------------------------------------------------------------------------


@dataclass
class _Layer:
    x: np.ndarray
    y: np.ndarray
    v: np.ndarray
    r: np.ndarray


@dataclass
class Mosaic:
    """Grid one or more waterfalls into a raster.

    ``method``: ``"mean"`` (average overlapping pixels), ``"max"`` or
    ``"nearest"`` (keep the pixel closest to its own track – the usual choice
    for side-scan because it avoids smearing far-range data over near-range).
    """

    cell_size: float = 0.25
    method: str = "nearest"
    epsg: Optional[int] = None
    layers: List[_Layer] = field(default_factory=list)
    _zone: Optional[int] = None
    _north: bool = True

    def add(self, wf: Waterfall, values: Optional[np.ndarray] = None, **track_kwargs) -> None:
        """Add a (ground-range, magnitude) waterfall.  ``values`` overrides
        ``wf.image()`` (e.g. a stretched uint8 image of the same shape)."""
        if not wf.ground_range:
            import warnings

            warnings.warn("adding a slant-range waterfall; run slant_range_correct first for accurate positions")
        if self._zone is not None and wf.geographic:
            track_kwargs.setdefault("zone", self._zone)
        trk = track(wf, **track_kwargs)
        if self.epsg is None:
            self.epsg = trk.epsg
        self._zone, self._north = trk.zone, trk.north
        x, y = pixel_coordinates(wf, trk)
        v = np.abs(wf.image()) if values is None else np.asarray(values)
        r = np.broadcast_to(np.abs(wf.across_track())[None, :], v.shape)
        keep = v != 0
        self.layers.append(
            _Layer(x[keep].astype(np.float64), y[keep].astype(np.float64), v[keep].astype(np.float32), r[keep].astype(np.float32))
        )

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        xs = [(l.x.min(), l.x.max()) for l in self.layers if l.x.size]
        ys = [(l.y.min(), l.y.max()) for l in self.layers if l.y.size]
        if not xs:
            raise ValueError("mosaic is empty")
        return min(a for a, _ in xs), min(a for a, _ in ys), max(b for _, b in xs), max(b for _, b in ys)

    def render(self, fill_holes: int = 1) -> "Raster":
        xmin, ymin, xmax, ymax = self.bounds
        cs = self.cell_size
        x0 = math.floor(xmin / cs) * cs
        y1 = math.ceil(ymax / cs) * cs
        nx = int(math.ceil((xmax - x0) / cs)) + 1
        ny = int(math.ceil((y1 - ymin) / cs)) + 1
        x = np.concatenate([l.x for l in self.layers])
        y = np.concatenate([l.y for l in self.layers])
        v = np.concatenate([l.v for l in self.layers])
        r = np.concatenate([l.r for l in self.layers])
        col = np.clip(((x - x0) / cs).astype(np.int64), 0, nx - 1)
        row = np.clip(((y1 - y) / cs).astype(np.int64), 0, ny - 1)
        cell = row * nx + col
        out = np.full(nx * ny, np.nan, dtype=np.float32)
        if self.method == "mean":
            s = np.bincount(cell, weights=v.astype(np.float64), minlength=nx * ny)
            c = np.bincount(cell, minlength=nx * ny)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(c > 0, s / c, np.nan)
        elif self.method == "max":
            best = np.full(nx * ny, -np.inf, dtype=np.float32)
            np.maximum.at(best, cell, v)
            out = np.where(np.isfinite(best), best, np.nan)
        elif self.method == "nearest":
            nearest = np.full(nx * ny, np.inf, dtype=np.float32)
            np.minimum.at(nearest, cell, r)
            sel = r <= nearest[cell]
            out[cell[sel]] = v[sel]
        else:
            raise ValueError("method must be 'mean', 'max' or 'nearest'")
        grid = out.reshape(ny, nx).astype(np.float32, copy=False)
        for _ in range(max(0, fill_holes)):
            grid = _fill_holes(grid)
        return Raster(grid, x0, y1, cs, self.epsg)


def _fill_holes(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells that have at least 3 valid 8-neighbours with their mean."""
    valid = np.isfinite(grid)
    g = np.where(valid, grid, np.float32(0.0))
    pv = np.pad(valid.astype(np.float32), 1)
    pg = np.pad(g, 1)
    s = np.zeros_like(g)
    c = np.zeros_like(g)
    h, w = grid.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            s += pg[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            c += pv[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    fill = ~valid & (c >= 3)
    out = grid.copy()
    out[fill] = s[fill] / c[fill]
    return out


@dataclass
class Raster:
    """A north-up raster: ``data[row, col]``, upper-left corner at (x0, y0)."""

    data: np.ndarray
    x0: float
    y0: float
    cell_size: float
    epsg: Optional[int] = None

    @property
    def shape(self):
        return self.data.shape

    def world_file(self) -> str:
        cs = self.cell_size
        return f"{cs}\n0.0\n0.0\n{-cs}\n{self.x0 + cs / 2}\n{self.y0 - cs / 2}\n"

    def to_png(self, path: str | Path, **stretch_kwargs) -> Path:
        """Write an 8-bit PNG (percentile stretched) plus a ``.pgw`` world file."""
        from .imageio import write_png
        from .processing import stretch

        path = Path(path)
        data = np.nan_to_num(self.data, nan=0.0)
        write_png(path, stretch(data, **stretch_kwargs))
        path.with_suffix(".pgw").write_text(self.world_file())
        if self.epsg:
            path.with_suffix(".epsg").write_text(f"EPSG:{self.epsg}\n")
        return path

    def to_geotiff(self, path: str | Path, dtype: str = "float32", **stretch_kwargs) -> Path:
        """Write a GeoTIFF (requires Pillow).  ``dtype="uint8"`` writes a
        stretched 8-bit image; ``"float32"`` keeps values with NaN nodata."""
        from .imageio import write_geotiff
        from .processing import stretch

        if dtype == "uint8":
            data = stretch(np.nan_to_num(self.data, nan=0.0), **stretch_kwargs)
            nodata = 0
        else:
            data = self.data.astype(np.float32)
            nodata = float("nan")
        return write_geotiff(path, data, self.x0, self.y0, self.cell_size, self.epsg, nodata)
