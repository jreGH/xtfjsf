"""Side-scan / SAS image processing.

All functions take and return :class:`~xtfjsf.waterfall.Waterfall` objects
(or plain arrays where noted) and never modify their input in place.  A
typical chain is::

    wf = build_waterfall(pings)
    alt = bottom_track(wf)
    wf = slant_range_correct(wf, alt)
    wf = tvg(wf, spreading=30, absorption_db_per_km=40)
    wf = normalize_beam_pattern(wf, mode="angle", altitude=alt)
    img = stretch(wf.image())
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .waterfall import Waterfall


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _moving_average(a: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Centred moving average with edge-shrinking windows (no scipy needed)."""
    if size <= 1:
        return a.astype(np.float64, copy=True)
    a = np.moveaxis(np.asarray(a, dtype=np.float64), axis, 0)
    n = a.shape[0]
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)], axis=0)
    half = size // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + (size - half), 0, n)
    counts = (hi - lo).reshape((-1,) + (1,) * (a.ndim - 1))
    out = (c[hi] - c[lo]) / counts
    return np.moveaxis(out, 0, axis)


def _median_filter_1d(a: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or a.size == 0:
        return a.copy()
    half = size // 2
    padded = np.pad(a, (half, size - 1 - half), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.nanmedian(win, axis=1)


def median_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
    """2-D median filter with edge replication (pure numpy, processed in row blocks)."""
    if size <= 1:
        return img.copy()
    half = size // 2
    padded = np.pad(img, ((half, size - 1 - half), (half, size - 1 - half)), mode="edge")
    out = np.empty(img.shape, dtype=np.result_type(img.dtype, np.float32))
    block = max(1, int(4_000_000 // max(1, img.shape[1] * size * size)))
    for r0 in range(0, img.shape[0], block):
        r1 = min(img.shape[0], r0 + block)
        win = np.lib.stride_tricks.sliding_window_view(padded[r0 : r1 + size - 1], (size, size))
        out[r0:r1] = np.median(win.reshape(win.shape[0], win.shape[1], -1), axis=-1)
    return out


def to_db(a: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Amplitude to decibels: ``20 * log10(|a|)``."""
    return (20.0 * np.log10(np.maximum(np.abs(a), floor))).astype(np.float32)


def from_db(a: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(a) / 20.0).astype(np.float32)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _valid_altitude(wf: Waterfall, altitude: Optional[np.ndarray]) -> np.ndarray:
    if altitude is None:
        altitude = wf.altitude
    altitude = np.asarray(altitude, dtype=np.float64)
    if altitude.shape != (wf.num_pings,):
        raise ValueError("altitude must have one value per ping")
    return altitude


def slant_ranges(wf: Waterfall, altitude: Optional[np.ndarray] = None) -> np.ndarray:
    """Slant range (m) of every bin, shape (n_pings, n_bins)."""
    r = wf.ranges[None, :]
    if not wf.ground_range:
        return np.broadcast_to(r, (wf.num_pings, wf.num_bins))
    h = np.nan_to_num(_valid_altitude(wf, altitude), nan=0.0)[:, None]
    return np.sqrt(r**2 + h**2)


def grazing_angles(wf: Waterfall, altitude: Optional[np.ndarray] = None) -> np.ndarray:
    """Angle from nadir (degrees) of every bin; NaN inside the water column."""
    h = _valid_altitude(wf, altitude)[:, None]
    r = wf.ranges[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        if wf.ground_range:
            ang = np.degrees(np.arctan2(r, h))
        else:
            ratio = h / r
            ang = np.where(ratio <= 1.0, np.degrees(np.arccos(np.clip(ratio, -1, 1))), np.nan)
    return ang


# ---------------------------------------------------------------------------
# bottom tracking / slant range correction
# ---------------------------------------------------------------------------


def bottom_track(
    wf: Waterfall,
    threshold: float = 0.35,
    min_altitude: float = 0.5,
    max_altitude: Optional[float] = None,
    smooth_m: float = 0.3,
    filter_pings: int = 9,
    use_sensor_altitude: bool = False,
) -> np.ndarray:
    """Estimate sensor altitude (m) per ping from the first seabed return.

    The port and starboard magnitudes are summed, smoothed over ``smooth_m``
    metres and normalised by a robust per-ping reference level (90th
    percentile).  The first bin exceeding ``threshold`` times that level
    (beyond ``min_altitude``) is the bottom.  Outliers are removed with a
    running median over ``filter_pings`` pings.  With ``use_sensor_altitude``
    valid altimeter values in the file take precedence.
    """
    if wf.ground_range:
        raise ValueError("bottom tracking requires slant-range data")
    prof = np.abs(wf.port).astype(np.float64) + np.abs(wf.starboard).astype(np.float64)
    prof = _moving_average(prof, max(1, int(round(smooth_m / wf.resolution))), axis=1)
    i0 = int(math.ceil(min_altitude / wf.resolution))
    i1 = wf.num_bins if max_altitude is None else min(wf.num_bins, int(max_altitude / wf.resolution) + 1)
    seg = prof[:, i0:i1]
    alt = np.full(wf.num_pings, np.nan)
    if seg.shape[1] == 0:
        return alt
    ref = np.percentile(prof[:, i0:], 90, axis=1)
    floor = np.percentile(seg[:, : max(1, seg.shape[1] // 50)], 50, axis=1)
    level = floor + threshold * (ref - floor)
    above = seg >= level[:, None]
    has = above.any(axis=1) & (ref > floor)
    first = np.argmax(above, axis=1)
    alt[has] = (first[has] + i0) * wf.resolution
    if filter_pings > 1:
        alt = _median_filter_1d(alt, filter_pings)
    if use_sensor_altitude:
        good = np.isfinite(wf.altitude) & (wf.altitude > 0)
        alt = np.where(good, wf.altitude, alt)
    return alt


def slant_range_correct(
    wf: Waterfall,
    altitude: Optional[np.ndarray] = None,
    resolution: Optional[float] = None,
) -> Waterfall:
    """Project slant-range data onto a flat-seabed ground-range grid.

    ``altitude`` defaults to the file's altimeter values; pings with no valid
    altitude are bottom-tracked.  The water column is removed.
    """
    if wf.ground_range:
        return wf
    alt = np.array(wf.altitude if altitude is None else altitude, dtype=np.float64)
    bad = ~np.isfinite(alt) | (alt <= 0)
    if bad.any():
        tracked = bottom_track(wf)
        alt[bad] = tracked[bad]
    alt = np.nan_to_num(alt, nan=0.0)
    alt = np.clip(alt, 0.0, wf.max_range)
    res = resolution or wf.resolution
    max_ground = math.sqrt(max(wf.max_range**2 - float(np.min(alt)) ** 2, 0.0))
    n_bins = max(1, int(round(max_ground / res)))
    g = (np.arange(n_bins) + 0.5) * res
    src = wf.ranges
    out_dtype = wf.port.dtype
    port = np.zeros((wf.num_pings, n_bins), dtype=out_dtype)
    stbd = np.zeros((wf.num_pings, n_bins), dtype=out_dtype)
    for i in range(wf.num_pings):
        s = np.sqrt(g**2 + alt[i] ** 2)
        for dst, row in ((port, wf.port[i]), (stbd, wf.starboard[i])):
            if np.iscomplexobj(row):
                dst[i] = np.interp(s, src, row.real, right=0) + 1j * np.interp(s, src, row.imag, right=0)
            else:
                dst[i] = np.interp(s, src, row, right=0)
    return wf.with_data(port, stbd, resolution=res, ground_range=True, altitude=alt)


def mask_water_column(wf: Waterfall, altitude: Optional[np.ndarray] = None, fill: float = 0.0) -> Waterfall:
    """Zero (or ``fill``) slant-range bins closer than the seabed."""
    if wf.ground_range:
        return wf
    alt = _valid_altitude(wf, altitude)
    mask = wf.ranges[None, :] < np.nan_to_num(alt, nan=0.0)[:, None]
    port = np.where(mask, fill, wf.port).astype(wf.port.dtype)
    stbd = np.where(mask, fill, wf.starboard).astype(wf.starboard.dtype)
    return wf.with_data(port, stbd)


# ---------------------------------------------------------------------------
# radiometric corrections
# ---------------------------------------------------------------------------


def tvg(
    wf: Waterfall,
    spreading: float = 20.0,
    absorption_db_per_km: float = 0.0,
    altitude: Optional[np.ndarray] = None,
    reference_range: float = 1.0,
) -> Waterfall:
    """Apply time-varied gain ``spreading*log10(R/R0) + 2*alpha*R`` (dB, amplitude).

    Typical values: ``spreading`` 20 (cylindrical/area-normalised) to 40
    (spherical two-way); ``absorption_db_per_km`` ~ 10 at 100 kHz, ~ 60 at
    400 kHz, ~ 150 at 900 kHz in sea water.  Use ``remove_tvg`` style
    negative values to undo an onboard TVG.
    """
    r = np.maximum(slant_ranges(wf, altitude), 1e-3)
    gain_db = spreading * np.log10(r / reference_range) + 2.0 * absorption_db_per_km * r / 1000.0
    g = np.power(10.0, gain_db / 20.0).astype(np.float32)
    return wf.with_data(wf.port * g, wf.starboard * g)


def normalize_beam_pattern(
    wf: Waterfall,
    mode: str = "range",
    altitude: Optional[np.ndarray] = None,
    window: Optional[int] = None,
    angle_bin_deg: float = 0.5,
    statistic: str = "median",
    target: Optional[float] = None,
    max_samples: int = 2_000_000,
) -> Waterfall:
    """Empirical gain / beam-pattern normalisation (a.k.a. AGC, "flattening").

    ``mode="range"`` divides every across-track column by its ``statistic``
    (median or mean) over all pings, or by its mean over a running ``window``
    of pings, which follows gain changes along the line.  ``mode="angle"``
    uses the response per grazing-angle bin instead, which stays correct when
    altitude varies; it needs altitude.  The median default keeps bright
    targets and shadows from biasing the curve.  Port
    and starboard are normalised separately and rescaled to a common
    ``target`` level (default: overall mean).
    """
    mag_p = np.abs(wf.port).astype(np.float64)
    mag_s = np.abs(wf.starboard).astype(np.float64)
    if target is None:
        valid = np.concatenate([mag_p[mag_p > 0], mag_s[mag_s > 0]])
        target = float(np.mean(valid)) if valid.size else 1.0
    eps = 1e-12

    if mode == "range":

        def curve(m):
            mm = np.where(m > 0, m, np.nan)
            if window:
                num = _moving_average(np.nan_to_num(mm), window, axis=0)
                den = _moving_average(np.isfinite(mm).astype(float), window, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return num / den
            with np.errstate(all="ignore"):
                c = np.nanmedian(mm, axis=0) if statistic == "median" else np.nanmean(mm, axis=0)
            return np.broadcast_to(c, m.shape)

        cp, cs = curve(mag_p), curve(mag_s)
    elif mode == "angle":
        ang = grazing_angles(wf, altitude)
        nb = int(math.ceil(90.0 / angle_bin_deg)) + 1
        idx = np.clip(np.nan_to_num(ang, nan=-1) / angle_bin_deg, -1, nb - 1).astype(int)

        def curve(m):
            ok = (idx >= 0) & (m > 0)
            bins, vals = idx[ok], m[ok]
            if bins.size > max_samples:  # a subset is plenty to estimate <=181 bins
                step = bins.size // max_samples + 1
                bins, vals = bins[::step], vals[::step]
            cnts = np.bincount(bins, minlength=nb)
            if statistic == "median":
                order = np.lexsort((vals, bins))
                starts = np.concatenate([[0], np.cumsum(cnts)[:-1]])
                mid = np.minimum(starts + (cnts - 1) // 2, max(vals.size - 1, 0))
                table = np.where(cnts > 0, vals[order][mid] if vals.size else 0.0, np.nan)
            else:
                sums = np.bincount(bins, weights=vals, minlength=nb)
                with np.errstate(invalid="ignore", divide="ignore"):
                    table = sums / cnts
            # fill empty angle bins by interpolation
            good = np.isfinite(table)
            if good.any():
                table = np.interp(np.arange(nb), np.flatnonzero(good), table[good])
            return np.where(idx >= 0, table[np.maximum(idx, 0)], np.nan)

        cp, cs = curve(mag_p), curve(mag_s)
    else:
        raise ValueError("mode must be 'range' or 'angle'")

    with np.errstate(invalid="ignore", divide="ignore"):
        gp = np.where(np.isfinite(cp) & (cp > eps), target / cp, 0.0).astype(np.float32)
        gs = np.where(np.isfinite(cs) & (cs > eps), target / cs, 0.0).astype(np.float32)
    return wf.with_data(wf.port * gp, wf.starboard * gs)


def despeckle(wf: Waterfall, size: int = 3) -> Waterfall:
    """Median filter the magnitude images (``size`` x ``size`` kernel)."""
    return wf.magnitude().apply(lambda a: median_filter(a, size).astype(np.float32))


def multilook(wf: Waterfall, along: int = 1, across: int = 2) -> Waterfall:
    """Incoherent block averaging of intensity (|x|^2) to reduce speckle.

    Standard for SAS single-look-complex imagery; reduces the along-track
    sample count by ``along`` and the across-track bin count by ``across``.
    Returns amplitude (sqrt of mean power).
    """
    along, across = max(1, int(along)), max(1, int(across))

    def look(a):
        p = np.abs(a).astype(np.float64) ** 2
        n0 = (p.shape[0] // along) * along
        n1 = (p.shape[1] // across) * across
        p = p[:n0, :n1].reshape(n0 // along, along, n1 // across, across).mean(axis=(1, 3))
        return np.sqrt(p).astype(np.float32)

    sub = wf.select(slice(0, (wf.num_pings // along) * along, along)) if along > 1 else wf
    return sub.with_data(look(wf.port), look(wf.starboard), resolution=wf.resolution * across)


def stretch(
    img: np.ndarray,
    low: float = 1.0,
    high: float = 99.5,
    log: bool = False,
    gamma: float = 1.0,
    ignore_zeros: bool = True,
) -> np.ndarray:
    """Percentile contrast stretch of an image to uint8 (0-255)."""
    a = np.abs(np.asarray(img)).astype(np.float64)
    if log:
        a = np.log10(np.maximum(a, 1e-6))
        zero_mask = np.asarray(img) == 0
    else:
        zero_mask = a == 0
    sample = a[~zero_mask] if ignore_zeros and (~zero_mask).any() else a.ravel()
    if sample.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    if sample.size > 2_000_000:
        sample = sample[:: sample.size // 2_000_000 + 1]
    lo, hi = np.percentile(sample, [low, high])
    if hi <= lo:
        hi = lo + 1e-12
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        out = out ** (1.0 / gamma)
    out = (out * 255.0 + 0.5).astype(np.uint8)
    if ignore_zeros:
        out[zero_mask] = 0
    return out


def process(
    wf: Waterfall,
    slant_correct: bool = True,
    spreading: float = 20.0,
    absorption_db_per_km: float = 0.0,
    normalize: Optional[str] = "angle",
    despeckle_size: int = 0,
    looks: tuple = (1, 1),
) -> Waterfall:
    """Convenience pipeline: bottom track -> slant correct -> TVG -> normalise
    -> optional multilook / despeckle."""
    if tuple(looks) != (1, 1):
        wf = multilook(wf, *looks)
    wf = wf.magnitude()
    alt = None
    if not wf.ground_range:
        sensor = wf.altitude
        alt = np.where(np.isfinite(sensor) & (sensor > 0), sensor, np.nan)
        if not np.isfinite(alt).all():
            tracked = bottom_track(wf)
            alt = np.where(np.isfinite(alt), alt, tracked)
    if spreading or absorption_db_per_km:
        wf = tvg(wf, spreading, absorption_db_per_km, altitude=alt)
    if slant_correct and not wf.ground_range:
        wf = slant_range_correct(wf, alt)
    if normalize:
        norm_alt = wf.altitude if normalize == "angle" else None
        if normalize == "angle" and not np.isfinite(norm_alt).any():
            normalize = "range"
        wf = normalize_beam_pattern(wf, mode=normalize, altitude=norm_alt)
    if despeckle_size and despeckle_size > 1:
        wf = despeckle(wf, despeckle_size)
    return wf
