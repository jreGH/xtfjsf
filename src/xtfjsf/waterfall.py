"""Assemble pings into regularly-sampled port/starboard waterfall arrays."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, List, Optional

import numpy as np

from .core import PORT, STARBOARD, ChannelData, Ping


@dataclass
class Waterfall:
    """Port and starboard intensities on a common across-track grid.

    ``port`` and ``starboard`` have shape ``(n_pings, n_bins)`` and are both
    ordered from nadir outwards.  ``resolution`` is the bin size in metres and
    ``ground_range`` tells whether the across-track axis is slant range (the
    raw data) or horizontal ground range (after slant-range correction).
    Complex SAS data are kept complex until :meth:`magnitude` is called.
    """

    port: np.ndarray
    starboard: np.ndarray
    resolution: float
    time: np.ndarray  # POSIX seconds
    ping_number: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    easting: np.ndarray
    northing: np.ndarray
    heading: np.ndarray
    altitude: np.ndarray
    depth: np.ndarray
    speed: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    layback: np.ndarray
    ground_range: bool = False
    frequency: float = math.nan
    metadata: dict = field(default_factory=dict)

    # -- geometry ------------------------------------------------------
    @property
    def num_pings(self) -> int:
        return self.port.shape[0]

    @property
    def num_bins(self) -> int:
        return self.port.shape[1]

    @property
    def ranges(self) -> np.ndarray:
        """Across-track distance (m) at the centre of every bin."""
        return (np.arange(self.num_bins) + 0.5) * self.resolution

    @property
    def max_range(self) -> float:
        return self.num_bins * self.resolution

    @property
    def is_complex(self) -> bool:
        return np.iscomplexobj(self.port) or np.iscomplexobj(self.starboard)

    # -- conversions ---------------------------------------------------
    def image(self) -> np.ndarray:
        """Classic waterfall: port (mirrored) | starboard, shape (n_pings, 2*n_bins)."""
        return np.concatenate([self.port[:, ::-1], self.starboard], axis=1)

    def across_track(self) -> np.ndarray:
        """Signed across-track offset for each column of :meth:`image` (port < 0)."""
        r = self.ranges
        return np.concatenate([-r[::-1], r])

    def with_data(self, port: np.ndarray, starboard: np.ndarray, **changes) -> "Waterfall":
        return replace(self, port=port, starboard=starboard, **changes)

    def magnitude(self) -> "Waterfall":
        return self.with_data(np.abs(self.port).astype(np.float32), np.abs(self.starboard).astype(np.float32))

    def apply(self, func: Callable[[np.ndarray], np.ndarray]) -> "Waterfall":
        """Apply ``func`` to port and starboard arrays independently."""
        return self.with_data(func(self.port), func(self.starboard))

    def select(self, index) -> "Waterfall":
        """Subset pings with a slice, boolean mask or integer index array."""
        kw = {}
        for name in self._PER_PING:
            kw[name] = getattr(self, name)[index]
        return replace(self, port=self.port[index], starboard=self.starboard[index], **kw)

    _PER_PING = (
        "time",
        "ping_number",
        "latitude",
        "longitude",
        "easting",
        "northing",
        "heading",
        "altitude",
        "depth",
        "speed",
        "pitch",
        "roll",
        "layback",
    )

    @property
    def geographic(self) -> bool:
        return bool(np.isfinite(self.latitude).any() and np.isfinite(self.longitude).any())

    def navigation_table(self) -> np.ndarray:
        """Per-ping navigation as a numpy structured array."""
        dtype = [(n, "f8") for n in self._PER_PING]
        out = np.empty(self.num_pings, dtype=dtype)
        for n in self._PER_PING:
            out[n] = getattr(self, n)
        return out


def _pick_channel(ping: Ping, side: str, frequency: Optional[float], channel: Optional[int]) -> Optional[ChannelData]:
    candidates = [c for c in ping.channels if c.side == side]
    if channel is not None:
        candidates = [c for c in ping.channels if c.channel == channel]
    if not candidates:
        return None
    if frequency is not None:
        return min(candidates, key=lambda c: abs((c.frequency if math.isfinite(c.frequency) else 0) - frequency))
    return candidates[0]


def _pick_single(ping: Ping, side: str, frequency: Optional[float], channel: Optional[int]) -> Optional[ChannelData]:
    """The one channel of a single-side file: by number, else the channel
    labelled ``side``, else the first channel with data."""
    chans = [c for c in ping.channels if c.num_samples]
    if channel is not None:
        chans = [c for c in chans if c.channel == channel]
    if not chans:
        return None
    if frequency is not None:
        return min(chans, key=lambda c: abs((c.frequency if math.isfinite(c.frequency) else 0) - frequency))
    labelled = [c for c in chans if c.side == side]
    return (labelled or chans)[0]


def _resample(ch: ChannelData, grid: np.ndarray, out_dtype) -> np.ndarray:
    """Interpolate a channel onto slant-range bin centres ``grid``."""
    if ch.num_samples == 0 or not math.isfinite(ch.range_resolution) or ch.range_resolution <= 0:
        return np.zeros(grid.shape, dtype=out_dtype)
    # Extend by the half-sample at either end so each sample covers its whole
    # range cell; outside [start_range, slant_range] there is no data (0).
    src = np.concatenate([[ch.start_range], ch.ranges, [ch.slant_range]])

    def interp(v):
        v = np.concatenate([v[:1], v, v[-1:]]).astype(np.float64)
        return np.interp(grid, src, v, left=0.0, right=0.0)

    if np.iscomplexobj(ch.samples) and np.iscomplexobj(np.empty(0, out_dtype)):
        return (interp(ch.samples.real) + 1j * interp(ch.samples.imag)).astype(out_dtype)
    vals = np.abs(ch.samples) if np.iscomplexobj(ch.samples) else ch.samples
    return interp(vals).astype(out_dtype)


def _warn_missing(rows) -> None:
    import warnings

    for k, side in ((0, PORT), (1, STARBOARD)):
        missing = sum(r[k] is None for r in rows)
        if missing == len(rows):
            warnings.warn(f"no {side} channel in any ping: that half of the waterfall will be empty. "
                          f"If this file holds only one side (some systems write one file per side), "
                          f"use read_waterfall_pair()/combine_sides() or `--starboard FILE` on the command "
                          f"line; otherwise check the channel sides with `xtfjsf info`.")
        elif missing:
            warnings.warn(f"{missing} of {len(rows)} pings have no {side} channel")
    untimed = sum(1 for r in rows for c in r if c is not None and not (c.range_resolution > 0))
    if untimed:
        warnings.warn(f"{untimed} channel(s) have no sample-interval / range information and are left empty")


def build_waterfall(
    pings: Iterable[Ping],
    frequency: Optional[float] = None,
    port_channel: Optional[int] = None,
    starboard_channel: Optional[int] = None,
    resolution: Optional[float] = None,
    max_range: Optional[float] = None,
    keep_complex: bool = False,
    side: Optional[str] = None,
) -> Waterfall:
    """Stack pings into a :class:`Waterfall`.

    Each channel is resampled onto a common slant-range grid so that range
    changes, differing sample counts and differing sample rates within a line
    are handled.  By default the grid uses the finest sample spacing found and
    the longest range.  ``frequency`` (Hz) selects the nearest channel on
    multi-frequency systems; ``port_channel``/``starboard_channel`` select by
    channel number instead.

    ``side="port"`` or ``side="starboard"`` is for files holding a single side
    (some systems write one file per side): one channel per ping is taken,
    whatever the file labels it, and placed on that side; the other side is
    left empty.  Combine two such waterfalls with :func:`combine_sides`.
    """
    if side not in (None, PORT, STARBOARD):
        raise ValueError("side must be None, 'port' or 'starboard'")
    rows: List[tuple] = []
    nav: List[tuple] = []
    freqs = []
    res_seen = []
    range_seen = []
    any_complex = False
    for p in pings:
        if side is None:
            pc = _pick_channel(p, PORT, frequency, port_channel)
            sc = _pick_channel(p, STARBOARD, frequency, starboard_channel)
        else:
            wanted = port_channel if side == PORT else starboard_channel
            ch = _pick_single(p, side, frequency, wanted)
            pc, sc = (ch, None) if side == PORT else (None, ch)
        if pc is None and sc is None:
            continue
        for c in (pc, sc):
            if c is not None and c.num_samples:
                if math.isfinite(c.range_resolution) and c.range_resolution > 0:
                    res_seen.append(c.range_resolution)
                    range_seen.append(c.slant_range)
                if math.isfinite(c.frequency):
                    freqs.append(c.frequency)
                any_complex |= c.is_complex
        rows.append((pc, sc))
        nav.append(
            (
                p.timestamp,
                p.ping_number,
                p.latitude,
                p.longitude,
                p.easting,
                p.northing,
                p.heading,
                p.altitude,
                p.depth,
                p.speed,
                p.pitch,
                p.roll,
                p.layback,
            )
        )
    if not rows:
        raise ValueError("no pings with port or starboard channels found")
    if side is None:
        _warn_missing(rows)
    if resolution is None:
        resolution = float(np.min(res_seen)) if res_seen else 1.0
    if max_range is None:
        max_range = float(np.max(range_seen)) if range_seen else resolution * max(
            (c.num_samples for r in rows for c in r if c is not None), default=1
        )
    n_bins = max(1, int(round(max_range / resolution)))
    grid = (np.arange(n_bins) + 0.5) * resolution
    dtype = np.complex64 if (keep_complex and any_complex) else np.float32
    port = np.zeros((len(rows), n_bins), dtype=dtype)
    stbd = np.zeros((len(rows), n_bins), dtype=dtype)
    for i, (pc, sc) in enumerate(rows):
        if pc is not None:
            port[i] = _resample(pc, grid, dtype)
        if sc is not None:
            stbd[i] = _resample(sc, grid, dtype)
    navarr = np.array(nav, dtype=np.float64)
    cols = {name: navarr[:, i] for i, name in enumerate(Waterfall._PER_PING)}
    return Waterfall(
        port=port,
        starboard=stbd,
        resolution=resolution,
        frequency=float(np.median(freqs)) if freqs else math.nan,
        **cols,
    )


def waterfall_to_pings(wf: Waterfall, sound_velocity: float = 1500.0) -> List[Ping]:
    """Turn a (processed) waterfall back into pings, e.g. to write an XTF."""
    from datetime import datetime, timezone

    out = []
    dt = 2.0 * wf.resolution / sound_velocity
    for i in range(wf.num_pings):
        t = wf.time[i]
        p = Ping(
            time=datetime.fromtimestamp(t, tz=timezone.utc) if math.isfinite(t) else None,
            ping_number=int(wf.ping_number[i]),
            latitude=wf.latitude[i],
            longitude=wf.longitude[i],
            easting=wf.easting[i],
            northing=wf.northing[i],
            heading=wf.heading[i],
            altitude=wf.altitude[i],
            depth=wf.depth[i],
            speed=wf.speed[i],
            pitch=wf.pitch[i],
            roll=wf.roll[i],
            layback=wf.layback[i],
        )
        for ch_no, (side, data) in enumerate(((PORT, wf.port[i]), (STARBOARD, wf.starboard[i]))):
            p.channels.append(
                ChannelData(
                    samples=data,
                    side=side,
                    channel=ch_no,
                    frequency=wf.frequency,
                    sample_interval=dt,
                    sound_velocity=sound_velocity,
                )
            )
        out.append(p)
    return out


def _regrid(rows: np.ndarray, src_res: float, dst_res: float, n_bins: int) -> np.ndarray:
    """Resample (n, m) rows from bin size ``src_res`` onto ``n_bins`` bins of ``dst_res``."""
    if rows.shape[1] == n_bins and abs(src_res - dst_res) < 1e-12:
        return rows
    src = np.concatenate([[0.0], (np.arange(rows.shape[1]) + 0.5) * src_res, [rows.shape[1] * src_res]])
    dst = (np.arange(n_bins) + 0.5) * dst_res
    out = np.zeros((rows.shape[0], n_bins), dtype=rows.dtype)

    def interp(v):
        return np.interp(dst, src, np.concatenate([v[:1], v, v[-1:]]), left=0.0, right=0.0)

    for i, r in enumerate(rows):
        out[i] = interp(r.real) + 1j * interp(r.imag) if np.iscomplexobj(rows) else interp(r)
    return out


def combine_sides(
    port: Waterfall,
    starboard: Waterfall,
    max_time_diff: Optional[float] = None,
    resolution: Optional[float] = None,
) -> Waterfall:
    """Merge a port-only and a starboard-only waterfall into one.

    Use this when a system writes each side to its own file.  Pings are
    paired by time (nearest starboard ping within ``max_time_diff`` seconds,
    default half the median ping interval), or by ping number when the files
    carry no usable times.  Unpaired pings are kept with the other side empty.
    Only the port half of ``port`` and the starboard half of ``starboard``
    are used; if one of the files holds its data on the other half (for
    example it was read without ``side=``), that half is used instead.
    """
    if port.ground_range != starboard.ground_range:
        raise ValueError("both waterfalls must be slant range, or both ground range")

    def half(wf: Waterfall, want: str) -> np.ndarray:
        data = wf.port if want == PORT else wf.starboard
        other = wf.starboard if want == PORT else wf.port
        return other if not np.any(data) and np.any(other) else data

    p_rows, s_rows = half(port, PORT), half(starboard, STARBOARD)
    res = resolution or min(port.resolution, starboard.resolution)
    n_bins = max(1, int(round(max(port.max_range, starboard.max_range) / res)))
    dtype = np.result_type(p_rows.dtype, s_rows.dtype)
    p_rows = _regrid(p_rows.astype(dtype), port.resolution, res, n_bins)
    s_rows = _regrid(s_rows.astype(dtype), starboard.resolution, res, n_bins)

    # pair pings: port index -> starboard index (or -1)
    tp, ts = port.time, starboard.time
    match = np.full(port.num_pings, -1)
    if np.isfinite(tp).all() and np.isfinite(ts).all() and starboard.num_pings:
        if max_time_diff is None:
            ref = tp if port.num_pings > 1 else ts
            max_time_diff = 0.5 * float(np.median(np.diff(np.sort(ref)))) if ref.size > 1 else np.inf
        order = np.argsort(ts)
        tss = ts[order]
        pos = np.searchsorted(tss, tp)
        lo = np.clip(pos - 1, 0, len(tss) - 1)
        hi = np.clip(pos, 0, len(tss) - 1)
        nearest = order[np.where(np.abs(tss[lo] - tp) <= np.abs(tss[hi] - tp), lo, hi)]
        ok = np.abs(ts[nearest] - tp) <= max_time_diff
        match[ok] = nearest[ok]
    else:
        lookup = {int(n): i for i, n in enumerate(starboard.ping_number)}
        match = np.array([lookup.get(int(n), -1) for n in port.ping_number])
    # each starboard ping is used once (keep the closest pairing)
    if (match >= 0).any():
        used = {}
        for i in np.flatnonzero(match >= 0):
            j = match[i]
            if j in used:
                k = used[j]
                if abs(ts[j] - tp[i]) < abs(ts[j] - tp[k]):
                    match[k] = -1
                    used[j] = i
                else:
                    match[i] = -1
            else:
                used[j] = i
    unmatched_s = np.setdiff1d(np.arange(starboard.num_pings), match[match >= 0])

    n = port.num_pings + len(unmatched_s)
    out_p = np.zeros((n, n_bins), dtype=dtype)
    out_s = np.zeros((n, n_bins), dtype=dtype)
    out_p[: port.num_pings] = p_rows
    has = match >= 0
    out_s[: port.num_pings][has] = s_rows[match[has]]
    out_s[port.num_pings :] = s_rows[unmatched_s]
    nav = {}
    for name in Waterfall._PER_PING:
        a, b = getattr(port, name), getattr(starboard, name)
        col = np.concatenate([a, b[unmatched_s]]).astype(np.float64)
        # fill port-side nav gaps from the paired starboard ping
        gap = ~np.isfinite(col[: port.num_pings]) & has
        col[: port.num_pings][gap] = b[match[gap]]
        nav[name] = col
    order = np.argsort(nav["time"], kind="stable") if np.isfinite(nav["time"]).all() else np.arange(n)
    freqs = [f for f in (port.frequency, starboard.frequency) if math.isfinite(f)]
    return Waterfall(
        port=out_p[order],
        starboard=out_s[order],
        resolution=res,
        ground_range=port.ground_range,
        frequency=float(np.mean(freqs)) if freqs else math.nan,
        metadata={"paired_pings": int(has.sum()), "port_only": int((~has).sum()), "starboard_only": int(len(unmatched_s))},
        **{k: v[order] for k, v in nav.items()},
    )
