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


def build_waterfall(
    pings: Iterable[Ping],
    frequency: Optional[float] = None,
    port_channel: Optional[int] = None,
    starboard_channel: Optional[int] = None,
    resolution: Optional[float] = None,
    max_range: Optional[float] = None,
    keep_complex: bool = False,
) -> Waterfall:
    """Stack pings into a :class:`Waterfall`.

    Each channel is resampled onto a common slant-range grid so that range
    changes, differing sample counts and differing sample rates within a line
    are handled.  By default the grid uses the finest sample spacing found and
    the longest range.  ``frequency`` (Hz) selects the nearest channel on
    multi-frequency systems; ``port_channel``/``starboard_channel`` select by
    channel number instead.
    """
    rows: List[tuple] = []
    nav: List[tuple] = []
    freqs = []
    res_seen = []
    range_seen = []
    any_complex = False
    for p in pings:
        pc = _pick_channel(p, PORT, frequency, port_channel)
        sc = _pick_channel(p, STARBOARD, frequency, starboard_channel)
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
