"""Synthetic side-scan data, for tests, demos and validating processing chains."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .core import PORT, STARBOARD, ChannelData, Ping

EARTH_RADIUS = 6371000.0


def simulate_line(
    num_pings: int = 200,
    num_samples: int = 1000,
    max_range: float = 50.0,
    altitude: float | Sequence[float] = 8.0,
    frequency: float = 400e3,
    ping_rate: float = 10.0,
    speed: float = 2.0,
    heading: float = 45.0,
    start_lat: float = 50.0,
    start_lon: float = -4.0,
    start_time: Optional[datetime] = None,
    targets: Sequence[Tuple[int, float, float]] = ((100, 20.0, 2.0),),
    complex_data: bool = False,
    seed: int = 0,
    sound_velocity: float = 1500.0,
    amplitude: float = 1000.0,
) -> List[Ping]:
    """Generate a straight survey line over a flat seabed.

    ``targets`` is a list of ``(ping_index, across_track_m, height_m)``;
    positive across-track is starboard.  Each target produces a bright return
    followed by an acoustic shadow, as on real records.  The raw data carry
    ``1/R`` amplitude loss so that TVG has something to correct.
    """
    rng = np.random.default_rng(seed)
    start_time = start_time or datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    alts = np.broadcast_to(np.asarray(altitude, dtype=float), (num_pings,))
    dt = 2.0 * max_range / (sound_velocity * num_samples)
    r = (np.arange(num_samples) + 0.5) * max_range / num_samples
    step = speed / ping_rate
    hd = math.radians(heading)
    pings = []
    for i in range(num_pings):
        h = alts[i]
        along = i * step
        dn, de = along * math.cos(hd), along * math.sin(hd)
        lat = start_lat + math.degrees(dn / EARTH_RADIUS)
        lon = start_lon + math.degrees(de / (EARTH_RADIUS * math.cos(math.radians(start_lat))))
        p = Ping(
            time=start_time + timedelta(seconds=i / ping_rate),
            ping_number=i + 1,
            latitude=lat,
            longitude=lon,
            heading=heading,
            altitude=h,
            depth=20.0,
            speed=speed,
            pitch=0.0,
            roll=0.0,
            layback=0.0,
        )
        for side_no, side in enumerate((PORT, STARBOARD)):
            seabed = r > h
            with np.errstate(invalid="ignore"):
                ground = np.sqrt(np.maximum(r**2 - h**2, 0.0))
                grazing = np.where(seabed, np.arcsin(np.clip(h / r, 0, 1)), 0.0)
            # Lambert-ish backscatter with a bright first return, 1/R spreading loss
            level = np.where(seabed, 0.2 + 0.8 * np.sin(grazing) + 0.3, 0.02)
            level = level / np.maximum(r, 1.0)
            for ti, ty, th in targets:
                if (ty > 0) != (side == STARBOARD) or abs(i - ti) > 3:
                    continue
                gy = abs(ty)
                hit = np.abs(ground - gy) < 0.6
                level = np.where(hit & seabed, level * 8.0, level)
                shadow_len = gy * th / max(h - th, 0.1)
                shadow = (ground > gy + 0.6) & (ground < gy + 0.6 + shadow_len)
                level = np.where(shadow & seabed, level * 0.05, level)
            speckle = rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)
            sig = amplitude * level * speckle / math.sqrt(2)
            samples = sig.astype(np.complex64) if complex_data else np.abs(sig).astype(np.float32)
            p.channels.append(
                ChannelData(
                    samples=samples,
                    side=side,
                    channel=side_no,
                    frequency=frequency,
                    sample_interval=dt,
                    sound_velocity=sound_velocity,
                )
            )
        pings.append(p)
    return pings
