"""Format-independent data model for side-scan / SAS pings."""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

PORT = "port"
STARBOARD = "starboard"
OTHER = "other"


@dataclass
class ChannelData:
    """One channel (one side / one frequency) of one ping.

    ``samples`` is ordered from nadir outwards.  It is real for detected
    (magnitude) data and complex for analytic / SAS single-look-complex data.
    """

    samples: np.ndarray
    side: str = OTHER
    channel: int = 0
    frequency: float = math.nan  # centre frequency, Hz
    sample_interval: float = math.nan  # two-way time between samples, s
    sound_velocity: float = 1500.0  # one-way, m/s
    start_range: float = 0.0  # slant range of first sample, m
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def range_resolution(self) -> float:
        """Slant-range distance between consecutive samples in metres."""
        return self.sound_velocity * self.sample_interval / 2.0

    @property
    def slant_range(self) -> float:
        """Slant range of the far edge of the last sample, in metres."""
        return self.start_range + self.num_samples * self.range_resolution

    @property
    def ranges(self) -> np.ndarray:
        """Slant range (m) at the centre of every sample."""
        return self.start_range + (np.arange(self.num_samples) + 0.5) * self.range_resolution

    @property
    def is_complex(self) -> bool:
        return np.iscomplexobj(self.samples)

    def intensity(self) -> np.ndarray:
        """Magnitude of the samples as float32."""
        return np.abs(self.samples).astype(np.float32, copy=False)


@dataclass
class Ping:
    """A single ping, possibly with several channels, plus its navigation."""

    time: Optional[datetime] = None
    ping_number: int = 0
    channels: List[ChannelData] = field(default_factory=list)
    latitude: float = math.nan  # sensor (towfish) position, degrees
    longitude: float = math.nan
    easting: float = math.nan  # projected sensor position when the file is not geographic
    northing: float = math.nan
    heading: float = math.nan  # degrees true
    pitch: float = math.nan  # degrees
    roll: float = math.nan  # degrees
    heave: float = math.nan  # metres
    altitude: float = math.nan  # sensor height above seabed, metres
    depth: float = math.nan  # sensor depth below surface, metres
    speed: float = math.nan  # m/s
    layback: float = math.nan  # metres
    subsystem: int = 0  # format specific group id (e.g. JSF subsystem = frequency)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def channel(self, side: str) -> Optional[ChannelData]:
        for ch in self.channels:
            if ch.side == side:
                return ch
        return None

    @property
    def port(self) -> Optional[ChannelData]:
        return self.channel(PORT)

    @property
    def starboard(self) -> Optional[ChannelData]:
        return self.channel(STARBOARD)

    @property
    def timestamp(self) -> float:
        return self.time.timestamp() if self.time is not None else math.nan


class SonarFile(abc.ABC):
    """Base class for format readers.

    Readers are context managers and iterate over :class:`Ping` objects.
    """

    format_name = "unknown"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")

    # -- context manager -------------------------------------------------
    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # -- API ---------------------------------------------------------------
    @abc.abstractmethod
    def pings(self, **kwargs) -> Iterator[Ping]:
        """Yield every sonar ping in file order."""

    def __iter__(self) -> Iterator[Ping]:
        return self.pings()

    def read_all(self, **kwargs) -> List[Ping]:
        return list(self.pings(**kwargs))

    def summary(self) -> Dict[str, Any]:
        """Quick statistics about the file (reads every ping)."""
        n = 0
        t0 = t1 = None
        sides: Dict[str, int] = {}
        freqs = set()
        max_range = 0.0
        subsystems = set()
        for p in self.pings():
            n += 1
            subsystems.add(p.subsystem)
            if p.time is not None:
                t0 = p.time if t0 is None or p.time < t0 else t0
                t1 = p.time if t1 is None or p.time > t1 else t1
            for ch in p.channels:
                sides[ch.side] = sides.get(ch.side, 0) + 1
                if not math.isnan(ch.frequency):
                    freqs.add(round(ch.frequency))
                if np.isfinite(ch.slant_range):
                    max_range = max(max_range, ch.slant_range)
        return {
            "path": str(self.path),
            "format": self.format_name,
            "pings": n,
            "start": t0,
            "end": t1,
            "channels_by_side": sides,
            "frequencies_hz": sorted(freqs),
            "subsystems": sorted(subsystems),
            "max_slant_range_m": max_range,
        }
