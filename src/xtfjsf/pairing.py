"""Pair port and starboard files recorded from the same survey line.

Some systems write each side of a line to its own file, and there is no
general rule for how the two filenames relate to each other. What *is*
reliable is that pings logged at the same instant on both sides carry the
same timestamps, so pairing is done purely from each file's recorded ping
times, never from filenames.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .formats import open_sonar

TimeBounds = Tuple[Optional[datetime], Optional[datetime], int]


@dataclass
class MissionPair:
    """A port file and starboard file believed to be the same survey line."""

    port: Path
    starboard: Path
    overlap_start: Optional[datetime]
    overlap_end: Optional[datetime]
    port_pings: int
    starboard_pings: int

    @property
    def overlap_seconds(self) -> float:
        if self.overlap_start is None or self.overlap_end is None:
            return 0.0
        return (self.overlap_end - self.overlap_start).total_seconds()


@dataclass
class MatchResult:
    pairs: List[MissionPair]
    unmatched_port: List[Path]
    unmatched_starboard: List[Path]

    def __iter__(self):
        return iter(self.pairs)


def _bounds(path) -> TimeBounds:
    with open_sonar(path) as f:
        return f.time_bounds()


def _overlap_score(a: TimeBounds, b: TimeBounds) -> Optional[float]:
    """Seconds of time overlap between two files; negative is the gap when
    they don't overlap.  ``None`` when either file has no timed pings."""
    a0, a1, _ = a
    b0, b1, _ = b
    if a0 is None or b0 is None:
        return None
    lo, hi = max(a0, b0), min(a1, b1)
    if hi >= lo:
        return (hi - lo).total_seconds()
    gap = max(a0 - b1, b0 - a1)
    return -gap.total_seconds()


def match_mission_files(
    port_paths: Sequence,
    starboard_paths: Sequence,
    tolerance_s: float = 5.0,
) -> MatchResult:
    """Pair each port file with the starboard file recorded at the same time.

    Every file's ping-time range is read once (cheaply: packet headers only,
    no sample data). Pairs are then formed greedily, largest time overlap
    first, so that a file is never claimed by two lines even when several
    lines' time windows are close together. A pair whose files don't
    overlap at all is still accepted if the gap between them is within
    ``tolerance_s`` (clock drift between the two loggers).

    Returns a :class:`MatchResult`; ``unmatched_port``/``unmatched_starboard``
    list files that had no acceptable counterpart (check these by hand).
    """
    port_paths = [Path(p) for p in port_paths]
    starboard_paths = [Path(p) for p in starboard_paths]
    p_bounds: Dict[Path, TimeBounds] = {p: _bounds(p) for p in port_paths}
    s_bounds: Dict[Path, TimeBounds] = {p: _bounds(p) for p in starboard_paths}

    candidates = []
    for pp in port_paths:
        for sp in starboard_paths:
            score = _overlap_score(p_bounds[pp], s_bounds[sp])
            if score is not None and score >= -tolerance_s:
                candidates.append((score, pp, sp))
    candidates.sort(key=lambda c: -c[0])

    used_p, used_s = set(), set()
    pairs = []
    for score, pp, sp in candidates:
        if pp in used_p or sp in used_s:
            continue
        used_p.add(pp)
        used_s.add(sp)
        p0, p1, pn = p_bounds[pp]
        s0, s1, sn = s_bounds[sp]
        overlap_start = max(p0, s0) if p0 is not None and s0 is not None else None
        overlap_end = min(p1, s1) if p1 is not None and s1 is not None else None
        pairs.append(MissionPair(pp, sp, overlap_start, overlap_end, pn, sn))

    pairs.sort(key=lambda m: (m.overlap_start is None, m.overlap_start or m.port))
    return MatchResult(
        pairs=pairs,
        unmatched_port=[p for p in port_paths if p not in used_p],
        unmatched_starboard=[p for p in starboard_paths if p not in used_s],
    )
