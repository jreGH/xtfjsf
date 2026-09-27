"""xtfjsf - read, process and mosaic side-scan and synthetic aperture sonar data.

Quick start::

    import xtfjsf as xj

    with xj.open_sonar("line.xtf") as f:         # or .jsf - detected from content
        wf = xj.build_waterfall(f.pings())
    wf = xj.process(wf, spreading=20, absorption_db_per_km=60)
    xj.save_waterfall_png(wf, "line.png")

    mosaic = xj.Mosaic(cell_size=0.2)
    mosaic.add(wf)
    mosaic.render().to_geotiff("line.tif")
"""

from pathlib import Path

from .core import OTHER, PORT, STARBOARD, ChannelData, Ping, SonarFile
from .formats import (
    JSFFile,
    JSFWriter,
    XTFFile,
    XTFWriter,
    detect_format,
    formats,
    open_sonar,
    register_format,
    writer_for,
)
from .georef import Mosaic, Raster, latlon_to_utm, pixel_coordinates, track, utm_to_latlon
from .pairing import MatchResult, MissionPair, match_mission_files
from .processing import (
    bottom_track,
    despeckle,
    mask_water_column,
    multilook,
    normalize_beam_pattern,
    process,
    slant_range_correct,
    stretch,
    to_db,
    tvg,
)
from .waterfall import Waterfall, build_waterfall, combine_sides, waterfall_to_pings

__version__ = "0.1.0"


def read_waterfall(path, subsystem=None, **kwargs) -> Waterfall:
    """Open a file and build its waterfall in one call.

    ``subsystem`` is passed to JSF readers (e.g. 20 = low, 21 = high frequency).
    Remaining keyword arguments go to :func:`build_waterfall`; pass
    ``side="port"`` / ``side="starboard"`` for a file holding only one side.
    """
    with open_sonar(path) as f:
        pings = f.pings(subsystem=subsystem) if isinstance(f, JSFFile) else f.pings()
        return build_waterfall(pings, **kwargs)


def read_waterfall_pair(
    port_path,
    starboard_path,
    subsystem=None,
    max_time_diff=None,
    reverse_port=False,
    reverse_starboard=False,
    **kwargs,
) -> Waterfall:
    """Build one two-sided waterfall from separate port and starboard files.

    Each file is read with its side forced (so a mislabelled single-side file
    still lands on the right side) and pings are paired by time; see
    :func:`combine_sides`.  ``reverse_port``/``reverse_starboard`` correct a
    channel recorded back-to-front (far range first); see
    :func:`build_waterfall`.
    """
    kwargs.pop("side", None)
    port = read_waterfall(port_path, subsystem=subsystem, side=PORT, reverse_port=reverse_port, **kwargs)
    stbd = read_waterfall(starboard_path, subsystem=subsystem, side=STARBOARD, reverse_starboard=reverse_starboard, **kwargs)
    return combine_sides(port, stbd, max_time_diff=max_time_diff)


def save_waterfall_png(wf: Waterfall, path, **stretch_kwargs) -> Path:
    """Write the port|starboard waterfall as an 8-bit PNG (first ping at the top)."""
    from .imageio import write_png

    return write_png(path, stretch(wf.image(), **stretch_kwargs))


def convert(src, dst, dst_format=None, subsystem=None, **writer_kwargs) -> int:
    """Copy all pings from ``src`` into a new file (format from ``dst`` extension).

    Returns the number of pings written.
    """
    with open_sonar(src) as f:
        pings = f.pings(subsystem=subsystem) if isinstance(f, JSFFile) else f.pings()
        with writer_for(dst, dst_format, **writer_kwargs) as w:
            for p in pings:
                w.write_ping(p)
            return w.pings_written


__all__ = [
    "OTHER",
    "PORT",
    "STARBOARD",
    "ChannelData",
    "JSFFile",
    "JSFWriter",
    "MatchResult",
    "MissionPair",
    "Mosaic",
    "Ping",
    "Raster",
    "SonarFile",
    "Waterfall",
    "XTFFile",
    "XTFWriter",
    "bottom_track",
    "build_waterfall",
    "combine_sides",
    "convert",
    "despeckle",
    "detect_format",
    "formats",
    "latlon_to_utm",
    "match_mission_files",
    "mask_water_column",
    "multilook",
    "normalize_beam_pattern",
    "open_sonar",
    "pixel_coordinates",
    "process",
    "read_waterfall",
    "read_waterfall_pair",
    "register_format",
    "save_waterfall_png",
    "slant_range_correct",
    "stretch",
    "to_db",
    "track",
    "tvg",
    "utm_to_latlon",
    "waterfall_to_pings",
    "writer_for",
]
