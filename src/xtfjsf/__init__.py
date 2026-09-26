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
from .waterfall import Waterfall, build_waterfall, waterfall_to_pings

__version__ = "0.1.0"


def read_waterfall(path, subsystem=None, **kwargs) -> Waterfall:
    """Open a file and build its waterfall in one call.

    ``subsystem`` is passed to JSF readers (e.g. 20 = low, 21 = high frequency).
    Remaining keyword arguments go to :func:`build_waterfall`.
    """
    with open_sonar(path) as f:
        pings = f.pings(subsystem=subsystem) if isinstance(f, JSFFile) else f.pings()
        return build_waterfall(pings, **kwargs)


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
    "Mosaic",
    "Ping",
    "Raster",
    "SonarFile",
    "Waterfall",
    "XTFFile",
    "XTFWriter",
    "bottom_track",
    "build_waterfall",
    "convert",
    "despeckle",
    "detect_format",
    "formats",
    "latlon_to_utm",
    "mask_water_column",
    "multilook",
    "normalize_beam_pattern",
    "open_sonar",
    "pixel_coordinates",
    "process",
    "read_waterfall",
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
