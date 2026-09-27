"""Command line interface: ``xtfjsf info|nav|waterfall|mosaic|convert``."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import (
    JSFFile,
    Mosaic,
    build_waterfall,
    combine_sides,
    convert,
    open_sonar,
    process,
    save_waterfall_png,
)


def _pings(f, subsystem):
    return f.pings(subsystem=subsystem) if isinstance(f, JSFFile) else f.pings()


def _load(path, args, side=None):
    with open_sonar(path) as f:
        return build_waterfall(
            _pings(f, args.subsystem),
            frequency=args.frequency * 1000 if args.frequency else None,
            keep_complex=True,
            side=side,
        )


def _load_input(args):
    """Waterfall for `waterfall`: one file, or a port file + --starboard file."""
    if args.starboard:
        return combine_sides(_load(args.file, args, "port"), _load(args.starboard, args, "starboard"))
    return _load(args.file, args, getattr(args, "side", None))


def _process(wf, args):
    return process(
        wf,
        slant_correct=not args.no_slant,
        spreading=args.spreading,
        absorption_db_per_km=args.absorption,
        normalize=None if args.normalize == "none" else args.normalize,
        despeckle_size=args.despeckle,
        looks=(args.looks_along, args.looks_across),
    )


def _table_row(path) -> dict:
    with open_sonar(path) as f:
        n, t0, t1, first = 0, None, None, None
        for p in f.pings():
            n += 1
            first = first or p
            if p.time is not None:
                t0 = p.time if t0 is None else min(t0, p.time)
                t1 = p.time if t1 is None else max(t1, p.time)
        records = f.record_counts()
        if isinstance(f, JSFFile):
            source = "subsystems " + ",".join(map(str, f.subsystems))
        else:
            h = f.header.fields
            source = " ".join(x for x in (h.get("RecordingProgramName"), h.get("SonarName")) if x)
    chans = []
    for c in first.channels if first else []:
        fmt = "cplx" if c.is_complex else str(c.samples.dtype)
        freq = f"{c.frequency / 1000:.0f}kHz" if math.isfinite(c.frequency) else "?kHz"
        chans.append(f"{c.side[:4]} {freq} {c.num_samples}x{fmt} {c.range_resolution * 100:.1f}cm {c.slant_range:.0f}m")
    return {
        "file": Path(path).name,
        "format": f.format_name,
        "source": source or "-",
        "pings": str(n),
        "start": t0.strftime("%Y-%m-%d %H:%M:%S") if t0 else "-",
        "dur_s": f"{(t1 - t0).total_seconds():.0f}" if t0 and t1 else "-",
        "records": ",".join(f"{k}:{v}" for k, v in sorted(records.items())),
        "channels (first ping)": "; ".join(chans) or "-",
    }


def cmd_info(args) -> int:
    if args.table:
        rows = [_table_row(p) for p in args.files]
        cols = list(rows[0]) if rows else []
        width = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
        print("  ".join(c.ljust(width[c]) for c in cols))
        for r in rows:
            print("  ".join(r[c].ljust(width[c]) for c in cols))
        return 0
    for path in args.files:
        with open_sonar(path) as f:
            info = f.summary()
            info["records"] = f.record_counts()
            if isinstance(f, JSFFile):
                info["subsystems_present"] = f.subsystems
            else:
                info["sonar_name"] = f.header.fields.get("SonarName")
                info["channels"] = [
                    {k: c[k] for k in ("TypeOfChannel", "ChannelName", "Frequency", "BytesPerSample", "SampleFormat")}
                    for c in f.header.channels
                ]
            with open_sonar(path) as f2:
                first = next(iter(f2.pings()), None)
            info["first_ping_channels"] = [
                {
                    "side": c.side,
                    "channel": c.channel,
                    "header_channel_number": c.metadata.get("ChannelNumber"),
                    "packet_index": c.metadata.get("packet_index"),
                    "samples": c.num_samples,
                    "bytes_per_sample": c.metadata.get("bytes_per_sample"),
                    "frequency_hz": c.frequency,
                    "range_resolution_m": c.range_resolution,
                    "slant_range_m": c.slant_range,
                    "complex": c.is_complex,
                }
                for c in (first.channels if first else [])
            ]
        for k in ("start", "end"):
            if info[k] is not None:
                info[k] = info[k].isoformat()
        print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_nav(args) -> int:
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        w = csv.writer(out)
        w.writerow(["file", "time", "ping", "subsystem", "latitude", "longitude", "easting", "northing",
                    "heading", "altitude", "depth", "speed", "pitch", "roll"])
        for path in args.files:
            with open_sonar(path) as f:
                for p in _pings(f, args.subsystem):
                    w.writerow([
                        Path(path).name,
                        p.time.isoformat() if p.time else "",
                        p.ping_number,
                        p.subsystem,
                        *("" if not math.isfinite(v) else f"{v:.8f}" if i < 2 else f"{v:.3f}"
                          for i, v in enumerate((p.latitude, p.longitude, p.easting, p.northing, p.heading,
                                                 p.altitude, p.depth, p.speed, p.pitch, p.roll))),
                    ])
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def cmd_waterfall(args) -> int:
    wf = _load_input(args)
    wf = wf.magnitude() if args.raw else _process(wf, args)
    save_waterfall_png(wf, args.output, low=args.low, high=args.high, log=args.log)
    print(f"wrote {args.output}: {wf.num_pings} pings x {2 * wf.num_bins} bins @ {wf.resolution:.3f} m")
    return 0


def cmd_mosaic(args) -> int:
    m = Mosaic(cell_size=args.cell_size, method=args.method)
    inputs = [(p, None) for p in args.files] + [(p, "port") for p in args.port] + [(p, "starboard") for p in args.starboard]
    if not inputs:
        raise SystemExit("mosaic: no input files")
    for path, side in inputs:
        wf = _process(_load(path, args, side), args)
        m.add(wf, smooth_pings=args.smooth)
        print(f"added {path}: {wf.num_pings} pings", file=sys.stderr)
    r = m.render(fill_holes=args.fill)
    out = Path(args.output)
    if out.suffix.lower() in (".tif", ".tiff"):
        r.to_geotiff(out, dtype="uint8" if args.uint8 else "float32", low=args.low, high=args.high, log=args.log)
    else:
        r.to_png(out, low=args.low, high=args.high, log=args.log)
    print(f"wrote {out}: {r.shape[1]} x {r.shape[0]} cells @ {args.cell_size} m, EPSG:{r.epsg}")
    return 0


def cmd_convert(args) -> int:
    kwargs = {}
    if Path(args.output).suffix.lower() == ".xtf":
        kwargs["dtype"] = args.dtype
    n = convert(args.input, args.output, subsystem=args.subsystem, **kwargs)
    print(f"wrote {n} pings to {args.output}")
    return 0


def _add_processing_args(p):
    g = p.add_argument_group("processing")
    g.add_argument("--no-slant", action="store_true", help="skip slant-range correction")
    g.add_argument("--spreading", type=float, default=20.0, help="TVG spreading coefficient (dB/decade)")
    g.add_argument("--absorption", type=float, default=0.0, help="TVG absorption (dB/km)")
    g.add_argument("--normalize", choices=["angle", "range", "none"], default="angle")
    g.add_argument("--despeckle", type=int, default=0, help="median filter size (0 = off)")
    g.add_argument("--looks-along", type=int, default=1, help="SAS multilook factor along track")
    g.add_argument("--looks-across", type=int, default=1, help="SAS multilook factor across track")
    g.add_argument("--low", type=float, default=1.0, help="lower percentile for contrast stretch")
    g.add_argument("--high", type=float, default=99.5, help="upper percentile for contrast stretch")
    g.add_argument("--log", action="store_true", help="log-scale before stretching")


def _add_select_args(p):
    p.add_argument("--subsystem", type=int, default=None, help="JSF subsystem (20 low, 21 high frequency...)")
    p.add_argument("--frequency", type=float, default=None, help="pick channel nearest this frequency (kHz)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="xtfjsf", description="Side-scan / SAS sonar file tools (XTF, JSF, ...)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="summarise files")
    p.add_argument("files", nargs="+")
    p.add_argument("--table", action="store_true", help="one line per file, to compare many files")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("nav", help="export per-ping navigation to CSV")
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--output")
    p.add_argument("--subsystem", type=int, default=None)
    p.set_defaults(func=cmd_nav)

    p = sub.add_parser("waterfall", help="render a (processed) waterfall PNG")
    p.add_argument("file", help="input file (the port file when --starboard is given)")
    p.add_argument("output")
    p.add_argument("--starboard", metavar="FILE", help="separate starboard file to combine with FILE (port)")
    p.add_argument("--side", choices=["port", "starboard"], help="FILE holds only this side")
    p.add_argument("--raw", action="store_true", help="no processing, just stack the pings")
    _add_select_args(p)
    _add_processing_args(p)
    p.set_defaults(func=cmd_waterfall)

    p = sub.add_parser("mosaic", help="georeferenced mosaic (.tif GeoTIFF or .png + world file)")
    p.add_argument("files", nargs="*", help="two-sided files, or single-side files labelled correctly")
    p.add_argument("--port", nargs="+", default=[], metavar="FILE", help="files holding only the port side")
    p.add_argument("--starboard", nargs="+", default=[], metavar="FILE", help="files holding only the starboard side")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--cell-size", type=float, default=0.25)
    p.add_argument("--method", choices=["nearest", "mean", "max"], default="nearest")
    p.add_argument("--smooth", type=int, default=5, help="navigation smoothing window (pings)")
    p.add_argument("--fill", type=int, default=1, help="hole filling passes")
    p.add_argument("--uint8", action="store_true", help="write a stretched 8-bit GeoTIFF")
    _add_select_args(p)
    _add_processing_args(p)
    p.set_defaults(func=cmd_mosaic)

    p = sub.add_parser("convert", help="convert between formats (e.g. JSF -> XTF)")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--subsystem", type=int, default=None)
    p.add_argument("--dtype", default="float32", choices=["uint8", "uint16", "uint32", "int16", "int32", "float32"],
                   help="XTF sample type")
    p.set_defaults(func=cmd_convert)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
