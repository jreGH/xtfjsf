# xtfjsf

A Python library for reading, processing and mosaicking **side-scan sonar** and
**synthetic aperture sonar (SAS)** data stored in the standard XTF (Triton) and
JSF (EdgeTech) formats. The only required dependency is numpy.

| Format | Read | Write | Notes |
|---|---|---|---|
| XTF (Triton eXtended) | ✅ | ✅ | Sonar, hidden-sonar, notes and attitude packets decoded; all other packet types indexed as raw bytes. 1/2/4-byte int, IEEE float and IBM float samples; >6 channel headers |
| JSF (EdgeTech) | ✅ | ✅ | Message 80 (side-scan, sub-bottom, SAS) including analytic/complex data (format 1), NMEA (2002); all other messages indexed as raw bytes |
| Anything else | plug-in | plug-in | `xtfjsf.register_format(...)` (see below) |

The readers are robust against real-world damage: they resynchronise on the
next valid packet after corrupt bytes, tolerate XTF records whose declared
length includes padding that was never written, and keep the readable part
of a truncated final record.

## Install

```bash
pip install -e .            # core (numpy only)
pip install -e ".[images]"  # + Pillow for GeoTIFF output
pip install -e ".[test]"    # + pytest
```

## Quick start

```python
import xtfjsf as xj

# Any supported file; the format is detected from content, not extension
with xj.open_sonar("line_001.jsf") as f:
    print(f.summary())
    for ping in f.pings(subsystem=21):   # JSF: 20 = LF, 21 = HF side-scan
        print(ping.time, ping.latitude, ping.longitude, ping.altitude,
              ping.port.num_samples, ping.port.range_resolution)

# Stack pings into a regular port/starboard waterfall
wf = xj.read_waterfall("line_001.xtf")          # or build_waterfall(pings)

# Standard processing chain
alt = xj.bottom_track(wf)                       # first-return altitude per ping
wf = xj.tvg(wf, spreading=20, absorption_db_per_km=60)
wf = xj.slant_range_correct(wf, alt)            # -> ground range, water column removed
wf = xj.normalize_beam_pattern(wf, mode="angle")  # empirical beam-pattern / gain flattening
wf = xj.despeckle(wf, 3)
xj.save_waterfall_png(wf, "line_001.png")

# ...or all of it in one call
wf = xj.process(xj.read_waterfall("line_001.xtf"), absorption_db_per_km=60)

# Georeferenced mosaic of several lines (UTM, EPSG chosen automatically)
mosaic = xj.Mosaic(cell_size=0.2, method="nearest")
for path in ["line_001.xtf", "line_002.jsf"]:
    mosaic.add(xj.process(xj.read_waterfall(path)))
raster = mosaic.render()
raster.to_geotiff("mosaic.tif")      # float32 GeoTIFF, NaN = no data
raster.to_png("mosaic.png")          # 8-bit PNG + .pgw world file

# Convert between formats (e.g. JSF -> XTF for other software)
xj.convert("line_001.jsf", "line_001.xtf", dtype="float32")
```

## Synthetic aperture sonar

SAS data typically arrive as high-resolution, beamformed imagery, often
complex (single-look complex, SLC). The library keeps SAS data complex until
you decide otherwise:

```python
wf = xj.read_waterfall("sas.jsf", keep_complex=True)   # complex64 port/starboard
print(wf.is_complex, wf.resolution)
ml = xj.multilook(wf, along=2, across=2)               # incoherent speckle reduction
img = xj.process(wf, looks=(2, 2))                     # multilook + full chain
```

The library does not include beamforming of raw per-element SAS data
(micronavigation, back-projection). That step is sensor-specific and needs
the array geometry and navigation that come with the vendor's raw formats.

## Command line

```bash
xtfjsf info *.xtf *.jsf                      # JSON summary: pings, times, channels, record types
xtfjsf nav line.jsf -o nav.csv                # per-ping navigation / attitude to CSV
xtfjsf waterfall line.xtf line.png --absorption 60 --despeckle 3
xtfjsf waterfall sas.jsf sas.png --looks-along 2 --looks-across 2
xtfjsf mosaic *.xtf -o mosaic.tif --cell-size 0.25
xtfjsf convert line.jsf line.xtf --dtype float32
```

## Data model

* `Ping`: time (UTC), ping number, sensor position (lat/lon or projected
  easting/northing), heading, pitch, roll, heave, altitude, depth, speed,
  layback and a list of `ChannelData`. The full decoded header is in
  `ping.metadata`.
* `ChannelData`: samples ordered from nadir outwards (real or complex), side
  (`"port"`, `"starboard"`, `"other"`), frequency, sample interval, sound
  velocity and start range. It also exposes `range_resolution`, `slant_range`
  and `ranges`.
* `Waterfall`: `(n_pings, n_bins)` port and starboard arrays on a common grid,
  plus per-ping navigation arrays. `image()` gives the classic port|starboard
  view.

## Adding a format

```python
class MyReader(xj.SonarFile):
    format_name = "sdf"
    def pings(self, **kw):
        ...  # yield xj.Ping objects

xj.register_format("sdf", MyReader, detect=lambda head: head[:4] == b"\xff\xff\xff\xff",
                   extensions=(".sdf",))
```

Everything downstream (waterfalls, processing, mosaics, CLI) then works with it.

## Troubleshooting channel sides

XTF writers disagree on how channels are labelled. Some set every channel
header's `ChannelNumber` to 0, some count from 1, and some leave
`TypeOfChannel` or `BytesPerSample` unset. The reader works out the layout
from the data itself, and assigns port and starboard from `ChannelNumber`,
then from packet order, then from the even = port / odd = starboard
convention. If an image still looks one-sided, run:

```bash
xtfjsf info line.xtf     # see "first_ping_channels": side, channel numbers, samples, range
```

`build_waterfall` also warns when a whole side is empty. You can override the
choice with `build_waterfall(pings, port_channel=0, starboard_channel=1)`.

## Validation status

* XTF was cross-checked against the independent `pyxtf` library: files written
  here read correctly in pyxtf, and files written by pyxtf read correctly here.
* JSF follows EdgeTech's JSF specification (message header and 240-byte
  message-80 header). It has so far only been tested with files written by this
  library, so please check it against a few of your own files (`xtfjsf info`,
  `xtfjsf waterfall --raw`). JSF coordinate unit 4 is assumed to mean
  degrees × 10⁷.
* The synthetic-data generator (`xtfjsf.simulate`) produces lines with known
  altitude, targets and shadows. The tests use it to check bottom tracking,
  slant-range correction and mosaic geolocation (targets land within 1.5 m).

```bash
pytest
```
