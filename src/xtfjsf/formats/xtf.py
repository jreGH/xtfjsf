"""Triton eXtended Triton Format (XTF) reader and writer.

Implements the file header, CHANINFO, sonar (type 0), notes (type 1) and
attitude (type 3) packets of the Triton XTF specification.  All other packet
types are indexed and exposed as raw bytes so nothing in the file is lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .._binary import BinaryRecord
from ..core import OTHER, PORT, STARBOARD, ChannelData, Ping, SonarFile

KNOTS_TO_MS = 0.514444
FILE_FORMAT_ID = 0x7B
MAGIC = 0xFACE
MAGIC_BYTES = b"\xce\xfa"

# Packet header types (subset of the Triton list, used for naming only).
HEADER_TYPES = {
    0: "sonar",
    1: "notes",
    2: "bathy",
    3: "attitude",
    4: "forward",
    5: "elac",
    6: "raw_serial",
    7: "embed_head",
    8: "hidden_sonar",
    9: "seaview_angles",
    10: "seaview_depths",
    11: "highspeed_sensor",
    12: "echostrength",
    13: "georec",
    14: "klein_raw_bathy",
    15: "highspeed_sensor2",
    16: "elac_xse",
    17: "bathy_xyza",
    18: "k5000_bathy_iq",
    19: "bathy_snippet",
    20: "gps",
    21: "stat",
    22: "singlebeam",
    23: "gyro",
    24: "trackpoint",
    25: "multibeam",
    26: "q_singlebeam",
    27: "q_multitx",
    28: "q_multibeam",
    50: "time",
    60: "benthos_caati_sara",
    61: "7125",
    62: "7125_snippet",
    65: "qinsy_r2sonic_bathy",
    66: "qinsy_r2sonic_fts",
    68: "r2sonic_bathy",
    69: "r2sonic_fts",
    73: "coda_echoscope_image",
    74: "edgetech_4600",
    78: "reson_7018_watercolumn",
    100: "position",
    102: "bathy_proc",
    103: "attitude_proc",
    104: "singlebeam_proc",
    105: "aux_proc",
    106: "klein3000_data_page",
    107: "pos_raw_navigation",
    108: "kleinv4_data_page",
    200: "user_defined",
}
SONAR_TYPES = (0, 8)

FILE_HEADER = BinaryRecord(
    "XTFFILEHEADER",
    [
        ("FileFormat", "B"),
        ("SystemType", "B"),
        ("RecordingProgramName", "8s"),
        ("RecordingProgramVersion", "8s"),
        ("SonarName", "16s"),
        ("SonarType", "H"),
        ("NoteString", "64s"),
        ("ThisFileName", "64s"),
        ("NavUnits", "H"),
        ("NumberOfSonarChannels", "H"),
        ("NumberOfBathymetryChannels", "H"),
        ("NumberOfSnippetChannels", "B"),
        ("NumberOfForwardLookArrays", "B"),
        ("NumberOfEchoStrengthChannels", "H"),
        ("NumberOfInterferometryChannels", "B"),
        ("_Reserved1", "B"),
        ("_Reserved2", "H"),
        ("ReferencePointHeight", "f"),
        ("ProjectionType", "12s"),
        ("SpheriodType", "10s"),
        ("NavigationLatency", "i"),
        ("OriginY", "f"),
        ("OriginX", "f"),
        ("NavOffsetY", "f"),
        ("NavOffsetX", "f"),
        ("NavOffsetZ", "f"),
        ("NavOffsetYaw", "f"),
        ("MRUOffsetY", "f"),
        ("MRUOffsetX", "f"),
        ("MRUOffsetZ", "f"),
        ("MRUOffsetYaw", "f"),
        ("MRUOffsetPitch", "f"),
        ("MRUOffsetRoll", "f"),
    ],
    size=256,
)

CHAN_INFO = BinaryRecord(
    "CHANINFO",
    [
        ("TypeOfChannel", "B"),
        ("SubChannelNumber", "B"),
        ("CorrectionFlags", "H"),
        ("UniPolar", "H"),
        ("BytesPerSample", "H"),
        ("_Reserved", "I"),
        ("ChannelName", "16s"),
        ("VoltScale", "f"),
        ("Frequency", "f"),
        ("HorizBeamAngle", "f"),
        ("TiltAngle", "f"),
        ("BeamWidth", "f"),
        ("OffsetX", "f"),
        ("OffsetY", "f"),
        ("OffsetZ", "f"),
        ("OffsetYaw", "f"),
        ("OffsetPitch", "f"),
        ("OffsetRoll", "f"),
        ("BeamsPerArray", "H"),
        ("SampleFormat", "B"),
        ("_ReservedArea2", "53s"),
    ],
    size=128,
)

PACKET_HEADER = BinaryRecord(
    "XTFPACKETHEADER",
    [
        ("MagicNumber", "H"),
        ("HeaderType", "B"),
        ("SubChannelNumber", "B"),
        ("NumChansToFollow", "H"),
        ("_Reserved1", "4s"),
        ("NumBytesThisRecord", "I"),
    ],
    size=14,
)

PING_HEADER = BinaryRecord(
    "XTFPINGHEADER",
    [
        ("MagicNumber", "H"),
        ("HeaderType", "B"),
        ("SubChannelNumber", "B"),
        ("NumChansToFollow", "H"),
        ("_Reserved1", "4s"),
        ("NumBytesThisRecord", "I"),
        ("Year", "H"),
        ("Month", "B"),
        ("Day", "B"),
        ("Hour", "B"),
        ("Minute", "B"),
        ("Second", "B"),
        ("HSeconds", "B"),
        ("JulianDay", "H"),
        ("EventNumber", "I"),
        ("PingNumber", "I"),
        ("SoundVelocity", "f"),
        ("OceanTide", "f"),
        ("_Reserved2", "I"),
        ("ConductivityFreq", "f"),
        ("TemperatureFreq", "f"),
        ("PressureFreq", "f"),
        ("PressureTemp", "f"),
        ("Conductivity", "f"),
        ("WaterTemperature", "f"),
        ("Pressure", "f"),
        ("ComputedSoundVelocity", "f"),
        ("MagX", "f"),
        ("MagY", "f"),
        ("MagZ", "f"),
        ("AuxVal1", "f"),
        ("AuxVal2", "f"),
        ("AuxVal3", "f"),
        ("AuxVal4", "f"),
        ("AuxVal5", "f"),
        ("AuxVal6", "f"),
        ("SpeedLog", "f"),
        ("Turbidity", "f"),
        ("ShipSpeed", "f"),
        ("ShipGyro", "f"),
        ("ShipYcoordinate", "d"),
        ("ShipXcoordinate", "d"),
        ("ShipAltitude", "H"),
        ("ShipDepth", "H"),
        ("FixTimeHour", "B"),
        ("FixTimeMinute", "B"),
        ("FixTimeSecond", "B"),
        ("FixTimeHsecond", "B"),
        ("SensorSpeed", "f"),
        ("KP", "f"),
        ("SensorYcoordinate", "d"),
        ("SensorXcoordinate", "d"),
        ("SonarStatus", "H"),
        ("RangeToFish", "H"),
        ("BearingToFish", "H"),
        ("CableOut", "H"),
        ("Layback", "f"),
        ("CableTension", "f"),
        ("SensorDepth", "f"),
        ("SensorPrimaryAltitude", "f"),
        ("SensorAuxAltitude", "f"),
        ("SensorPitch", "f"),
        ("SensorRoll", "f"),
        ("SensorHeading", "f"),
        ("Heave", "f"),
        ("Yaw", "f"),
        ("AttitudeTimeTag", "I"),
        ("DOT", "f"),
        ("NavFixMilliseconds", "I"),
        ("ComputerClockHour", "B"),
        ("ComputerClockMinute", "B"),
        ("ComputerClockSecond", "B"),
        ("ComputerClockHsec", "B"),
        ("FishPositionDeltaX", "h"),
        ("FishPositionDeltaY", "h"),
        ("FishPositionErrorCode", "B"),
        ("OptionalOffsey", "I"),
        ("CableOutHundredths", "B"),
        ("_ReservedSpace2", "6s"),
    ],
    size=256,
)

PING_CHAN_HEADER = BinaryRecord(
    "XTFPINGCHANHEADER",
    [
        ("ChannelNumber", "H"),
        ("DownsampleMethod", "H"),
        ("SlantRange", "f"),
        ("GroundRange", "f"),
        ("TimeDelay", "f"),
        ("TimeDuration", "f"),
        ("SecondsPerPing", "f"),
        ("ProcessingFlags", "H"),
        ("Frequency", "H"),
        ("InitialGainCode", "H"),
        ("GainCode", "H"),
        ("BandWidth", "H"),
        ("ContactNumber", "I"),
        ("ContactClassification", "H"),
        ("ContactSubNumber", "B"),
        ("ContactType", "B"),
        ("NumSamples", "I"),
        ("MillivoltScale", "H"),
        ("ContactTimeOffTrack", "f"),
        ("ContactCloseNumber", "B"),
        ("_Reserved2", "B"),
        ("FixedVSOP", "f"),
        ("Weight", "h"),
        ("_ReservedSpace", "4s"),
    ],
    size=64,
)

ATTITUDE = BinaryRecord(
    "XTFATTITUDEDATA",
    [
        ("MagicNumber", "H"),
        ("HeaderType", "B"),
        ("SubChannelNumber", "B"),
        ("NumChansToFollow", "H"),
        ("_Reserved1", "4s"),
        ("NumBytesThisRecord", "I"),
        ("_Reserved2", "8s"),
        ("Pitch", "f"),
        ("Roll", "f"),
        ("Heave", "f"),
        ("Yaw", "f"),
        ("TimeTag", "I"),
        ("Heading", "f"),
        ("Year", "H"),
        ("Month", "B"),
        ("Day", "B"),
        ("Hour", "B"),
        ("Minutes", "B"),
        ("Seconds", "B"),
        ("Milliseconds", "H"),
        ("_Reserved3", "9s"),
    ],
    size=64,
)

NOTES = BinaryRecord(
    "XTFNOTESHEADER",
    [
        ("MagicNumber", "H"),
        ("HeaderType", "B"),
        ("SubChannelNumber", "B"),
        ("NumChansToFollow", "H"),
        ("_Reserved1", "4s"),
        ("NumBytesThisRecord", "I"),
        ("Year", "H"),
        ("Month", "B"),
        ("Day", "B"),
        ("Hour", "B"),
        ("Minute", "B"),
        ("Second", "B"),
        ("_ReservedBytes", "35s"),
        ("NotesText", "200s"),
    ],
    size=256,
)

# CHANINFO.SampleFormat -> numpy dtype.  0 means "legacy": derive the type from
# BytesPerSample as unsigned integers.
_SAMPLE_FORMATS = {
    2: np.dtype("<i4"),
    3: np.dtype("<i2"),
    5: np.dtype("<f4"),
    8: np.dtype("i1"),
}
_LEGACY_BY_BYTES = {1: np.dtype("u1"), 2: np.dtype("<u2"), 4: np.dtype("<u4"), 8: np.dtype("<f8")}


def _ibm32_to_float(raw: np.ndarray) -> np.ndarray:
    """Convert big-endian-free IBM System/360 32-bit floats (as uint32) to float64."""
    raw = raw.astype(np.uint32)
    sign = np.where(raw >> 31, -1.0, 1.0)
    exponent = ((raw >> 24) & 0x7F).astype(np.int32) - 64
    mantissa = (raw & 0x00FFFFFF).astype(np.float64) / float(1 << 24)
    return sign * mantissa * np.power(16.0, exponent)


def _decode_samples(buf: memoryview, offset: int, n: int, info: Dict[str, Any]) -> np.ndarray:
    fmt = info.get("SampleFormat", 0)
    nbytes = info.get("BytesPerSample", 2) or 2
    if fmt == 1:
        raw = np.frombuffer(buf, dtype="<u4", count=n, offset=offset)
        return _ibm32_to_float(raw).astype(np.float32)
    dtype = _SAMPLE_FORMATS.get(fmt) or _LEGACY_BY_BYTES.get(nbytes, np.dtype("<u2"))
    return np.frombuffer(buf, dtype=dtype, count=n, offset=offset).copy()


def _safe_datetime(year, month, day, hour, minute, second, micro=0) -> Optional[datetime]:
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc) + timedelta(
            microseconds=micro
        )
    except (ValueError, OverflowError):
        return None


@dataclass
class XTFRecordIndex:
    offset: int
    header_type: int
    sub_channel: int
    num_chans: int
    size: int

    @property
    def type_name(self) -> str:
        return HEADER_TYPES.get(self.header_type, f"type_{self.header_type}")


@dataclass
class XTFHeader:
    fields: Dict[str, Any]
    channels: List[Dict[str, Any]] = field(default_factory=list)
    size: int = 1024

    @property
    def geographic(self) -> bool:
        """True when navigation is stored as longitude/latitude (NavUnits == 3)."""
        return self.fields.get("NavUnits", 3) == 3

    @property
    def num_sonar_channels(self) -> int:
        return int(self.fields.get("NumberOfSonarChannels", 0))


class XTFFile(SonarFile):
    """Reader for Triton XTF files.

    >>> with XTFFile("line.xtf") as xtf:
    ...     for ping in xtf:
    ...         print(ping.time, ping.port.num_samples)
    """

    format_name = "xtf"

    def __init__(self, path: str | Path):
        super().__init__(path)
        self.header = self._read_header()
        self._index: Optional[List[XTFRecordIndex]] = None

    # -- header ------------------------------------------------------------
    def _read_header(self) -> XTFHeader:
        self._fh.seek(0)
        block = self._fh.read(1024)
        if len(block) < 1024:
            raise ValueError(f"{self.path}: too short to be an XTF file")
        fields = FILE_HEADER.unpack(block)
        if fields["FileFormat"] != FILE_FORMAT_ID:
            raise ValueError(f"{self.path}: not an XTF file (FileFormat={fields['FileFormat']:#x})")
        n_total = (
            fields["NumberOfSonarChannels"]
            + fields["NumberOfBathymetryChannels"]
            + fields["NumberOfSnippetChannels"]
            + fields["NumberOfForwardLookArrays"]
            + fields["NumberOfEchoStrengthChannels"]
            + fields["NumberOfInterferometryChannels"]
        )
        data = bytearray(block)
        size = 1024
        if n_total > 6:
            extra_blocks = math.ceil((n_total - 6) / 8)
            more = self._fh.read(1024 * extra_blocks)
            data += more
            size += 1024 * extra_blocks
        channels = []
        for i in range(max(n_total, 0)):
            if i < 6:
                off = 256 + i * 128
            else:
                off = 1024 + (i - 6) * 128
            if off + 128 > len(data):
                break
            channels.append(CHAN_INFO.unpack(data, off))
        # Some writers put the first packet right after the channel infos
        # instead of padding to 1024-byte blocks; verify and fall back.
        self._fh.seek(size)
        if self._fh.read(2) != MAGIC_BYTES:
            nxt = self._find_magic(256 + 128 * min(n_total, 6))
            if nxt is not None:
                size = nxt
        return XTFHeader(fields=fields, channels=channels, size=size)

    def _find_magic(self, start: int) -> Optional[int]:
        self._fh.seek(start)
        chunk_size = 1 << 20
        pos = start
        while True:
            chunk = self._fh.read(chunk_size + 1)
            if len(chunk) < 2:
                return None
            i = chunk.find(MAGIC_BYTES)
            if i >= 0 and i + 1 < len(chunk):
                return pos + i
            pos += chunk_size
            self._fh.seek(pos)

    # -- index -------------------------------------------------------------
    @property
    def index(self) -> List[XTFRecordIndex]:
        if self._index is None:
            self._index = list(self._scan())
        return self._index

    def _scan(self) -> Iterator[XTFRecordIndex]:
        fh = self._fh
        pos = self.header.size
        fh.seek(0, 2)
        end = fh.tell()
        while pos + PACKET_HEADER.size <= end:
            fh.seek(pos)
            h = PACKET_HEADER.unpack(fh.read(PACKET_HEADER.size))
            size = h["NumBytesThisRecord"]
            if h["MagicNumber"] == MAGIC and pos + size > end and self._find_magic(pos + 2) is None:
                # Final record truncated (e.g. logging stopped): keep what is there.
                size = end - pos
            if h["MagicNumber"] != MAGIC or size < PACKET_HEADER.size or pos + size > end:
                # Corrupt/truncated packet: resynchronise on the next magic number.
                nxt = self._find_magic(pos + 1)
                if nxt is None:
                    return
                pos = nxt
                continue
            if pos + size < end:
                fh.seek(pos + size)
                if fh.read(2) != MAGIC_BYTES:
                    # Declared length does not land on the next packet (some
                    # writers omit the 64-byte padding but keep the padded
                    # length).  Trust the next magic number if it comes earlier.
                    nxt = self._find_magic(pos + PACKET_HEADER.size)
                    if nxt is not None and nxt < pos + size:
                        size = nxt - pos
            yield XTFRecordIndex(pos, h["HeaderType"], h["SubChannelNumber"], h["NumChansToFollow"], size)
            pos += size

    def record_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.index:
            counts[r.type_name] = counts.get(r.type_name, 0) + 1
        return counts

    def read_raw(self, rec: XTFRecordIndex) -> bytes:
        self._fh.seek(rec.offset)
        return self._fh.read(rec.size)

    # -- packets -----------------------------------------------------------
    def records(self, types: Optional[Sequence[int]] = None) -> Iterator[tuple]:
        """Yield ``(XTFRecordIndex, parsed)`` for every packet.

        ``parsed`` is a :class:`Ping` for sonar packets, a dict for notes and
        attitude packets and raw ``bytes`` for everything else.
        """
        for rec in self.index:
            if types is not None and rec.header_type not in types:
                continue
            buf = self.read_raw(rec)
            if rec.header_type in SONAR_TYPES and len(buf) >= PING_HEADER.size:
                yield rec, self._parse_ping(buf)
            elif rec.header_type == 3 and len(buf) >= ATTITUDE.size:
                yield rec, ATTITUDE.unpack(buf)
            elif rec.header_type == 1 and len(buf) >= NOTES.size:
                yield rec, NOTES.unpack(buf)
            else:
                yield rec, buf

    def pings(self, include_hidden: bool = False) -> Iterator[Ping]:
        types = SONAR_TYPES if include_hidden else (0,)
        for _, ping in self.records(types):
            if isinstance(ping, Ping):
                yield ping

    def attitude(self) -> List[Dict[str, Any]]:
        return [d for _, d in self.records((3,))]

    def notes(self) -> List[Dict[str, Any]]:
        return [d for _, d in self.records((1,))]

    # -- channel layout / side resolution --------------------------------
    @property
    def sonar_channel_infos(self) -> List[Dict[str, Any]]:
        """CHANINFOs of the sonar channels, in order (port/starboard typed ones
        when the header declares types, else the first NumberOfSonarChannels)."""
        infos = self.header.channels
        typed = [c for c in infos if c.get("TypeOfChannel") in (1, 2)]
        if typed:
            return typed
        n = self.header.num_sonar_channels or len(infos)
        return infos[:n]

    @staticmethod
    def _walk(buf: bytes, nchans: int, bps_for) -> Optional[tuple]:
        """Locate channel headers/data assuming ``bps_for(k, chan_header)``
        bytes per sample.  Returns (channels, leftover, truncated) or None."""
        off = PING_HEADER.size
        out = []
        truncated = False
        for k in range(nchans):
            if off + PING_CHAN_HEADER.size > len(buf):
                return None
            ch = PING_CHAN_HEADER.unpack(buf, off)
            off += PING_CHAN_HEADER.size
            bps = bps_for(k, ch)
            if not bps:
                return None
            n = ch["NumSamples"]
            if off + n * bps > len(buf):
                if k != nchans - 1:
                    return None
                n = (len(buf) - off) // bps  # truncated final record
                truncated = True
            out.append((ch, off, n, bps))
            off += n * bps
        return out, len(buf) - off, truncated

    def _layout(self, buf: bytes, nchans: int) -> List[tuple]:
        infos = self.header.channels
        sonar = self.sonar_channel_infos

        def by_number(k, ch):
            num = ch["ChannelNumber"]
            return infos[num].get("BytesPerSample") if num < len(infos) else 0

        def by_position(k, ch):
            return sonar[k].get("BytesPerSample") if k < len(sonar) else 0

        candidates = [by_number, by_position] + [(lambda b: lambda k, ch: b)(b) for b in (2, 1, 4)]
        results = []
        for f in candidates:
            r = self._walk(buf, nchans, f)
            # 1st choice: a layout that fills the record up to its <64-byte padding
            if r is not None and not r[2] and r[1] < 64:
                return r[0]
            results.append(r)
        # then: the tightest complete layout, then a truncated one
        complete = [r for r in results if r is not None and not r[2]]
        if complete:
            return min(complete, key=lambda r: r[1])[0]
        for r in results:
            if r is not None:
                return r[0]
        return []

    def _assign_sides(self, chans: List[Dict[str, Any]]) -> tuple:
        """Return (sides, channel_ids, infos) for the channels of one ping."""
        infos = self.header.channels
        sonar = self.sonar_channel_infos
        n = len(chans)
        type_side = {1: PORT, 2: STARBOARD}

        def sides_from(info_list):
            if any(i is None for i in info_list):
                return None
            return [type_side.get(i.get("TypeOfChannel"), OTHER) for i in info_list]

        def usable(sides):
            # With two or more channels, a mapping is only trusted when it
            # yields both a port and a starboard channel; files whose header
            # does not declare both sides fall through to the convention.
            if sides is None:
                return False
            if n < 2:
                return True
            return PORT in sides and STARBOARD in sides

        nums = [c["ChannelNumber"] for c in chans]
        num_infos = [infos[x] if x < len(infos) else None for x in nums]
        if len(set(nums)) == n:
            sides = sides_from(num_infos)
            if usable(sides):
                return sides, nums, num_infos
        pos_infos = [sonar[k] if k < len(sonar) else None for k in range(n)]
        sides = sides_from(pos_infos)
        if usable(sides):
            return sides, list(range(n)), pos_infos
        # No trustworthy header information: Triton convention, even = port.
        sides = [PORT if k % 2 == 0 else STARBOARD for k in range(n)]
        return sides, list(range(n)), [p if p is not None else {} for p in pos_infos]

    def _parse_ping(self, buf: bytes) -> Ping:
        mv = memoryview(buf)
        h = PING_HEADER.unpack(mv)
        time = _safe_datetime(
            h["Year"], h["Month"], h["Day"], h["Hour"], h["Minute"], h["Second"], h["HSeconds"] * 10_000
        )
        sv = float(h["SoundVelocity"])
        if not math.isfinite(sv) or sv <= 0:
            sv = 1500.0
        elif sv < 1000:  # XTF traditionally stores half the speed of sound
            sv *= 2.0

        ping = Ping(
            time=time,
            ping_number=h["PingNumber"],
            heading=h["SensorHeading"],
            pitch=h["SensorPitch"],
            roll=h["SensorRoll"],
            heave=h["Heave"],
            altitude=h["SensorPrimaryAltitude"],
            depth=h["SensorDepth"],
            speed=h["SensorSpeed"] * KNOTS_TO_MS,
            layback=h["Layback"],
            subsystem=h["SubChannelNumber"],
            metadata=h,
        )
        x, y = h["SensorXcoordinate"], h["SensorYcoordinate"]
        if x == 0 and y == 0:
            x, y = h["ShipXcoordinate"], h["ShipYcoordinate"]
        if self.header.geographic:
            ping.longitude, ping.latitude = x, y
        else:
            ping.easting, ping.northing = x, y

        layout = self._layout(buf, h["NumChansToFollow"])
        if not layout:
            return ping
        sides, ids, infos = self._assign_sides([c for c, _, _, _ in layout])
        for k, (ch, off, n, bps) in enumerate(layout):
            info = dict(infos[k] or {})
            if info.get("BytesPerSample") != bps:
                # Header disagrees with the data: decode as unsigned integers
                # of the width that fits (keeping a declared 4-byte format).
                keep = bps == 4 and info.get("SampleFormat") in (1, 2, 5)
                info.update(BytesPerSample=bps, SampleFormat=info["SampleFormat"] if keep else 0)
            samples = _decode_samples(mv, off, n, info)
            start_range = max(ch["TimeDelay"], 0.0) * sv / 2.0
            if ch["SlantRange"] > 0 and n > 0:
                dt = 2.0 * max(ch["SlantRange"] - start_range, 0.0) / (sv * n)
            elif ch["TimeDuration"] > 0 and n > 0:
                dt = ch["TimeDuration"] / n
            else:
                dt = math.nan
            freq_khz = ch["Frequency"] or (info.get("Frequency") or math.nan)
            ping.channels.append(
                ChannelData(
                    samples=samples,
                    side=sides[k],
                    channel=ids[k],
                    frequency=float(freq_khz) * 1000.0,
                    sample_interval=dt,
                    sound_velocity=sv,
                    start_range=start_range,
                    metadata={**ch, "packet_index": k, "bytes_per_sample": bps},
                )
            )
        # A channel without range fields borrows the timing of a sibling
        # (port and starboard share sample rate on every common system).
        good = [c for c in ping.channels if math.isfinite(c.sample_interval) and c.sample_interval > 0]
        for c in ping.channels:
            if not (math.isfinite(c.sample_interval) and c.sample_interval > 0) and good:
                ref = min(good, key=lambda g: (abs(g.num_samples - c.num_samples), abs(g.channel - c.channel)))
                c.sample_interval = ref.sample_interval
                c.start_range = ref.start_range
                c.metadata["sample_interval_from_channel"] = ref.channel
        return ping


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_DTYPE_TO_FORMAT = {
    "uint8": (1, 0),
    "uint16": (2, 0),
    "uint32": (4, 0),
    "int16": (2, 3),
    "int32": (4, 2),
    "float32": (4, 5),
}


class XTFWriter:
    """Write pings to a new XTF file.

    Channel definitions are taken from the first ping written unless given
    explicitly.  ``dtype`` controls how samples are stored; ``"float32"``
    preserves full dynamic range (recommended for processed / SAS data),
    ``"uint16"`` is the most widely supported by third-party viewers.
    """

    def __init__(
        self,
        path: str | Path,
        dtype: str = "uint16",
        geographic: bool = True,
        sonar_name: str = "xtfjsf",
        note: str = "",
        scale: float = 1.0,
    ):
        if dtype not in _DTYPE_TO_FORMAT:
            raise ValueError(f"dtype must be one of {sorted(_DTYPE_TO_FORMAT)}")
        self.path = Path(path)
        self.dtype = np.dtype(dtype).newbyteorder("<")
        self.bytes_per_sample, self.sample_format = _DTYPE_TO_FORMAT[dtype]
        self.geographic = geographic
        self.sonar_name = sonar_name
        self.note = note
        self.scale = scale
        self._fh = open(self.path, "wb")
        self._channel_map: Optional[Dict[tuple, int]] = None
        self.pings_written = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._channel_map is None:
            self._write_header([])
        self._fh.close()

    @staticmethod
    def _chan_key(ch: ChannelData) -> tuple:
        freq = None if math.isnan(ch.frequency) else round(ch.frequency)
        return (ch.side, freq, ch.channel)

    def _write_header(self, channels: List[ChannelData]) -> None:
        hdr = FILE_HEADER.pack(
            FileFormat=FILE_FORMAT_ID,
            SystemType=1,
            RecordingProgramName="xtfjsf",
            RecordingProgramVersion="223",
            SonarName=self.sonar_name,
            SonarType=0,
            NoteString=self.note,
            ThisFileName=self.path.name[:63],
            NavUnits=3 if self.geographic else 0,
            NumberOfSonarChannels=len(channels),
        )
        infos = b""
        for ch in channels:
            infos += CHAN_INFO.pack(
                TypeOfChannel={PORT: 1, STARBOARD: 2}.get(ch.side, 0),
                SubChannelNumber=0,
                CorrectionFlags=1,
                UniPolar=0 if self.sample_format in (2, 3) else 1,
                BytesPerSample=self.bytes_per_sample,
                ChannelName=f"{ch.side[:4]} {0 if math.isnan(ch.frequency) else ch.frequency / 1000:.0f}kHz",
                VoltScale=5.0,
                Frequency=0.0 if math.isnan(ch.frequency) else ch.frequency / 1000.0,
                SampleFormat=self.sample_format,
            )
        n_blocks = 1 if len(channels) <= 6 else 1 + math.ceil((len(channels) - 6) / 8)
        header = bytearray(1024 * n_blocks)
        header[:256] = hdr
        first = infos[: 6 * 128]
        header[256 : 256 + len(first)] = first
        rest = infos[6 * 128 :]
        header[1024 : 1024 + len(rest)] = rest
        self._fh.write(bytes(header))

    def write_ping(self, ping: Ping) -> None:
        if self._channel_map is None:
            self._channel_map = {self._chan_key(ch): i for i, ch in enumerate(ping.channels)}
            self._write_header(ping.channels)
        t = ping.time or datetime(1970, 1, 1, tzinfo=timezone.utc)
        sv = next((c.sound_velocity for c in ping.channels), 1500.0)

        def f(v, default=0.0):
            return default if v is None or not math.isfinite(v) else float(v)

        if self.geographic:
            x, y = f(ping.longitude), f(ping.latitude)
        else:
            x, y = f(ping.easting), f(ping.northing)

        body = bytearray()
        n_chans = 0
        for ch in ping.channels:
            num = self._channel_map.get(self._chan_key(ch))
            if num is None:
                continue  # channel not declared in header
            data = ch.intensity() * self.scale
            if self.dtype.kind in "ui":
                info = np.iinfo(self.dtype)
                data = np.clip(np.rint(data), info.min, info.max)
            data = data.astype(self.dtype)
            n = data.shape[0]
            dt = ch.sample_interval if math.isfinite(ch.sample_interval) else 0.0
            body += PING_CHAN_HEADER.pack(
                ChannelNumber=num,
                SlantRange=f(ch.slant_range),
                TimeDelay=2.0 * ch.start_range / ch.sound_velocity if ch.sound_velocity else 0.0,
                TimeDuration=dt * n,
                Frequency=int(min(f(ch.frequency) / 1000.0, 65535)),
                NumSamples=n,
            )
            body += data.tobytes()
            n_chans += 1

        total = PING_HEADER.size + len(body)
        total += (-total) % 64  # records are padded to a multiple of 64 bytes
        hundredths = t.microsecond // 10_000
        header = PING_HEADER.pack(
            MagicNumber=MAGIC,
            HeaderType=0,
            SubChannelNumber=ping.subsystem & 0xFF,
            NumChansToFollow=n_chans,
            NumBytesThisRecord=total,
            Year=t.year,
            Month=t.month,
            Day=t.day,
            Hour=t.hour,
            Minute=t.minute,
            Second=t.second,
            HSeconds=hundredths,
            JulianDay=t.timetuple().tm_yday,
            PingNumber=ping.ping_number & 0xFFFFFFFF,
            SoundVelocity=sv / 2.0,
            SensorXcoordinate=x,
            SensorYcoordinate=y,
            ShipXcoordinate=x,
            ShipYcoordinate=y,
            SensorSpeed=f(ping.speed) / KNOTS_TO_MS,
            SensorHeading=f(ping.heading),
            SensorPitch=f(ping.pitch),
            SensorRoll=f(ping.roll),
            Heave=f(ping.heave),
            SensorPrimaryAltitude=f(ping.altitude),
            SensorDepth=f(ping.depth),
            Layback=f(ping.layback),
            NavFixMilliseconds=int(t.timestamp() * 1000) & 0xFFFFFFFF,
        )
        record = header + bytes(body)
        record += b"\x00" * (total - len(record))
        self._fh.write(record)
        self.pings_written += 1

    def write_note(self, text: str, time: Optional[datetime] = None) -> None:
        if self._channel_map is None:
            raise RuntimeError("write at least one ping before writing notes")
        t = time or datetime.now(timezone.utc)
        self._fh.write(
            NOTES.pack(
                MagicNumber=MAGIC,
                HeaderType=1,
                NumBytesThisRecord=NOTES.size,
                Year=t.year,
                Month=t.month,
                Day=t.day,
                Hour=t.hour,
                Minute=t.minute,
                Second=t.second,
                NotesText=text[:199],
            )
        )
