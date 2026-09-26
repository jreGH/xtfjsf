"""EdgeTech JSF reader and writer.

Decodes message type 80 (sonar data: side-scan, sub-bottom, SAS / analytic
data) including the 240-byte trace header, and message 2002 (NMEA strings).
Every other message type is indexed and available as raw bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .._binary import BinaryRecord
from ..core import OTHER, PORT, STARBOARD, ChannelData, Ping, SonarFile

START_MARKER = 0x1601
MARKER_BYTES = b"\x01\x16"
KNOTS_TO_MS = 0.514444

MESSAGE_TYPES = {
    80: "sonar_data",
    82: "side_scan_data",
    86: "4400_sas_processed",
    181: "navigation_offsets",
    182: "system_information",
    426: "file_timestamp",
    428: "file_padding",
    2000: "sensor_data",
    2002: "nmea_string",
    2020: "pitch_roll",
    2060: "pressure_sensor",
    2080: "doppler_velocity_log",
    2090: "situation",
    2091: "situation_comprehensive",
    2100: "cable_counter",
    2101: "kilometer_of_pipe",
    2111: "container_timestamp",
    3000: "bathymetric_data",
    3001: "attitude_data",
    3002: "pressure_data",
    3003: "altitude_data",
    3004: "position_data",
    3005: "status_data",
    9001: "discover2_general_prefix",
    9002: "discover2_situation",
}

SUBSYSTEMS = {
    0: "sub-bottom",
    20: "side-scan low frequency",
    21: "side-scan high frequency",
    22: "side-scan very high frequency",
    40: "bathymetric low frequency",
    41: "bathymetric high frequency",
    42: "bathymetric very high frequency",
    70: "bathymetric motion tolerant low frequency",
    71: "bathymetric motion tolerant high frequency",
    100: "raw serial/UDP",
    101: "parsed serial/UDP",
}

MESSAGE_HEADER = BinaryRecord(
    "JSFMessageHeader",
    [
        ("start_marker", "H"),
        ("version", "B"),
        ("session_id", "B"),
        ("message_type", "H"),
        ("command_type", "B"),
        ("subsystem", "B"),
        ("channel", "B"),
        ("sequence", "B"),
        ("_reserved", "2s"),
        ("message_size", "i"),
    ],
    size=16,
)

SONAR_HEADER = BinaryRecord(
    "JSFSonarDataHeader",
    [
        ("ping_time", "i"),  # 0: seconds since 1970-01-01
        ("starting_depth", "I"),  # 4: window offset in samples
        ("ping_number", "I"),  # 8
        ("_r0", "4s"),  # 12
        ("msbs", "H"),  # 16: high bits for samples / frequencies
        ("lsb1", "H"),  # 18
        ("lsb2", "H"),  # 20
        ("_r1", "2s"),  # 22
        ("id_code", "h"),  # 24
        ("validity_flag", "H"),  # 26
        ("_r2", "2s"),  # 28
        ("data_format", "h"),  # 30
        ("nmea_antenna_aft", "h"),  # 32 (cm)
        ("nmea_antenna_starboard", "h"),  # 34 (cm)
        ("_r3", "44s"),  # 36..79
        ("km_of_pipe", "f"),  # 80
        ("_r4", "16s"),  # 84..99
        ("x", "i"),  # 100: longitude / easting
        ("y", "i"),  # 104: latitude / northing
        ("coordinate_units", "h"),  # 108
        ("annotation", "24s"),  # 110
        ("samples", "H"),  # 134
        ("sample_interval_ns", "I"),  # 136
        ("adc_gain", "H"),  # 140
        ("pulse_power", "h"),  # 142
        ("_r5", "2s"),  # 144
        ("start_freq_dahz", "H"),  # 146
        ("end_freq_dahz", "H"),  # 148
        ("sweep_length_ms", "H"),  # 150
        ("pressure_mpsi", "i"),  # 152
        ("depth_mm", "i"),  # 156
        ("sample_freq_hz", "H"),  # 160
        ("pulse_id", "H"),  # 162
        ("altitude_mm", "i"),  # 164
        ("sound_speed", "f"),  # 168
        ("mixer_freq", "f"),  # 172
        ("year", "h"),  # 176
        ("day", "h"),  # 178
        ("hour", "h"),  # 180
        ("minute", "h"),  # 182
        ("second", "h"),  # 184
        ("time_basis", "h"),  # 186
        ("weighting_factor", "h"),  # 188
        ("num_pulses", "h"),  # 190
        ("heading", "H"),  # 192: 1/100 degree
        ("pitch", "h"),  # 194: * 180/32768 degrees
        ("roll", "h"),  # 196
        ("_r6", "4s"),  # 198
        ("trigger_source", "h"),  # 202
        ("mark_number", "H"),  # 204
        ("fix_hour", "h"),  # 206
        ("fix_minute", "h"),  # 208
        ("fix_second", "h"),  # 210
        ("course", "h"),  # 212: degrees
        ("speed", "h"),  # 214: 1/10 knot
        ("fix_day", "h"),  # 216
        ("fix_year", "h"),  # 218
        ("ms_today", "I"),  # 220
        ("max_adc", "H"),  # 224
        ("_r7", "4s"),  # 226
        ("software_version", "6s"),  # 230
        ("spherical_correction", "i"),  # 236
    ],
    size=240,
)

NMEA_HEADER = BinaryRecord(
    "JSFNMEA",
    [("time", "i"), ("ms_in_second", "I"), ("source", "B"), ("_r", "3s")],
    size=12,
)

ANGLE_SCALE = 180.0 / 32768.0

# data_format -> (dtype, samples are complex pairs)
_DATA_FORMATS = {
    0: (np.dtype("<u2"), False),  # envelope
    1: (np.dtype("<i2"), True),  # analytic signal, real/imag pairs
    2: (np.dtype("<i2"), False),  # raw
    3: (np.dtype("<i2"), False),  # real part of analytic signal
    4: (np.dtype("<u2"), False),  # pixel data
    9: (np.dtype("<u2"), False),  # match-filtered envelope (4200/4300 SAS)
}


def _coords(h: Dict[str, Any]) -> Tuple[str, float, float]:
    """Return (kind, x, y) where kind is "geographic" or "projected"."""
    units, x, y = h["coordinate_units"], h["x"], h["y"]
    if units == 2:  # minutes of arc * 10000
        return "geographic", x / 600000.0, y / 600000.0
    if units == 4:  # degrees * 1e7 (newer firmware)
        return "geographic", x / 1e7, y / 1e7
    if units == 1:  # millimetres
        return "projected", x / 1000.0, y / 1000.0
    if units == 3:  # decimetres
        return "projected", x / 10.0, y / 10.0
    return "none", math.nan, math.nan


@dataclass
class JSFRecordIndex:
    offset: int
    message_type: int
    subsystem: int
    channel: int
    size: int  # total size including the 16-byte header

    @property
    def type_name(self) -> str:
        return MESSAGE_TYPES.get(self.message_type, f"type_{self.message_type}")


class JSFFile(SonarFile):
    """Reader for EdgeTech JSF files.

    Port (channel 0) and starboard (channel 1) messages with the same ping
    number and subsystem are merged into one :class:`Ping`.  Use
    ``pings(subsystem=21)`` to select, e.g., the high-frequency side-scan.
    """

    format_name = "jsf"

    def __init__(self, path: str | Path):
        super().__init__(path)
        self._fh.seek(0)
        if self._fh.read(2) != MARKER_BYTES:
            raise ValueError(f"{self.path}: not a JSF file (missing 0x1601 start marker)")
        self._index: Optional[List[JSFRecordIndex]] = None

    @property
    def index(self) -> List[JSFRecordIndex]:
        if self._index is None:
            self._index = list(self._scan())
        return self._index

    def _find_marker(self, start: int) -> Optional[int]:
        fh = self._fh
        chunk_size = 1 << 20
        pos = start
        while True:
            fh.seek(pos)
            chunk = fh.read(chunk_size + 1)
            if len(chunk) < 2:
                return None
            i = chunk.find(MARKER_BYTES)
            if i >= 0:
                return pos + i
            pos += chunk_size

    def _scan(self) -> Iterator[JSFRecordIndex]:
        fh = self._fh
        fh.seek(0, 2)
        end = fh.tell()
        pos = 0
        while pos + MESSAGE_HEADER.size <= end:
            fh.seek(pos)
            h = MESSAGE_HEADER.unpack(fh.read(MESSAGE_HEADER.size))
            size = h["message_size"]
            if h["start_marker"] == START_MARKER and pos + 16 + size > end and self._find_marker(pos + 2) is None:
                size = end - pos - 16  # final message truncated: keep what is there
            if h["start_marker"] != START_MARKER or size < 0 or pos + 16 + size > end:
                nxt = self._find_marker(pos + 1)
                if nxt is None:
                    return
                pos = nxt
                continue
            yield JSFRecordIndex(pos, h["message_type"], h["subsystem"], h["channel"], 16 + size)
            pos += 16 + size

    def record_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.index:
            key = r.type_name
            if r.message_type == 80:
                key += f"/subsystem{r.subsystem}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def subsystems(self) -> List[int]:
        return sorted({r.subsystem for r in self.index if r.message_type == 80})

    def read_raw(self, rec: JSFRecordIndex) -> bytes:
        self._fh.seek(rec.offset)
        return self._fh.read(rec.size)

    def records(self, types: Optional[Sequence[int]] = None) -> Iterator[tuple]:
        """Yield ``(JSFRecordIndex, parsed)``; sonar messages are parsed to
        :class:`ChannelData`-bearing single-channel :class:`Ping` objects."""
        for rec in self.index:
            if types is not None and rec.message_type not in types:
                continue
            buf = self.read_raw(rec)
            if rec.message_type == 80 and rec.size >= 16 + SONAR_HEADER.size:
                yield rec, self._parse_sonar(buf)
            elif rec.message_type == 2002 and rec.size >= 16 + NMEA_HEADER.size:
                d = NMEA_HEADER.unpack(buf, 16)
                d["sentence"] = buf[16 + NMEA_HEADER.size :].split(b"\x00", 1)[0].decode("ascii", "replace").strip()
                yield rec, d
            else:
                yield rec, buf

    def nmea(self) -> List[Dict[str, Any]]:
        return [d for _, d in self.records((2002,))]

    def pings(self, subsystem: Optional[int] = None) -> Iterator[Ping]:
        """Yield merged pings.  Defaults to side-scan subsystems (20-22) when
        present, otherwise every subsystem."""
        if subsystem is None:
            ss = [s for s in self.subsystems if 20 <= s <= 29]
            wanted = set(ss) if ss else None
        else:
            wanted = {subsystem}
        pending: Dict[int, Ping] = {}
        for rec in self.index:
            if rec.message_type != 80 or (wanted is not None and rec.subsystem not in wanted):
                continue
            if rec.size < 16 + SONAR_HEADER.size:
                continue
            p = self._parse_sonar(self.read_raw(rec))
            cur = pending.get(rec.subsystem)
            if cur is not None and cur.ping_number == p.ping_number and not any(
                c.channel == p.channels[0].channel for c in cur.channels
            ):
                cur.channels.extend(p.channels)
                continue
            if cur is not None:
                yield cur
            pending[rec.subsystem] = p
        for p in pending.values():
            yield p

    # -- decoding ----------------------------------------------------------
    def _parse_sonar(self, buf: bytes) -> Ping:
        mh = MESSAGE_HEADER.unpack(buf)
        h = SONAR_HEADER.unpack(buf, 16)
        msbs = h["msbs"]
        n = h["samples"] + ((msbs & 0x000F) << 16)
        start_f = (h["start_freq_dahz"] + (((msbs >> 4) & 0xF) << 16)) * 10.0
        end_f = (h["end_freq_dahz"] + (((msbs >> 8) & 0xF) << 16)) * 10.0

        dtype, is_complex = _DATA_FORMATS.get(h["data_format"], (np.dtype("<u2"), False))
        payload = memoryview(buf)[16 + SONAR_HEADER.size :]
        values_per_sample = 2 if is_complex else 1
        avail = len(payload) // (dtype.itemsize * values_per_sample)
        n = min(n, avail)
        raw = np.frombuffer(payload, dtype=dtype, count=n * values_per_sample)
        scale = 2.0 ** (-h["weighting_factor"])
        if is_complex:
            raw = raw.astype(np.float32).reshape(-1, 2)
            samples = ((raw[:, 0] + 1j * raw[:, 1]) * scale).astype(np.complex64)
        else:
            samples = (raw.astype(np.float32) * scale).astype(np.float32)

        sv = float(h["sound_speed"])
        if not math.isfinite(sv) or sv < 1000 or sv > 2000:
            sv = 1500.0
        dt = h["sample_interval_ns"] * 1e-9 if h["sample_interval_ns"] else math.nan
        start_range = h["starting_depth"] * sv * dt / 2.0 if math.isfinite(dt) else 0.0

        chan = mh["channel"]
        if 20 <= mh["subsystem"] <= 29:
            side = PORT if chan == 0 else STARBOARD if chan == 1 else OTHER
        else:
            side = OTHER

        # Time: whole seconds from ping_time plus milliseconds-of-day if present.
        base = h["ping_time"]
        t = float(base)
        if h["ms_today"]:
            t = (base // 86400) * 86400 + h["ms_today"] / 1000.0
            if t - base > 43200:
                t -= 86400
            elif base - t > 43200:
                t += 86400
        try:
            time = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=t)
        except OverflowError:
            time = None

        kind, x, y = _coords(h)
        ping = Ping(
            time=time,
            ping_number=h["ping_number"],
            heading=h["heading"] / 100.0,
            pitch=h["pitch"] * ANGLE_SCALE,
            roll=h["roll"] * ANGLE_SCALE,
            altitude=h["altitude_mm"] / 1000.0,
            depth=h["depth_mm"] / 1000.0,
            speed=h["speed"] / 10.0 * KNOTS_TO_MS,
            subsystem=mh["subsystem"],
            metadata={**h, "message": mh},
        )
        if kind == "geographic":
            ping.longitude, ping.latitude = x, y
        elif kind == "projected":
            ping.easting, ping.northing = x, y

        ping.channels.append(
            ChannelData(
                samples=samples,
                side=side,
                channel=chan,
                frequency=(start_f + end_f) / 2.0 if (start_f or end_f) else math.nan,
                sample_interval=dt,
                sound_velocity=sv,
                start_range=start_range,
                metadata={
                    "data_format": h["data_format"],
                    "start_frequency": start_f,
                    "end_frequency": end_f,
                    "weighting_factor": h["weighting_factor"],
                    "subsystem": mh["subsystem"],
                },
            )
        )
        return ping


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class JSFWriter:
    """Write pings as JSF message-80 records (one message per channel).

    Real data is written as envelope (format 0, uint16); complex data as
    analytic pairs (format 1, int16).  ``weighting_factor`` N scales the
    stored integers by 2**N, so values are recovered as ``int * 2**-N``.
    """

    def __init__(self, path: str | Path, subsystem: int = 20, weighting_factor: int = 0, protocol_version: int = 16):
        self.path = Path(path)
        self.subsystem = subsystem
        self.weighting_factor = weighting_factor
        self.version = protocol_version
        self._fh = open(self.path, "wb")
        self._seq = 0
        self.pings_written = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._fh.close()

    def write_ping(self, ping: Ping, subsystem: Optional[int] = None) -> None:
        subsystem = self.subsystem if subsystem is None else subsystem
        for ch in ping.channels:
            self._write_channel(ping, ch, subsystem)
        self.pings_written += 1

    def write_nmea(self, sentence: str, time: datetime) -> None:
        ts = time.timestamp()
        body = NMEA_HEADER.pack(time=int(ts), ms_in_second=int(round((ts % 1) * 1000)), source=1)
        body += sentence.encode("ascii") + b"\x00"
        self._write_message(2002, 0, 0, body)

    def _write_message(self, mtype: int, subsystem: int, channel: int, body: bytes) -> None:
        self._fh.write(
            MESSAGE_HEADER.pack(
                start_marker=START_MARKER,
                version=self.version,
                message_type=mtype,
                command_type=2,
                subsystem=subsystem,
                channel=channel,
                sequence=self._seq & 0xFF,
                message_size=len(body),
            )
        )
        self._fh.write(body)
        self._seq += 1

    def _write_channel(self, ping: Ping, ch: ChannelData, subsystem: int) -> None:
        scale = 2.0**self.weighting_factor
        if ch.is_complex:
            fmt = 1
            pairs = np.empty((ch.num_samples, 2), dtype=np.float64)
            pairs[:, 0] = ch.samples.real * scale
            pairs[:, 1] = ch.samples.imag * scale
            data = np.clip(np.rint(pairs), -32768, 32767).astype("<i2").tobytes()
        else:
            fmt = 0
            data = np.clip(np.rint(np.asarray(ch.samples, dtype=np.float64) * scale), 0, 65535).astype("<u2").tobytes()
        n = ch.num_samples
        t = ping.time or datetime(1970, 1, 1, tzinfo=timezone.utc)
        ts = t.timestamp()
        day_start = datetime(t.year, t.month, t.day, tzinfo=t.tzinfo or timezone.utc)
        ms_today = int(round((t - day_start).total_seconds() * 1000))
        dt = ch.sample_interval if math.isfinite(ch.sample_interval) else 0.0

        def fi(v, s=1.0):
            return 0 if v is None or not math.isfinite(v) else int(round(v * s))

        start_f = int(round((ch.metadata.get("start_frequency") or (0 if math.isnan(ch.frequency) else ch.frequency)) / 10))
        end_f = int(round((ch.metadata.get("end_frequency") or (0 if math.isnan(ch.frequency) else ch.frequency)) / 10))
        msbs = ((n >> 16) & 0xF) | (((start_f >> 16) & 0xF) << 4) | (((end_f >> 16) & 0xF) << 8)
        if math.isfinite(ping.latitude) and math.isfinite(ping.longitude):
            units, x, y = 2, fi(ping.longitude, 600000), fi(ping.latitude, 600000)
        elif math.isfinite(ping.easting) and math.isfinite(ping.northing):
            units, x, y = 1, fi(ping.easting, 1000), fi(ping.northing, 1000)
        else:
            units, x, y = 0, 0, 0
        start_samples = int(round(ch.start_range / (ch.sound_velocity * dt / 2.0))) if dt > 0 else 0
        header = SONAR_HEADER.pack(
            ping_time=int(ts // 1),
            starting_depth=start_samples,
            ping_number=ping.ping_number & 0xFFFFFFFF,
            msbs=msbs,
            id_code=1,
            validity_flag=0,
            data_format=fmt,
            x=x,
            y=y,
            coordinate_units=units,
            samples=n & 0xFFFF,
            sample_interval_ns=int(round(dt * 1e9)),
            start_freq_dahz=start_f & 0xFFFF,
            end_freq_dahz=end_f & 0xFFFF,
            depth_mm=fi(ping.depth, 1000),
            altitude_mm=fi(ping.altitude, 1000),
            sound_speed=float(ch.sound_velocity),
            year=t.year,
            day=t.timetuple().tm_yday,
            hour=t.hour,
            minute=t.minute,
            second=t.second,
            weighting_factor=self.weighting_factor,
            heading=fi(ping.heading % 360.0 if math.isfinite(ping.heading) else math.nan, 100),
            pitch=fi(ping.pitch, 1 / ANGLE_SCALE),
            roll=fi(ping.roll, 1 / ANGLE_SCALE),
            speed=fi(ping.speed / KNOTS_TO_MS if math.isfinite(ping.speed) else math.nan, 10),
            ms_today=ms_today,
            software_version="xtfjsf",
        )
        channel = ch.channel if ch.side == OTHER else (0 if ch.side == PORT else 1)
        self._write_message(80, subsystem, channel, header + data)
