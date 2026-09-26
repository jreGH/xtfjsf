import json
import math
import struct

import numpy as np
import pytest

import xtfjsf as xj
from xtfjsf.formats import jsf as jsfmod
from xtfjsf.formats import xtf as xtfmod


def _write(writer_cls, path, pings, **kw):
    with writer_cls(path, **kw) as w:
        for p in pings:
            w.write_ping(p)
    return path


def test_record_layout_sizes():
    assert xtfmod.FILE_HEADER.size == 256
    assert xtfmod.CHAN_INFO.size == 128
    assert xtfmod.PING_HEADER.size == 256
    assert xtfmod.PING_CHAN_HEADER.size == 64
    assert jsfmod.MESSAGE_HEADER.size == 16
    assert jsfmod.SONAR_HEADER.size == 240


@pytest.mark.parametrize("dtype", ["float32", "uint16"])
def test_xtf_roundtrip(tmp_path, pings, dtype):
    path = _write(xj.XTFWriter, tmp_path / "line.xtf", pings, dtype=dtype)
    assert xj.detect_format(path) == "xtf"
    with xj.open_sonar(path) as f:
        assert isinstance(f, xj.XTFFile)
        assert f.header.num_sonar_channels == 2
        assert [c["TypeOfChannel"] for c in f.header.channels] == [1, 2]
        out = f.read_all()
    assert len(out) == len(pings)
    for a, b in zip(pings, out):
        assert b.ping_number == a.ping_number
        assert abs((b.time - a.time).total_seconds()) < 0.011
        assert b.latitude == pytest.approx(a.latitude, abs=1e-9)
        assert b.longitude == pytest.approx(a.longitude, abs=1e-9)
        assert b.heading == pytest.approx(a.heading)
        assert b.altitude == pytest.approx(a.altitude)
        assert b.speed == pytest.approx(a.speed, rel=1e-4)
        for side in (xj.PORT, xj.STARBOARD):
            ca, cb = a.channel(side), b.channel(side)
            assert cb.num_samples == ca.num_samples
            assert cb.frequency == pytest.approx(ca.frequency)
            assert cb.range_resolution == pytest.approx(ca.range_resolution, rel=1e-5)
            tol = 1e-3 if dtype == "float32" else 0.5
            np.testing.assert_allclose(cb.samples, ca.samples, atol=tol)


def test_xtf_notes_and_record_counts(tmp_path, pings):
    path = tmp_path / "n.xtf"
    with xj.XTFWriter(path) as w:
        w.write_ping(pings[0])
        w.write_note("hello survey", pings[0].time)
        w.write_ping(pings[1])
    with xj.XTFFile(path) as f:
        assert f.record_counts() == {"sonar": 2, "notes": 1}
        assert f.notes()[0]["NotesText"] == "hello survey"
        assert len(f.read_all()) == 2


def test_xtf_attitude_packet(tmp_path, pings):
    path = _write(xj.XTFWriter, tmp_path / "a.xtf", pings[:2])
    att = xtfmod.ATTITUDE.pack(
        MagicNumber=0xFACE, HeaderType=3, NumBytesThisRecord=64, Pitch=1.5, Roll=-2.0, Heave=0.1, Heading=123.0,
        Year=2024, Month=5, Day=1,
    )
    with open(path, "ab") as fh:
        fh.write(att)
    with xj.XTFFile(path) as f:
        a = f.attitude()
    assert len(a) == 1
    assert a[0]["Roll"] == pytest.approx(-2.0)
    assert a[0]["Heading"] == pytest.approx(123.0)


def test_xtf_resync_after_corruption(tmp_path, pings):
    path = _write(xj.XTFWriter, tmp_path / "c.xtf", pings[:5])
    data = bytearray(path.read_bytes())
    with xj.XTFFile(path) as f:
        second = f.index[1].offset
    # Smash the magic number of the second packet: reader must skip it and carry on.
    data[second : second + 2] = b"\x00\x00"
    path.write_bytes(bytes(data))
    with xj.XTFFile(path) as f:
        got = [p.ping_number for p in f.pings()]
    assert got == [1, 3, 4, 5]


def test_xtf_many_channels_header(tmp_path, pings):
    """More than 6 channels spill into a second 1024-byte header block."""
    base = pings[0]
    chans = []
    for i in range(8):
        c = base.channels[i % 2]
        chans.append(xj.ChannelData(samples=c.samples, side=c.side, channel=i, frequency=100e3 * (1 + i // 2),
                                    sample_interval=c.sample_interval))
    p = xj.Ping(time=base.time, ping_number=1, latitude=50, longitude=-4, channels=chans)
    path = _write(xj.XTFWriter, tmp_path / "m.xtf", [p])
    with xj.XTFFile(path) as f:
        assert f.header.size == 2048
        assert len(f.header.channels) == 8
        q = f.read_all()[0]
    assert len(q.channels) == 8
    assert sorted({round(c.frequency) for c in q.channels}) == [100000, 200000, 300000, 400000]


def test_xtf_legacy_half_sound_velocity(tmp_path, pings):
    path = _write(xj.XTFWriter, tmp_path / "v.xtf", pings[:1])
    with xj.XTFFile(path) as f:
        raw = f.read_raw(f.index[0])
        assert struct.unpack_from("<f", raw, 32)[0] == pytest.approx(750.0)
        assert f.read_all()[0].port.sound_velocity == pytest.approx(1500.0)


def test_jsf_roundtrip(tmp_path, pings):
    path = _write(xj.JSFWriter, tmp_path / "line.jsf", pings, weighting_factor=4)
    assert xj.detect_format(path) == "jsf"
    with xj.open_sonar(path) as f:
        assert isinstance(f, xj.JSFFile)
        assert f.subsystems == [20]
        assert f.record_counts() == {"sonar_data/subsystem20": 2 * len(pings)}
        out = f.read_all()
    assert len(out) == len(pings)
    for a, b in zip(pings, out):
        assert b.ping_number == a.ping_number
        assert abs((b.time - a.time).total_seconds()) < 0.0015
        # JSF stores 1/10000 minute of arc: ~0.2 m
        assert b.latitude == pytest.approx(a.latitude, abs=2e-6)
        assert b.longitude == pytest.approx(a.longitude, abs=2e-6)
        assert b.heading == pytest.approx(a.heading, abs=0.01)
        assert b.altitude == pytest.approx(a.altitude, abs=1e-3)
        for side in (xj.PORT, xj.STARBOARD):
            ca, cb = a.channel(side), b.channel(side)
            np.testing.assert_allclose(cb.samples, ca.samples, atol=2.0**-4)
            assert cb.range_resolution == pytest.approx(ca.range_resolution, rel=1e-4)
            assert cb.frequency == pytest.approx(ca.frequency)


def test_jsf_complex_sas_data(tmp_path, sas_pings):
    path = _write(xj.JSFWriter, tmp_path / "sas.jsf", sas_pings, weighting_factor=3, subsystem=21)
    with xj.JSFFile(path) as f:
        out = f.read_all()
    assert out[0].port.is_complex
    assert out[0].port.metadata["data_format"] == 1
    np.testing.assert_allclose(out[5].starboard.samples, sas_pings[5].starboard.samples, atol=2.0**-3)
    wf = xj.build_waterfall(out, keep_complex=True)
    assert wf.is_complex


def test_jsf_subsystem_selection_and_nmea(tmp_path, pings):
    path = tmp_path / "multi.jsf"
    with xj.JSFWriter(path) as w:
        for p in pings[:10]:
            w.write_ping(p, subsystem=20)
            w.write_ping(p, subsystem=21)
        w.write_nmea("$GPGGA,120000,5000.000,N,00400.000,W,1,08,0.9,0,M,0,M,,*47", pings[0].time)
    with xj.JSFFile(path) as f:
        assert f.subsystems == [20, 21]
        assert len(list(f.pings())) == 20
        assert len(list(f.pings(subsystem=21))) == 10
        assert all(p.subsystem == 21 for p in f.pings(subsystem=21))
        assert all(p.port is not None and p.starboard is not None for p in f.pings(subsystem=20))
        nmea = f.nmea()
    assert nmea[0]["sentence"].startswith("$GPGGA")


def test_jsf_resync_after_garbage(tmp_path, pings):
    path = _write(xj.JSFWriter, tmp_path / "g.jsf", pings[:4])
    data = path.read_bytes()
    with xj.JSFFile(path) as f:
        cut = f.index[2].offset
    path.write_bytes(data[:cut] + b"\x00garbage\xff" * 7 + data[cut:])
    with xj.JSFFile(path) as f:
        assert [p.ping_number for p in f.pings()] == [1, 2, 3, 4]


def test_detection_rejects_json_and_unknown(tmp_path):
    j = tmp_path / "x.json"
    j.write_text(json.dumps({"a": list(range(300))}))
    with pytest.raises(ValueError):
        xj.detect_format(j)
    r = tmp_path / "x.bin"
    r.write_bytes(b"\x00" * 2048)
    with pytest.raises(ValueError):
        xj.open_sonar(r)


def test_register_custom_format(tmp_path, pings):
    class Dummy(xj.SonarFile):
        format_name = "dummy"

        def pings(self, **kw):
            yield from pings[:3]

    xj.register_format("dummy", Dummy, lambda h: h.startswith(b"DUMMY"), (".dmy",))
    try:
        path = tmp_path / "d.dmy"
        path.write_bytes(b"DUMMY")
        with xj.open_sonar(path) as f:
            assert f.format_name == "dummy"
            assert len(f.read_all()) == 3
    finally:
        from xtfjsf.formats import _REGISTRY

        _REGISTRY.pop("dummy")


def test_convert_jsf_to_xtf(tmp_path, pings):
    src = _write(xj.JSFWriter, tmp_path / "s.jsf", pings[:20])
    n = xj.convert(src, tmp_path / "d.xtf", dtype="float32")
    assert n == 20
    a = xj.read_waterfall(src)
    b = xj.read_waterfall(tmp_path / "d.xtf")
    np.testing.assert_allclose(a.port, b.port, atol=1e-3)
    np.testing.assert_allclose(a.latitude, b.latitude, atol=1e-9)
    assert not math.isnan(b.frequency)


def test_xtf_tolerates_unpadded_records(tmp_path):
    """Records whose NumBytesThisRecord includes 64-byte padding that was not written."""
    from xtfjsf.simulate import simulate_line

    pings = simulate_line(num_pings=6, num_samples=1001, targets=())
    path = _write(xj.XTFWriter, tmp_path / "u.xtf", pings)
    with xj.XTFFile(path) as f:
        chunks = [f.read_raw(r) for r in f.index]
        header = path.read_bytes()[: f.header.size]
    content = xtfmod.PING_HEADER.size + 2 * (xtfmod.PING_CHAN_HEADER.size + 2 * 1001)
    assert all(len(c) > content for c in chunks)  # there is padding to drop
    path.write_bytes(header + b"".join(c[:content] for c in chunks))
    with xj.XTFFile(path) as f:
        out = f.read_all()
    assert [p.ping_number for p in out] == [1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(out[4].starboard.samples, np.rint(pings[4].starboard.samples))


@pytest.mark.parametrize("writer", [xj.XTFWriter, xj.JSFWriter])
def test_truncated_final_record(tmp_path, pings, writer):
    path = _write(writer, tmp_path / ("t" + (".xtf" if writer is xj.XTFWriter else ".jsf")), pings[:5])
    data = path.read_bytes()
    path.write_bytes(data[:-500])  # logging stopped mid-record
    with xj.open_sonar(path) as f:
        out = f.read_all()
    assert [p.ping_number for p in out] == [1, 2, 3, 4, 5]
    last = out[-1].starboard
    assert 0 < last.num_samples < 800
