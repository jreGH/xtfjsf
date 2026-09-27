"""Real-world XTF header quirks that must not lose the starboard channel.

Each case writes a normal two-channel (port/starboard) file and then patches
bytes to mimic what other acquisition software writes.
"""

import struct

import numpy as np
import pytest

import xtfjsf as xj
from xtfjsf.formats import xtf as xtfmod
from xtfjsf.simulate import simulate_line

N = 500


@pytest.fixture
def line(tmp_path):
    pings = simulate_line(num_pings=20, num_samples=N, max_range=25.0, altitude=5.0, targets=())
    path = tmp_path / "q.xtf"
    with xj.XTFWriter(path, dtype="uint16") as w:
        for p in pings:
            w.write_ping(p)
    return path, pings


def _patch(path, file_header=None, chan_header=None):
    """file_header(buf) patches the 1024-byte header; chan_header(buf, off, k)
    patches the k-th channel header (at byte ``off``) of every sonar packet."""
    data = bytearray(path.read_bytes())
    if file_header:
        file_header(data)
    if chan_header:
        with xj.XTFFile(path) as f:
            offsets = [r.offset for r in f.index if r.header_type == 0]
        for pos in offsets:
            off = pos + xtfmod.PING_HEADER.size
            for k in range(2):
                chan_header(data, off, k)
                off += xtfmod.PING_CHAN_HEADER.size + N * 2
    path.write_bytes(bytes(data))


def _check(path, pings):
    with xj.XTFFile(path) as f:
        out = f.read_all()
    assert len(out) == len(pings)
    for a, b in zip(pings, out):
        assert b.port is not None and b.starboard is not None, [c.side for c in b.channels]
        np.testing.assert_allclose(b.port.samples, np.rint(a.port.samples))
        np.testing.assert_allclose(b.starboard.samples, np.rint(a.starboard.samples))
        assert b.starboard.range_resolution == pytest.approx(a.starboard.range_resolution, rel=1e-5)
    wf = xj.build_waterfall(out)
    assert wf.port.any(axis=1).all() and wf.starboard.any(axis=1).all()
    return out


def test_reference_file_is_fine(line):
    _check(*line)


def test_channel_number_always_zero(line):
    path, pings = line
    _patch(path, chan_header=lambda d, off, k: struct.pack_into("<H", d, off, 0))
    _check(path, pings)


def test_channel_number_one_based(line):
    path, pings = line
    _patch(path, chan_header=lambda d, off, k: struct.pack_into("<H", d, off, k + 1))
    _check(path, pings)


def test_bytes_per_sample_missing(line):
    path, pings = line

    def fh(d):
        for i in range(2):
            struct.pack_into("<H", d, 256 + i * 128 + 6, 0)

    _patch(path, file_header=fh)
    _check(path, pings)


def test_bytes_per_sample_wrong(line):
    path, pings = line

    def fh(d):
        for i in range(2):
            struct.pack_into("<H", d, 256 + i * 128 + 6, 1)

    _patch(path, file_header=fh)
    _check(path, pings)


def test_channel_types_unset(line):
    path, pings = line

    def fh(d):
        for i in range(2):
            d[256 + i * 128] = 0

    _patch(path, file_header=fh)
    _check(path, pings)


def test_both_channel_types_port(line):
    path, pings = line

    def fh(d):
        for i in range(2):
            d[256 + i * 128] = 1

    _patch(path, file_header=fh)
    _check(path, pings)


def test_starboard_without_range_fields(line):
    path, pings = line

    def ch(d, off, k):
        if k == 1:
            struct.pack_into("<f", d, off + 4, 0.0)  # SlantRange
            struct.pack_into("<f", d, off + 16, 0.0)  # TimeDuration

    _patch(path, chan_header=ch)
    _check(path, pings)


def test_everything_at_once(line):
    path, pings = line

    def fh(d):
        for i in range(2):
            d[256 + i * 128] = 0
            struct.pack_into("<H", d, 256 + i * 128 + 6, 0)

    _patch(path, file_header=fh, chan_header=lambda d, off, k: struct.pack_into("<H", d, off, 0))
    _check(path, pings)
