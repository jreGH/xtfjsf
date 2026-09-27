"""Pairing files recorded per-line, per-side, purely by ping time."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import xtfjsf as xj
from xtfjsf.cli import main
from xtfjsf.simulate import simulate_line


def _write_line(tmp_path, name, start, num_pings=30, side_offset=0.0):
    """Write a port file and a starboard file for one mission, unrelated names."""
    pings = simulate_line(num_pings=num_pings, start_time=start, targets=())
    port_path = tmp_path / f"{name}_b_1.xtf"
    stbd_path = tmp_path / f"{name}_s1_h_01.xtf"
    with xj.XTFWriter(port_path, dtype="float32") as w:
        for p in pings:
            w.write_ping(replace(p, channels=[p.channel(xj.PORT)]))
    with xj.XTFWriter(stbd_path, dtype="float32") as w:
        for p in pings:
            w.write_ping(replace(
                p,
                time=p.time + timedelta(seconds=side_offset),
                channels=[p.channel(xj.STARBOARD)],
            ))
    return port_path, stbd_path


@pytest.fixture
def missions(tmp_path):
    """Three lines, minutes apart, named so filename similarity cannot help."""
    t0 = datetime(2024, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
    m1 = _write_line(tmp_path, "mission_alpha", t0)
    m2 = _write_line(tmp_path, "line_two", t0 + timedelta(minutes=10), side_offset=0.05)
    m3 = _write_line(tmp_path, "survey_c", t0 + timedelta(minutes=20), num_pings=15)
    return tmp_path, [m1, m2, m3]


def test_matches_by_time_not_name(missions):
    tmp_path, lines = missions
    ports = [p for p, s in lines]
    stbds = [s for p, s in lines]
    result = xj.match_mission_files(ports, stbds)
    assert len(result.pairs) == 3
    assert not result.unmatched_port and not result.unmatched_starboard
    got = {(m.port, m.starboard) for m in result.pairs}
    assert got == {(p, s) for p, s in lines}
    for m in result.pairs:
        assert m.overlap_seconds > 0
        assert m.port_pings == m.starboard_pings


def test_unmatched_file_is_reported(missions):
    tmp_path, lines = missions
    ports = [lines[0][0], lines[1][0]]
    stbds = [lines[0][1], lines[1][1], lines[2][1]]  # extra starboard, no port
    result = xj.match_mission_files(ports, stbds)
    assert len(result.pairs) == 2
    assert result.unmatched_starboard == [lines[2][1]]
    assert not result.unmatched_port


def test_clock_drift_tolerance(tmp_path):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # starboard logger's clock is 3 s ahead: lines just miss overlapping
    port, stbd = _write_line(tmp_path, "x", t0, num_pings=5, side_offset=0.0)
    pings = simulate_line(num_pings=5, start_time=t0 + timedelta(seconds=3.5), targets=())
    with xj.XTFWriter(stbd, dtype="float32") as w:
        for p in pings:
            w.write_ping(replace(p, channels=[p.channel(xj.STARBOARD)]))
    result = xj.match_mission_files([port], [stbd], tolerance_s=5.0)
    assert len(result.pairs) == 1
    result = xj.match_mission_files([port], [stbd], tolerance_s=1.0)
    assert not result.pairs
    assert result.unmatched_port == [port]


def test_time_bounds_matches_full_scan(missions):
    _, lines = missions
    port, stbd = lines[1]
    with xj.XTFFile(port) as f:
        t0, t1, n = f.time_bounds()
        pings = f.read_all()
    assert n == len(pings) == 30
    assert t0 == pings[0].time and t1 == pings[-1].time


def test_cli_pair_and_batch_waterfall(missions, tmp_path, capsys):
    _, lines = missions
    ports = [p for p, s in lines]
    stbds = [s for p, s in lines]
    assert main(["pair", "--port", *map(str, ports), "--starboard", *map(str, stbds)]) == 0
    out, err = capsys.readouterr()
    assert out.count("<->") == 3
    assert "3 pairs, 0 unmatched" in err

    outdir = tmp_path / "out"
    assert main([
        "batch-waterfall", "--port", *map(str, ports), "--starboard", *map(str, stbds),
        "-o", str(outdir), "--despeckle", "3",
    ]) == 0
    pngs = sorted(outdir.glob("*.png"))
    assert len(pngs) == 3
    assert {p.name for p in pngs} == {"mission_alpha_b_1.png", "line_two_b_1.png", "survey_c_b_1.png"}
    for png in pngs:
        assert png.read_bytes()[:4] == b"\x89PNG"
