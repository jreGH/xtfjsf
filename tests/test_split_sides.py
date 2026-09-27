"""Systems that write port and starboard to separate files."""

import math
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

import xtfjsf as xj
from xtfjsf.cli import main
from xtfjsf.core import Ping
from xtfjsf.simulate import simulate_line


def _single_side(p, side, label=None, dt=0.0):
    ch = p.channel(side)
    return replace(p, time=p.time + timedelta(seconds=dt), channels=[replace(ch, side=label or side)])


def _write(path, pings):
    with xj.XTFWriter(path, dtype="float32") as w:
        for p in pings:
            w.write_ping(p)
    return path


@pytest.fixture
def split(tmp_path):
    pings = simulate_line(num_pings=80, num_samples=600, max_range=30.0, altitude=5.0, targets=((40, 12.0, 1.0),))
    port = _write(tmp_path / "line_port_b_1.xtf", [_single_side(p, xj.PORT) for p in pings])
    # starboard clock 4 ms late and one ping missing
    stbd_pings = [_single_side(p, xj.STARBOARD, dt=0.004) for i, p in enumerate(pings) if i != 10]
    stbd = _write(tmp_path / "line_stbd_b_1.xtf", stbd_pings)
    return pings, port, stbd


def test_single_side_file_is_one_sided(split):
    _, port, _ = split
    with pytest.warns(UserWarning, match="no starboard channel in any ping"):
        wf = xj.read_waterfall(port)
    assert wf.port.any() and not wf.starboard.any()


def test_read_pair_matches_two_sided_line(split):
    pings, port, stbd = split
    ref = xj.build_waterfall(pings)
    wf = xj.read_waterfall_pair(port, stbd)
    assert wf.metadata == {"paired_pings": 79, "port_only": 1, "starboard_only": 0}
    assert wf.num_pings == 80 and wf.resolution == pytest.approx(ref.resolution)
    np.testing.assert_allclose(wf.port, ref.port, atol=1e-3)
    keep = np.arange(80) != 10
    np.testing.assert_allclose(wf.starboard[keep], ref.starboard[keep], atol=1e-3)
    assert not wf.starboard[10].any()
    np.testing.assert_allclose(wf.latitude, ref.latitude)


def test_mislabelled_starboard_file(tmp_path, split):
    pings, port, _ = split
    bad = _write(tmp_path / "bad.xtf", [_single_side(p, xj.STARBOARD, label=xj.PORT) for p in pings])
    with pytest.warns(UserWarning):
        assert xj.read_waterfall(bad).port.any()  # file claims to be port
    wf = xj.read_waterfall(bad, side="starboard")
    assert wf.starboard.any() and not wf.port.any()
    pair = xj.read_waterfall_pair(port, bad)
    assert pair.metadata["paired_pings"] == 80 and pair.starboard.any()


def test_combine_different_resolutions_and_ranges():
    a = xj.build_waterfall(simulate_line(num_pings=10, num_samples=400, max_range=20.0), side="port")
    b = xj.build_waterfall(simulate_line(num_pings=10, num_samples=1000, max_range=40.0), side="starboard")
    wf = xj.combine_sides(a, b)
    assert wf.resolution == pytest.approx(0.04)
    assert wf.max_range == pytest.approx(40.0)
    assert wf.port[:, : int(19.9 / 0.04)].all() and not wf.port[:, int(20.1 / 0.04) :].any()


def test_combine_by_ping_number_without_times():
    pings = simulate_line(num_pings=12, targets=())
    strip = lambda ps: [replace(p, time=None) for p in ps]
    a = xj.build_waterfall(strip(pings), side="port")
    b = xj.build_waterfall(strip(pings[::-1]), side="starboard")  # different order
    wf = xj.combine_sides(a, b)
    assert wf.metadata["paired_pings"] == 12
    ref = xj.build_waterfall(pings)
    np.testing.assert_allclose(wf.starboard, ref.starboard, atol=1e-4)


def test_invalid_side():
    with pytest.raises(ValueError):
        xj.build_waterfall(simulate_line(num_pings=2), side="left")


def test_cli_pair_waterfall_mosaic_and_table(split, tmp_path, capsys):
    pings, port, stbd = split
    out = tmp_path / "pair.png"
    assert main(["waterfall", str(port), str(out), "--starboard", str(stbd)]) == 0
    assert "80 pings" in capsys.readouterr().out

    main(["info", "--table", str(port), str(stbd)])
    table = capsys.readouterr().out.splitlines()
    assert len(table) == 3 and "port 400kHz 600xfloat32 5.0cm 30m" in table[1]
    assert "star 400kHz" in table[2]

    pytest.importorskip("PIL")
    # target is 12 m to starboard: it must land on the starboard side of the track
    for args in (["--port", str(port), "--starboard", str(stbd)], [str(port), str(stbd)]):
        tif = tmp_path / "m.tif"
        cmd = ["mosaic", *args, "-o", str(tif), "--cell-size", "0.25", "--despeckle", "3"]
        if args[0].startswith("--"):
            assert main(cmd) == 0
        else:  # unforced single-side files work too, but warn
            with pytest.warns(UserWarning, match="one file per side"):
                assert main(cmd) == 0
        from PIL import Image

        im = np.array(Image.open(tif))
        tag = Image.open(tif).tag_v2
        x0, y0 = tag[33922][3:5]
        rows, cols = np.nonzero(np.nan_to_num(im) > np.nanpercentile(im, 99.9))
        x = x0 + (cols.mean() + 0.5) * 0.25
        y = y0 - (rows.mean() + 0.5) * 0.25
        e, n, _, _ = xj.latlon_to_utm(pings[40].latitude, pings[40].longitude)
        hd = math.radians(pings[40].heading)
        across = (x - e) * math.cos(hd) - (y - n) * math.sin(hd)  # + = starboard
        assert across == pytest.approx(12.0, abs=2.0)
