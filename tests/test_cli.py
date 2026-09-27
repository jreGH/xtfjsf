import csv
import json

import pytest

import xtfjsf as xj
from xtfjsf.cli import main


@pytest.fixture
def files(tmp_path, pings):
    x = tmp_path / "l.xtf"
    j = tmp_path / "l.jsf"
    with xj.XTFWriter(x) as w, xj.JSFWriter(j) as w2:
        for p in pings:
            w.write_ping(p)
            w2.write_ping(p)
    return x, j


def test_info(files, capsys):
    assert main(["info", str(files[0])]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["format"] == "xtf" and info["pings"] == 120
    assert main(["info", str(files[1])]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["subsystems_present"] == [20]


def test_nav(files, tmp_path):
    out = tmp_path / "nav.csv"
    assert main(["nav", str(files[1]), "-o", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 120
    assert float(rows[0]["latitude"]) == pytest.approx(50.0, abs=1e-5)


def test_waterfall_and_mosaic(files, tmp_path):
    png = tmp_path / "wf.png"
    assert main(["waterfall", str(files[0]), str(png), "--absorption", "60", "--despeckle", "3"]) == 0
    assert png.read_bytes()[:4] == b"\x89PNG"
    assert main(["waterfall", str(files[1]), str(tmp_path / "raw.png"), "--raw", "--log"]) == 0
    pytest.importorskip("PIL")
    tif = tmp_path / "m.tif"
    assert main(["mosaic", str(files[0]), str(files[1]), "-o", str(tif), "--cell-size", "0.5"]) == 0
    assert tif.exists()


def test_convert(files, tmp_path):
    out = tmp_path / "c.xtf"
    assert main(["convert", str(files[1]), str(out)]) == 0
    with xj.open_sonar(out) as f:
        assert len(f.read_all()) == 120


def test_reverse_port_flag(files, tmp_path):
    fixed = tmp_path / "fixed.png"
    wrong = tmp_path / "wrong.png"
    assert main(["waterfall", str(files[0]), str(fixed), "--reverse-port", "--raw"]) == 0
    assert main(["waterfall", str(files[0]), str(wrong), "--raw"]) == 0
    assert fixed.read_bytes() != wrong.read_bytes()
