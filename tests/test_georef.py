import math

import numpy as np
import pytest

import xtfjsf as xj
from xtfjsf.georef import course_made_good, latlon_to_utm, utm_to_latlon, utm_zone
from xtfjsf.simulate import simulate_line


def test_utm_known_values():
    e, n, zone, north = latlon_to_utm(0.0, 3.0)
    assert (zone, north) == (31, True)
    assert float(e) == pytest.approx(500000.0, abs=1e-6)
    assert float(n) == pytest.approx(0.0, abs=1e-6)
    # On the central meridian, northing = k0 * meridian arc length (45 deg: 4 984 944.378 m)
    e, n, _, _ = latlon_to_utm(45.0, 9.0)
    assert float(n) == pytest.approx(0.9996 * 4984944.378, abs=0.01)
    assert utm_zone(-4.0) == 30 and utm_zone(179.9) == 60


def test_utm_roundtrip():
    lat = np.array([-60.0, -33.9, 0.5, 50.1, 71.0])
    lon = np.array([-70.5, 18.4, 3.9, -4.1, 25.8])
    for la, lo in zip(lat, lon):
        e, n, zone, north = latlon_to_utm(la, lo)
        la2, lo2 = utm_to_latlon(e, n, zone, north)
        assert float(la2) == pytest.approx(la, abs=1e-8)
        assert float(lo2) == pytest.approx(lo, abs=1e-8)


def test_course_made_good():
    t = np.arange(50.0)
    assert np.allclose(course_made_good(t, t), 45.0)
    assert np.allclose(course_made_good(-t, 0 * t), 270.0)


@pytest.mark.parametrize("heading", [0.0, 45.0, 200.0])
def test_mosaic_places_target_correctly(heading):
    pings = simulate_line(num_pings=160, heading=heading, altitude=6.0, max_range=40.0, num_samples=800,
                          targets=((80, 15.0, 1.5),))
    wf = xj.slant_range_correct(xj.build_waterfall(pings), np.full(160, 6.0))
    trk = xj.track(wf)
    # expected target position: 15 m to starboard of ping 80
    hd = math.radians(heading)
    ex = trk.x[80] + 15.0 * math.cos(hd)
    ey = trk.y[80] - 15.0 * math.sin(hd)

    m = xj.Mosaic(cell_size=0.25, method="nearest")
    m.add(wf)
    r = m.render(fill_holes=0)
    assert r.epsg == 32630
    img = np.nan_to_num(r.data)
    # smooth then find brightest cell
    k = 5
    c = np.cumsum(np.cumsum(np.pad(img, ((1, 0), (1, 0))), 0), 1)
    box = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    row, col = np.unravel_index(np.argmax(box), box.shape)
    px = r.x0 + (col + k / 2) * r.cell_size
    py = r.y0 - (row + k / 2) * r.cell_size
    assert math.hypot(px - ex, py - ey) < 1.5


def test_mosaic_methods_and_outputs(tmp_path, pings):
    wf = xj.process(xj.build_waterfall(pings))
    for method in ("nearest", "mean", "max"):
        m = xj.Mosaic(cell_size=0.5, method=method)
        m.add(wf)
        r = m.render()
        assert np.isfinite(r.data).sum() > 0.3 * r.data.size
    png = r.to_png(tmp_path / "m.png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    wld = (tmp_path / "m.pgw").read_text().split()
    assert float(wld[0]) == 0.5 and float(wld[3]) == -0.5
    assert (tmp_path / "m.epsg").read_text().strip() == "EPSG:32630"

    Image = pytest.importorskip("PIL.Image")
    tif = r.to_geotiff(tmp_path / "m.tif")
    im = Image.open(tif)
    assert im.mode == "F" and im.size == (r.shape[1], r.shape[0])
    assert im.tag_v2[33922][3:5] == (r.x0, r.y0)
    assert tuple(im.tag_v2[34735])[-1] == 32630
    tif8 = r.to_geotiff(tmp_path / "m8.tif", dtype="uint8")
    assert Image.open(tif8).mode == "L"


def test_projected_navigation_mosaic():
    pings = simulate_line(num_pings=50, targets=())
    for i, p in enumerate(pings):
        p.latitude = p.longitude = math.nan
        p.easting, p.northing = 400000.0 + i * 0.2, 6000000.0
    wf = xj.slant_range_correct(xj.build_waterfall(pings))
    m = xj.Mosaic(cell_size=1.0, epsg=25832)
    m.add(wf, use_heading="cog")
    r = m.render()
    assert r.epsg == 25832
    # track runs east along y = 6 000 000; swath extends ~49 m north and south
    assert r.x0 <= 400000.0 < r.x0 + r.shape[1] * r.cell_size
    assert r.y0 == pytest.approx(6000049.4, abs=1.5)
    assert r.shape[0] * r.cell_size == pytest.approx(99.0, abs=3.0)
