import numpy as np
import pytest

import xtfjsf as xj
from xtfjsf.core import ChannelData, Ping
from xtfjsf.processing import grazing_angles, median_filter
from xtfjsf.simulate import simulate_line


def test_waterfall_resamples_mixed_ranges():
    a = Ping(ping_number=1, channels=[
        ChannelData(np.ones(100, np.float32), xj.PORT, 0, sample_interval=2 * 0.1 / 1500),
        ChannelData(np.ones(100, np.float32), xj.STARBOARD, 1, sample_interval=2 * 0.1 / 1500),
    ])
    b = Ping(ping_number=2, channels=[
        ChannelData(np.full(400, 2, np.float32), xj.PORT, 0, sample_interval=2 * 0.05 / 1500),
    ])
    with pytest.warns(UserWarning, match="1 of 2 pings have no starboard"):
        wf = xj.build_waterfall([a, b])
    assert wf.resolution == pytest.approx(0.05)
    assert wf.max_range == pytest.approx(20.0)
    assert wf.port.shape == (2, 400)
    # first ping only reaches 10 m: beyond that it is zero-padded
    assert wf.port[0, :199].min() == pytest.approx(1.0)
    assert wf.port[0, 205:].max() == 0
    assert wf.starboard[1].max() == 0
    assert wf.image().shape == (2, 800)
    assert wf.across_track()[0] == pytest.approx(-19.975)


def test_bottom_track_follows_altitude():
    alt = 5.0 + 2.0 * np.sin(np.linspace(0, 3, 150))
    wf = xj.build_waterfall(simulate_line(num_pings=150, altitude=alt, targets=()))
    est = xj.bottom_track(wf)
    assert np.nanmax(np.abs(est - alt)) < 0.25


def test_slant_range_correction_places_target_at_ground_range(pings):
    wf = xj.build_waterfall(pings)
    gr = xj.slant_range_correct(wf, np.full(wf.num_pings, 6.0))
    assert gr.ground_range
    assert gr.max_range == pytest.approx(np.sqrt(40**2 - 36), abs=0.1)
    row = gr.starboard[60]
    peak_range = gr.ranges[np.argmax(xj.processing._moving_average(row, 9, 0))]
    assert peak_range == pytest.approx(15.0, abs=0.6)
    # the shadow behind the target is dark
    behind = (gr.ranges > 16.0) & (gr.ranges < 17.0)
    assert row[behind].mean() < 0.3 * gr.starboard[10][behind].mean()


def test_tvg_and_normalization_flatten_across_track(pings):
    wf = xj.build_waterfall(pings)
    wf = xj.slant_range_correct(wf)
    before = wf.starboard[:, 50:].mean(axis=0)
    flat = xj.normalize_beam_pattern(xj.tvg(wf, spreading=20), mode="angle")
    after = flat.starboard[:, 50:].mean(axis=0)
    cv = lambda c: np.std(xj.processing._moving_average(c, 25, 0)) / np.mean(c)
    assert cv(after) < 0.1 < cv(before)
    np.testing.assert_allclose(flat.port[:, 50:].mean(), flat.starboard[:, 50:].mean(), rtol=0.1)


def test_range_mode_windowed_normalization(pings):
    wf = xj.build_waterfall(pings).magnitude()
    out = xj.normalize_beam_pattern(wf, mode="range", window=21)
    assert out.port.shape == wf.port.shape
    assert np.isfinite(out.port).all()


def test_grazing_angles():
    wf = xj.build_waterfall(simulate_line(num_pings=3, altitude=10.0, targets=()))
    ang = grazing_angles(wf)
    assert np.isnan(ang[0, 0])
    i = np.argmin(np.abs(wf.ranges - 20.0))
    assert ang[0, i] == pytest.approx(60.0, abs=0.2)


def test_multilook_and_despeckle(sas_pings):
    wf = xj.build_waterfall(sas_pings, keep_complex=True)
    ml = xj.multilook(wf, along=2, across=3)
    assert ml.num_pings == wf.num_pings // 2
    assert ml.num_bins == wf.num_bins // 3
    assert ml.resolution == pytest.approx(wf.resolution * 3)
    assert not ml.is_complex
    # speckle contrast (std/mean) falls with more looks
    seabed = slice(ml.num_bins // 2, None)
    single = np.abs(wf.starboard[:, wf.num_bins // 2 :])
    assert (ml.starboard[:, seabed].std() / ml.starboard[:, seabed].mean()) < single.std() / single.mean()
    ds = xj.despeckle(wf, 3)
    assert ds.port.shape == wf.port.shape


def test_median_filter_matches_naive():
    rng = np.random.default_rng(1)
    a = rng.random((7, 9))
    out = median_filter(a, 3)
    p = np.pad(a, 1, mode="edge")
    assert out[3, 4] == pytest.approx(np.median(p[3:6, 4:7]))
    assert out[0, 0] == pytest.approx(np.median(p[0:3, 0:3]))


def test_stretch_and_db():
    img = np.linspace(0, 100, 1000).reshape(10, 100)
    s = xj.stretch(img, 0, 100)
    assert s.dtype == np.uint8 and s.max() == 255 and s[0, 0] == 0
    assert xj.to_db(np.array([1.0, 10.0]))[1] == pytest.approx(20.0)


def test_process_pipeline_complex(sas_pings):
    wf = xj.build_waterfall(sas_pings, keep_complex=True)
    out = xj.process(wf, looks=(1, 2), despeckle_size=3)
    assert out.ground_range and not out.is_complex
    assert np.isfinite(out.image()).all()


def test_waterfall_to_pings_roundtrip(tmp_path, pings):
    wf = xj.build_waterfall(pings)
    with xj.XTFWriter(tmp_path / "p.xtf", dtype="float32") as w:
        for p in xj.waterfall_to_pings(wf):
            w.write_ping(p)
    back = xj.read_waterfall(tmp_path / "p.xtf")
    np.testing.assert_allclose(back.port, wf.port, rtol=1e-5, atol=1e-3)


def test_waterfall_warns_when_a_whole_side_is_missing(pings):
    port_only = [Ping(ping_number=p.ping_number, channels=[p.port]) for p in pings[:5]]
    with pytest.warns(UserWarning, match="no starboard channel in any ping"):
        xj.build_waterfall(port_only)
