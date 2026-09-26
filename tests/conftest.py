import pytest

from xtfjsf.simulate import simulate_line


@pytest.fixture(scope="session")
def pings():
    return simulate_line(num_pings=120, num_samples=800, max_range=40.0, altitude=6.0, targets=((60, 15.0, 1.5),))


@pytest.fixture(scope="session")
def sas_pings():
    return simulate_line(num_pings=40, num_samples=600, max_range=30.0, complex_data=True, targets=())
