import pytest
from elsm.parameters import PolyakovParameters
from elsm.potential import polyakov_log_potential


def test_polyakov_standard_coefficients_are_configured():
    assert polyakov_log_potential(0.1, 0.1, 0.15, PolyakovParameters()) < float('inf')


def test_polyakov_rejects_fields_outside_unit_interval():
    assert polyakov_log_potential(1.0, 0.1, 0.15, PolyakovParameters()) == float('inf')
