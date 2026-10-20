import numpy as np
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.potential import (mesonic_potential, mesonic_gradient_analytic,
                            mesonic_hessian, polyakov_log_potential,
                            quark_thermal_potential, vacuum_quark_potential)


def params():
    return SU3Parameters(m0_sq=-0.066564, lambda1=3.15, lambda2=36.253,
                         c_anomaly=2.828, h_x=0.001, h_y=0.02)


def test_gradient_matches_finite_difference():
    p = params(); x = np.array([0.11, 0.14]); eps = 1e-6
    numerical = np.array([
        (mesonic_potential(*(x + np.eye(2)[i]*eps), p) -
         mesonic_potential(*(x - np.eye(2)[i]*eps), p))/(2*eps)
        for i in range(2)])
    assert np.allclose(numerical, mesonic_gradient_analytic(*x, p), rtol=1e-5, atol=1e-7)


def test_hessian_matches_gradient_difference():
    p = params(); x = np.array([0.11, 0.14]); eps = 1e-6
    numerical = np.column_stack([
        (mesonic_gradient_analytic(*(x + np.eye(2)[i]*eps), p) -
         mesonic_gradient_analytic(*(x - np.eye(2)[i]*eps), p))/(2*eps)
        for i in range(2)])
    assert np.allclose(numerical, mesonic_hessian(*x, p), rtol=1e-5, atol=1e-7)


def test_polyakov_reference_parameters_are_finite_in_domain():
    u = polyakov_log_potential(0.1, 0.1, 0.2, PolyakovParameters())
    assert np.isfinite(u)


def test_polyakov_invalid_haar_domain_is_penalized():
    assert np.isinf(polyakov_log_potential(1.0, 0.1, 0.2, PolyakovParameters()))


def test_thermal_quark_potential_vanishes_at_zero_temperature():
    assert quark_thermal_potential(0.0, 0.1, 0.14, 0.1, 0.1) == 0.0


def test_vacuum_term_requires_explicit_cutoff():
    try:
        vacuum_quark_potential(0.1, 0.14)
    except ValueError:
        pass
    else:
        raise AssertionError("vacuum term must require an explicit cutoff")


def test_polyakov_thermal_log_does_not_double_count_color():
    from elsm.potential import quark_thermal_potential
    # The Polyakov-modified logarithm contains the color trace explicitly.
    # The legacy n_c argument must therefore not rescale the result.
    args = (0.15, 0.10, 0.12, 0.2, 0.2)
    omega_nc3 = quark_thermal_potential(*args, n_c=3, n_points=100, p_max=2.0)
    omega_nc1 = quark_thermal_potential(*args, n_c=1, n_points=100, p_max=2.0)
    assert np.isclose(omega_nc3, omega_nc1, rtol=0, atol=1e-14)


def test_literal_eq27_thermal_normalization_is_explicit_factor_six():
    args = (0.15, 0.10, 0.12, 0.2, 0.2)
    standard = quark_thermal_potential(*args, n_points=100, p_max=2.0)
    literal_prefactor = quark_thermal_potential(
        *args, n_points=100, p_max=2.0, thermal_normalization=6.0)
    assert np.isclose(literal_prefactor, 6.0*standard, rtol=1e-13)
