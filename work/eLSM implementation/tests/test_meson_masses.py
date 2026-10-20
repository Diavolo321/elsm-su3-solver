import numpy as np
from elsm.meson_masses import wavefunction_factors, tree_level_meson_mass_squared, mixed_mass_eigenvalues


def test_wavefunction_factors_positive_for_consistent_synthetic_inputs():
    z = wavefunction_factors(0.10, 0.14, g1=5.58,
                             m_a1=1.10, m_k1=1.25, m_f1s=1.42, m_kstar=0.89)
    assert set(z) == {"Z_pi", "Z_K", "Z_etaS", "Z_K0star"}
    assert all(np.isfinite(v) and v > 0 for v in z.values())


def test_table_a1_mass_formulas_return_matrices_and_channels():
    z = wavefunction_factors(0.10, 0.14, g1=5.58,
                             m_a1=1.10, m_k1=1.25, m_f1s=1.42, m_kstar=0.89)
    m = tree_level_meson_mass_squared(
        0.10, 0.14, m0_sq=-0.066564, c_anomaly=2.828,
        lambda1=3.15, lambda2=36.253, Z=z,
        g1=5.58, c1=0.44, h2=40.975, h3=-15.216)
    assert m["scalar_isoscalar_matrix"].shape == (2, 2)
    assert m["pseudoscalar_isoscalar_matrix"].shape == (2, 2)
    assert "rho" in m and "a1" in m and "K" in m
    assert np.all(np.isfinite(mixed_mass_eigenvalues(m["scalar_isoscalar_matrix"])))


def test_scalar_anomaly_terms_match_table_a1_and_potential_hessian():
    # Table A1 gives -c*sigma_y/sqrt(2) and -c*sigma_x/sqrt(2).
    # These entries must match the Hessian of Eq. (14) in the (sigma_x,sigma_y) basis.
    from elsm.parameters import SU3Parameters
    from elsm.potential import mesonic_hessian
    sx, sy = 0.126, 0.098
    p = SU3Parameters(m0_sq=-0.066564, lambda1=3.15, lambda2=36.253,
                     c_anomaly=2.828, h_x=0.01, h_y=0.02, g_yukawa=5.5)
    z = wavefunction_factors(sx, sy, g1=5.58, m_a1=1.06976,
                             m_k1=1.25346, m_f1s=1.4264, m_kstar=.89555)
    m = tree_level_meson_mass_squared(sx, sy, m0_sq=p.m0_sq,
        c_anomaly=p.c_anomaly, lambda1=p.lambda1, lambda2=p.lambda2, Z=z)
    h = mesonic_hessian(sx, sy, p)
    assert np.isclose(m["sigma_N"], h[0, 0], rtol=0, atol=1e-12)
    assert np.isclose(m["sigma_NS"], h[0, 1], rtol=0, atol=1e-12)
