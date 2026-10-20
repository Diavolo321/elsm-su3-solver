import numpy as np
from elsm import fit_condensates_pcac


def test_pcac_fit_reconstructs_decay_constants_and_reports_yukawa_tension():
    fit = fit_condensates_pcac(
        f_pi=0.09502, f_K=0.1069, m_a1=1.06976, m_k1=1.25346,
        g1=5.58, m0_sq=-0.066564, lambda1=3.15, lambda2=36.253,
        c_anomaly=2.828, g_yukawa=5.5, constituent_mass=0.300)
    assert np.isclose(fit.f_pi_reconstructed, 0.09502, atol=1e-8)
    assert np.isclose(fit.f_K_reconstructed, 0.1069, atol=1e-8)
    assert fit.sigma_x > 0 and fit.sigma_y > 0
    # The paper's independent g=2*m_q/sigma_x condition does not match g=5.5.
    assert not np.isclose(fit.g_from_constituent_mass, 5.5, atol=0.05)
    assert np.isfinite(fit.h_x) and np.isfinite(fit.h_y)
