import numpy as np
from elsm.parameters import SU3Parameters
from elsm.potential import mesonic_gradient_analytic


def test_hx_hy_from_condensates_enforce_tree_level_stationarity():
    p = SU3Parameters.from_vacuum_condensates(
        m0_sq=-0.066564, lambda1=3.15, lambda2=36.253,
        c_anomaly=2.828, sigma_x=0.109, sigma_y=0.14)
    grad = mesonic_gradient_analytic(0.109, 0.14, p)
    assert np.allclose(grad, [0, 0], atol=1e-12)
    assert np.isfinite(p.h_x) and np.isfinite(p.h_y)
