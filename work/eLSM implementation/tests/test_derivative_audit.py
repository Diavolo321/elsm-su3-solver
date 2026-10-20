import numpy as np
from elsm.parameters import SU3Parameters
from elsm.potential import mesonic_potential, mesonic_gradient_analytic, mesonic_hessian, quark_masses
from elsm.thermal_masses import sigma_field_mass_derivatives


def _central_gradient(fun, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i in range(x.size):
        xp=x.copy(); xm=x.copy(); xp[i]+=eps; xm[i]-=eps
        out[i]=(fun(*xp)-fun(*xm))/(2*eps)
    return out


def test_mesonic_gradient_matches_finite_difference():
    p = SU3Parameters(m0_sq=-0.066564, lambda1=3.15, lambda2=36.253,
                      c_anomaly=2.828, h_x=0.012, h_y=0.018)
    x=np.array([0.105, 0.135])
    numeric=_central_gradient(lambda sx,sy: mesonic_potential(sx,sy,p), x)
    assert np.allclose(mesonic_gradient_analytic(*x,p), numeric, rtol=2e-6, atol=2e-8)


def test_mesonic_hessian_matches_gradient_finite_difference():
    p = SU3Parameters(m0_sq=-0.066564, lambda1=3.15, lambda2=36.253,
                      c_anomaly=2.828, h_x=0.012, h_y=0.018)
    x=np.array([0.105, 0.135]); eps=1e-6
    numeric=np.column_stack([
        (mesonic_gradient_analytic(*(x+np.eye(2)[i]*eps),p)-
         mesonic_gradient_analytic(*(x-np.eye(2)[i]*eps),p))/(2*eps)
        for i in range(2)])
    assert np.allclose(mesonic_hessian(*x,p), numeric, rtol=2e-6, atol=2e-8)


def test_sigma_quark_mass_derivatives_match_finite_difference():
    sx,sy,g=0.112,0.145,5.5
    d1,d2=sigma_field_mass_derivatives(sx,sy,g)
    # First derivatives of m_f^2
    eps=1e-6
    for j,delta in enumerate(((eps,0.0),(0.0,eps))):
        mp=quark_masses(sx+delta[0],sy+delta[1],g)**2
        mm=quark_masses(sx-delta[0],sy-delta[1],g)**2
        assert np.allclose((mp-mm)/(2*eps), d1[:,j], rtol=2e-6, atol=2e-8)
    # Second derivatives along the diagonal coordinates
    for f in range(3):
        for j,delta in enumerate(((eps,0.0),(0.0,eps))):
            mp=quark_masses(sx+delta[0],sy+delta[1],g)[f]**2
            m0=quark_masses(sx,sy,g)[f]**2
            mm=quark_masses(sx-delta[0],sy-delta[1],g)[f]**2
            assert np.isclose((mp-2*m0+mm)/eps**2, d2[f,j,j], rtol=2e-4, atol=2e-5)
