"""Regression tests tying the implementation to the source paper.

1. Mass formulas vs. an independent numerical Hessian of the Lagrangian (Eqs. 2, 5).
2. The Yukawa vacuum (sigma_x = 2 m_q/g) reproduces Tables I and V.
3. Eq. (32) normalization equals the second derivative of the -2T quark potential.
4. T_chi (c != 0, no F(T)) agrees with Table III within one grid step.
"""
import numpy as np
from elsm.meson_masses import wavefunction_factors, tree_level_meson_mass_squared
from elsm.vacuum_fit import fit_condensates_yukawa, TABLE_I_SU3

ROW_C = dict(m0_sq=-0.258**2, lambda1=3.15, lambda2=36.253, c_anomaly=2.828)
ROW_0 = dict(m0_sq=-0.282**2, lambda1=-10.515, lambda2=55.576, c_anomaly=0.0)

# ---------- 1. independent Lagrangian Hessian --------------------------------
_l = np.zeros((9, 3, 3), complex)
_l[0] = np.sqrt(2/3)*np.eye(3)
_l[1][0, 1] = _l[1][1, 0] = 1
_l[2][0, 1], _l[2][1, 0] = -1j, 1j
_l[3][0, 0], _l[3][1, 1] = 1, -1
_l[4][0, 2] = _l[4][2, 0] = 1
_l[5][0, 2], _l[5][2, 0] = -1j, 1j
_l[6][1, 2] = _l[6][2, 1] = 1
_l[7][1, 2], _l[7][2, 1] = -1j, 1j
_l[8] = np.diag([1, 1, -2])/np.sqrt(3)
_T = _l/2
_R = np.array([[np.sqrt(2), 1], [1, -np.sqrt(2)]])/np.sqrt(3)   # (N,S) <- (0,8)


def _lagrangian_curvatures(sx, sy, m0_sq, lambda1, lambda2, c_anomaly):
    hx = m0_sq*sx - c_anomaly/np.sqrt(2)*sx*sy + lambda1*sx*sy**2 + (2*lambda1+lambda2)/2*sx**3
    hy = m0_sq*sy - c_anomaly/(2*np.sqrt(2))*sx**2 + lambda1*sx**2*sy + (lambda1+lambda2)*sy**3
    H = np.diag([hx/2, hx/2, hy/np.sqrt(2)])
    s0 = np.zeros(9); s0[[0, 8]] = np.linalg.solve(_R, [sx, sy])

    def V(s, p):
        P = np.einsum('a,aij->ij', s+1j*p, _T); PP = P.conj().T @ P
        return (m0_sq*np.trace(PP) + lambda1*np.trace(PP)**2 + lambda2*np.trace(PP @ PP)
                - c_anomaly*(np.linalg.det(P)+np.linalg.det(P.conj().T))
                - np.trace(H @ (P+P.conj().T))).real

    def hess(f, x0, idx, h=1e-4):
        out = np.zeros((len(idx), len(idx)))
        for a, i in enumerate(idx):
            for b, j in enumerate(idx):
                ei = np.zeros(9); ej = np.zeros(9); ei[i] = h; ej[j] = h
                out[a, b] = (f(x0+ei+ej)-f(x0+ei-ej)-f(x0-ei+ej)+f(x0-ei-ej))/(4*h*h)
        return out
    z9 = np.zeros(9)
    hp = hess(lambda p: V(s0, p), z9, [1, 4, 0, 8])
    hs = hess(lambda s: V(s, z9), s0, [1, 4, 0, 8])
    return dict(pi=hp[0, 0], K=hp[1, 1], eta=_R @ hp[2:, 2:] @ _R.T,
                a0=hs[0, 0], K0star=hs[1, 1], sig=_R @ hs[2:, 2:] @ _R.T)


def test_table_a1_matches_lagrangian_hessian_both_rows():
    sx, sy = 0.10909, 0.1036
    ones = {"Z_pi": 1.0, "Z_K": 1.0, "Z_etaS": 1.0, "Z_K0star": 1.0}
    for row in (ROW_C, ROW_0):
        L = _lagrangian_curvatures(sx, sy, **row)
        m = tree_level_meson_mass_squared(sx, sy, Z=ones, **row)
        for key in ("pi", "K", "a0", "K0star"):
            assert np.isclose(m[key], L[key], atol=2e-5), key
        assert np.allclose(m["pseudoscalar_isoscalar_matrix"], L["eta"], atol=2e-5)
        assert np.allclose(m["scalar_isoscalar_matrix"], L["sig"], atol=2e-5)


# ---------- 2. Tables I and V -------------------------------------------------
def test_yukawa_vacuum_reproduces_table_I_and_V():
    for row, key in ((ROW_C, "cfin"), (ROW_0, "c0")):
        v = fit_condensates_yukawa(**row)
        assert np.isclose(v.sigma_x, 0.6/5.5)
        assert 0.1030 < v.sigma_y < 0.1042
        for ch, target in TABLE_I_SU3[key].items():
            assert abs(v.masses[ch]/target - 1) < 0.005, (row, ch, v.masses[ch])
    v = fit_condensates_yukawa(**ROW_C)
    # Table V (SU(3)): a0=0.926, K0*=1.032, f0=1.124 GeV -- predictions, not fitted
    for ch, target in (("a0", 0.926), ("K0star", 1.032), ("f0", 1.124)):
        assert abs(v.masses[ch]/target - 1) < 0.01, (ch, v.masses[ch])
    # Vector sector of Table I is reproduced with the same sigma_x (no PCAC):
    c1, h2, h3, g1 = 0.44, 40.975, -15.216, 5.58
    assert abs(np.sqrt(c1 + 0.5*(h2+h3)*v.sigma_x**2) - 0.77011) < 1e-3
    assert abs(np.sqrt(c1 + (g1**2 + 0.5*(h2-h3))*v.sigma_x**2) - 1.06976) < 1e-3


def test_kaon_known_discrepancy_with_table_I_is_documented():
    # Consistent model value ~495 MeV (= experiment 495.6); Table I prints 420.49.
    v = fit_condensates_yukawa(**ROW_C)
    assert 0.490 < v.masses["K"] < 0.500


# ---------- 3. Eq. (32) normalization -----------------------------------------
def test_eq32_equals_second_derivative_of_thermal_potential():
    from elsm.potential import quark_thermal_potential
    from elsm.thermal_masses import thermal_mass_matrix_correction, sigma_field_mass_derivatives
    T, sx, sy, phi, g, h = 0.2, 0.06, 0.09, 0.4, 5.5, 1e-4
    f = lambda a, b: quark_thermal_potential(T, a, b, phi, phi, g, n_points=3000, p_max=4.0)
    x = np.array([sx, sy]); fd = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i]*h; ej = np.eye(2)[j]*h
            fd[i, j] = (f(*(x+ei+ej))-f(*(x+ei-ej))-f(*(x-ei+ej))+f(*(x-ei-ej)))/(4*h*h)
    d1, d2 = sigma_field_mass_derivatives(sx, sy, g)
    eq32 = thermal_mass_matrix_correction(T, sx, sy, phi, phi, d1, d2, g, n_points=3000, p_max=4.0)
    assert np.allclose(eq32, fd, rtol=1e-4, atol=1e-7)


# ---------- 4. Table III (narrow window, ~20 s) -------------------------------
def test_tchi_c_finite_matches_table_III_within_grid_step():
    from scipy.optimize import minimize
    from elsm.parameters import SU3Parameters, PolyakovParameters
    from elsm.potential import grand_potential
    v = fit_condensates_yukawa(**ROW_C)
    p = SU3Parameters(h_x=v.h_x, h_y=v.h_y, g_yukawa=5.5, **{
        "m0_sq": ROW_C["m0_sq"], "lambda1": ROW_C["lambda1"],
        "lambda2": ROW_C["lambda2"], "c_anomaly": ROW_C["c_anomaly"]})
    poly = PolyakovParameters(T0=0.270)
    temps = np.arange(0.200, 0.2601, 0.0025); sx = []; prev = (0.09, 0.10, 0.1)
    for T in temps:
        obj = lambda z: grand_potential(float(T), z[0], z[1], z[2], z[2], p, polyakov_parameters=poly,
                                        momentum_n_points=160, momentum_p_max=3.0)
        r = min((minimize(obj, x0, method="L-BFGS-B", bounds=((0, .6), (0, .8), (0, .999)))
                 for x0 in (prev, (0.02, 0.08, 0.6))), key=lambda q: q.fun)
        prev = r.x; sx.append(r.x[0])
    t_chi = temps[int(np.argmax(np.abs(np.gradient(sx, temps))))]
    assert abs(t_chi - 0.2325) <= 0.0026, t_chi
