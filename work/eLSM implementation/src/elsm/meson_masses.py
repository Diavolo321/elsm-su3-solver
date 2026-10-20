"""SU(3) tree-level meson mass formulas from Appendix B, Table A1.

Masses are returned as mass-squared values in GeV^2. Mixing channels return a
2x2 mass-squared matrix in the (non-strange, strange) basis. The formulas use
the article's condensate convention and explicit wave-function factors Z_i.
Finite-temperature Eq. (32) corrections are applied separately in thermal_masses.py.
"""
import numpy as np


def wavefunction_factors(sigma_x, sigma_y, *, g1, m_a1, m_k1, m_f1s, m_kstar):
    """Compute Z_pi, Z_K, Z_etaS, Z_K0star from Appendix B Eq. (A13).

    The axial/vector masses must be the same fitted vacuum masses used by the
    chosen parameter set. A non-positive radicand indicates an inconsistent fit.
    """
    sx, sy = float(sigma_x), float(sigma_y)
    g1 = float(g1)
    rad = {
        "Z_pi": m_a1**2 - g1**2*sx**2,
        "Z_K": 4*m_k1**2 - g1**2*(sx + np.sqrt(2)*sy)**2,
        "Z_etaS": m_f1s**2 - 2*g1**2*sy**2,
        "Z_K0star": 4*m_kstar**2 - g1**2*(sx - np.sqrt(2)*sy)**2,
    }
    bad = [name for name, value in rad.items() if value <= 0]
    if bad:
        raise ValueError("Non-positive wave-function-factor radicand(s): " + ", ".join(bad))
    return {
        "Z_pi": float(m_a1/np.sqrt(rad["Z_pi"])),
        "Z_K": float(2*m_k1/np.sqrt(rad["Z_K"])),
        "Z_etaS": float(m_f1s/np.sqrt(rad["Z_etaS"])),
        "Z_K0star": float(2*m_kstar/np.sqrt(rad["Z_K0star"])),
    }


def tree_level_meson_mass_squared(sigma_x, sigma_y, *, m0_sq, c_anomaly,
                                  lambda1, lambda2, Z=None,
                                  g1=None, c1=None, h2=None, h3=None,
                                  delta_x=0.0, delta_y=0.0):
    """Return the Table A1 SU(3) tree-level mass-squared formulas.

    `Z` is the mapping returned by `wavefunction_factors`. Scalar and
    pseudoscalar masses are always returned; vector/axial-vector masses are
    also returned if g1,c1,h2,h3 are supplied. Delta parameters are explicit
    symmetry-breaking terms in the vector/axial-vector sector.
    """
    sx, sy = float(sigma_x), float(sigma_y)
    m0 = float(m0_sq); c = float(c_anomaly)
    l1 = float(lambda1); l2 = float(lambda2)
    Z = {} if Z is None else dict(Z)
    required = ("Z_pi", "Z_K", "Z_etaS", "Z_K0star")
    missing = [k for k in required if k not in Z]
    if missing:
        raise ValueError("Provide wave-function factors: " + ", ".join(missing))

    out = {
        "a0": m0 + c/np.sqrt(2)*sy + (l1 + 1.5*l2)*sx**2 + l1*sy**2,
        # FIX: Table A1 prints the anomaly term as +c*sigma_y/(2 sqrt 2).
        # Direct differentiation of the Lagrangian (Eqs. 2, 5) gives +c*sigma_x/2,
        # the mirror image of the -c*sigma_x/2 term in m_K^2.
        "K0star": Z["Z_K0star"]**2*(
            m0 + c/2*sx + (l1 + 0.5*l2)*sx**2
            + l2/np.sqrt(2)*sx*sy + (l1+l2)*sy**2),
        "sigma_N": m0 - c/np.sqrt(2)*sy + 1.5*(2*l1+l2)*sx**2 + l1*sy**2,
        "sigma_S": m0 + 3*(l1+l2)*sy**2 + l1*sx**2,
        "sigma_NS": -c/np.sqrt(2)*sx + 2*l1*sx*sy,
        "pi": Z["Z_pi"]**2*(
            m0 - c/np.sqrt(2)*sy + (l1+0.5*l2)*sx**2 + l1*sy**2),
        "K": Z["Z_K"]**2*(
            m0 - c/2*sx + (l1+0.5*l2)*sx**2 + (l1+l2)*sy**2
            - l2/np.sqrt(2)*sx*sy),
        "eta_N": Z["Z_pi"]**2*(
            m0 + c/np.sqrt(2)*sy + (l1+0.5*l2)*sx**2 + l1*sy**2),
        "eta_S": Z["Z_etaS"]**2*(m0 + l1*sx**2 + (l1+l2)*sy**2),
        # FIX: Table A1 prints c/2*sigma_x. The Lagrangian gives c/sqrt(2)*sigma_x,
        # and only this value reproduces Table I (eta=531.9, eta'=965.4 MeV).
        "eta_NS": Z["Z_pi"]*Z["Z_etaS"]*c/np.sqrt(2)*sx,
    }
    out["scalar_isoscalar_matrix"] = np.array(
        [[out["sigma_N"], out["sigma_NS"]],
         [out["sigma_NS"], out["sigma_S"]]], dtype=float)
    out["pseudoscalar_isoscalar_matrix"] = np.array(
        [[out["eta_N"], out["eta_NS"]],
         [out["eta_NS"], out["eta_S"]]], dtype=float)

    if all(v is not None for v in (g1, c1, h2, h3)):
        g1, c1, h2, h3 = map(float, (g1, c1, h2, h3))
        out.update({
            "rho": c1 + 0.5*(h2+h3)*sx**2 + 2*delta_x,
            "Kstar": (c1 + 0.25*(g1**2+h2)*(sx**2+2*sy**2)
                      + sx*sy/np.sqrt(2)*(h3-g1**2) + delta_x+delta_y),
            "omega_S": c1 + (h2+h3)*sy**2 + 2*delta_y,
            "omega_N": c1 + 0.5*(h2+h3)*sx**2 + 2*delta_x,
            "a1": c1 + (g1**2+0.5*(h2-h3))*sx**2 + 2*delta_x,
            "K1": (c1 + 0.25*(g1**2+h2)*(sx**2+2*sy**2)
                   - sx*sy/np.sqrt(2)*(h3-g1**2) + delta_x+delta_y),
            "f1_S": c1 + (2*g1**2+h2-h3)*sy**2 + 2*delta_y,
            "f1_N": c1 + (g1**2+0.5*(h2-h3))*sx**2 + 2*delta_x,
        })
    return out


def mixed_mass_eigenvalues(mass_squared_matrix):
    """Sorted mass-squared eigenvalues for a real symmetric mixing matrix."""
    matrix = np.asarray(mass_squared_matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("Expected a 2x2 mass-squared mixing matrix")
    return np.linalg.eigvalsh(0.5*(matrix+matrix.T))
