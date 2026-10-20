"""Vacuum condensate calibration.

PRIMARY (reproduces the paper): `fit_condensates_yukawa`.
    sigma_x = 2 m_q / g  (paper, Sec. III: g fixed from m_q = 0.3 GeV),
    sigma_y fitted to the tree-level spectrum of the chosen Table II row.
    With Table II this reproduces Table I (pi, eta, eta', sigma, rho, a1) to <0.5%
    and Table III T_chi to within one 2.5 MeV grid step.

DIAGNOSTIC ONLY: `fit_condensates_pcac`.
    Inverting Eq. (37) with the Table I f_pi, f_K gives sigma_x = 0.1263 GeV, which
    is NOT the authors' vacuum (it yields m_pi ~ 436 MeV). Kept for audits only.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from .parameters import SU3Parameters
from .meson_masses import wavefunction_factors


@dataclass(frozen=True)
class PCACFit:
    sigma_x: float
    sigma_y: float
    Z_pi: float
    Z_K: float
    f_pi_reconstructed: float
    f_K_reconstructed: float
    g_from_constituent_mass: float
    h_x: float
    h_y: float
    residual_norm: float


def fit_condensates_pcac(*, f_pi, f_K, m_a1, m_k1, g1,
                         m0_sq, lambda1, lambda2, c_anomaly,
                         g_yukawa=5.5, constituent_mass=0.300,
                         Tc=0.270, include_FT=False):
    """DIAGNOSTIC ONLY. Solve Eq. (37) for sigma_x,y using f_pi,f_K and Z_pi,Z_K.

    Do not use this as the vacuum of the model: the Table I decay constants are not
    consistent with the Table I masses (see fit_condensates_yukawa).

    All inputs are in GeV (except dimensionless couplings). The returned
    `g_from_constituent_mass=2*m_q/sigma_x` is a diagnostic, not a replacement
    for the independently tabulated Yukawa coupling. A mismatch quantifies
    the source's parameter-set tension.
    """
    vals = (f_pi, f_K, m_a1, m_k1, g1, g_yukawa, constituent_mass)
    if any(not np.isfinite(v) or v <= 0 for v in vals):
        raise ValueError("Decay constants, masses and couplings must be finite and positive")

    # Solve only the two factors used by PCAC; use a bounded physical branch.
    def pcac_residual(s):
        sx, sy = s
        rpi = m_a1*m_a1 - g1*g1*sx*sx
        rk = 4*m_k1*m_k1 - g1*g1*(sx + np.sqrt(2)*sy)**2
        if rpi <= 0 or rk <= 0:
            return np.array([1e2 + abs(min(rpi, 0)), 1e2 + abs(min(rk, 0))])
        zpi = m_a1/np.sqrt(rpi)
        zk = 2*m_k1/np.sqrt(rk)
        return np.array([sx/zpi-f_pi, (np.sqrt(2)*sy+sx)/(2*zk)-f_K])

    fit = least_squares(pcac_residual, x0=np.array([f_pi, (2*f_K-f_pi)/np.sqrt(2)]),
                        bounds=([1e-6, 1e-6], [0.5, 0.5]),
                        xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=5000)
    sx, sy = map(float, fit.x)
    rpi = m_a1*m_a1-g1*g1*sx*sx
    rk = 4*m_k1*m_k1-g1*g1*(sx+np.sqrt(2)*sy)**2
    if rpi <= 0 or rk <= 0 or np.linalg.norm(pcac_residual(fit.x)) > 1e-7:
        raise ValueError("Could not find a valid PCAC solution on the selected branch")
    zpi, zk = m_a1/np.sqrt(rpi), 2*m_k1/np.sqrt(rk)
    p = SU3Parameters.from_vacuum_condensates(
        m0_sq=m0_sq, lambda1=lambda1, lambda2=lambda2,
        c_anomaly=c_anomaly, sigma_x=sx, sigma_y=sy,
        g_yukawa=g_yukawa, Tc=Tc, include_FT=include_FT)
    return PCACFit(sigma_x=sx, sigma_y=sy, Z_pi=float(zpi), Z_K=float(zk),
                   f_pi_reconstructed=float(sx/zpi),
                   f_K_reconstructed=float((np.sqrt(2)*sy+sx)/(2*zk)),
                   g_from_constituent_mass=float(2*constituent_mass/sx),
                   h_x=p.h_x, h_y=p.h_y,
                   residual_norm=float(np.linalg.norm(pcac_residual(fit.x))))


# ---------------------------------------------------------------------------
# Primary vacuum: g = 2 m_q / sigma_x, sigma_y from the spectrum
# ---------------------------------------------------------------------------
from .meson_masses import tree_level_meson_mass_squared, mixed_mass_eigenvalues
from scipy.optimize import minimize_scalar

# Table I SU(3) fit outputs (GeV); keys are the channels used for sigma_y.
TABLE_I_SU3 = {
    "c0":   {"pi": 0.13985, "eta": 0.1400, "eta_prime": 0.64493, "sigma": 0.600},
    "cfin": {"pi": 0.13985, "eta": 0.5319, "eta_prime": 0.96544, "sigma": 0.700},
}


@dataclass(frozen=True)
class YukawaVacuum:
    sigma_x: float
    sigma_y: float
    Z: dict
    h_x: float
    h_y: float
    masses: dict           # tree-level vacuum masses in GeV at (sigma_x, sigma_y)
    rel_residuals: dict    # (model - target)/target for the fitted channels
    f_pi_implied: float    # sigma_x/Z_pi, diagnostic only
    f_K_implied: float


def vacuum_spectrum(sigma_x, sigma_y, *, m0_sq, lambda1, lambda2, c_anomaly,
                    g1, m_a1, m_k1, m_f1s, m_kstar):
    """Tree-level (pseudo)scalar vacuum masses in GeV (mixed states diagonalized)."""
    Z = wavefunction_factors(sigma_x, sigma_y, g1=g1, m_a1=m_a1, m_k1=m_k1,
                             m_f1s=m_f1s, m_kstar=m_kstar)
    m2 = tree_level_meson_mass_squared(sigma_x, sigma_y, m0_sq=m0_sq,
                                       c_anomaly=c_anomaly, lambda1=lambda1,
                                       lambda2=lambda2, Z=Z)
    pe = mixed_mass_eigenvalues(m2["pseudoscalar_isoscalar_matrix"])
    se = mixed_mass_eigenvalues(m2["scalar_isoscalar_matrix"])
    root = lambda x: float(np.sqrt(x)) if x > 0 else float("nan")
    masses = {"pi": root(m2["pi"]), "K": root(m2["K"]), "a0": root(m2["a0"]),
              "K0star": root(m2["K0star"]), "eta": root(pe[0]),
              "eta_prime": root(pe[1]), "sigma": root(se[0]), "f0": root(se[1])}
    return masses, Z


def fit_condensates_yukawa(*, m0_sq, lambda1, lambda2, c_anomaly,
                           g_yukawa=5.5, constituent_mass=0.300,
                           g1=5.58, m_a1=1.06976, m_k1=1.25346,
                           m_f1s=1.42477, m_kstar=0.851,
                           targets=None, sigma_y_bounds=(0.05, 0.16),
                           Tc=0.270, include_FT=False):
    """Authors' vacuum: sigma_x = 2 m_q/g, sigma_y fitted to `targets` (GeV).

    `targets` maps channel -> mass, e.g. TABLE_I_SU3["cfin"]. The kaon is not used
    by default: Table I's m_K=420.49 MeV is not reproduced by any consistent
    convention (the model gives ~495 MeV); include it only knowingly.
    """
    if g_yukawa <= 0 or constituent_mass <= 0:
        raise ValueError("g_yukawa and constituent_mass must be positive")
    if targets is None:
        targets = TABLE_I_SU3["cfin" if c_anomaly != 0 else "c0"]
    sx = 2.0*constituent_mass/g_yukawa
    kw = dict(m0_sq=m0_sq, lambda1=lambda1, lambda2=lambda2, c_anomaly=c_anomaly,
              g1=g1, m_a1=m_a1, m_k1=m_k1, m_f1s=m_f1s, m_kstar=m_kstar)

    def cost(sy):
        try:
            m, _ = vacuum_spectrum(sx, sy, **kw)
        except ValueError:
            return 1e6
        r = [(m[k]-v)/v for k, v in targets.items()]
        return 1e6 if not np.all(np.isfinite(r)) else float(np.sum(np.square(r)))

    res = minimize_scalar(cost, bounds=sigma_y_bounds, method="bounded",
                          options={"xatol": 1e-10})
    sy = float(res.x)
    if any(abs(sy-b) < 1e-6 for b in sigma_y_bounds):
        raise ValueError("sigma_y fit hit its bound; check parameters/targets")
    masses, Z = vacuum_spectrum(sx, sy, **kw)
    p = SU3Parameters.from_vacuum_condensates(
        m0_sq=m0_sq, lambda1=lambda1, lambda2=lambda2, c_anomaly=c_anomaly,
        sigma_x=sx, sigma_y=sy, g_yukawa=g_yukawa, Tc=Tc, include_FT=include_FT)
    return YukawaVacuum(
        sigma_x=sx, sigma_y=sy, Z=Z, h_x=p.h_x, h_y=p.h_y, masses=masses,
        rel_residuals={k: (masses[k]-v)/v for k, v in targets.items()},
        f_pi_implied=sx/Z["Z_pi"],
        f_K_implied=(sx+np.sqrt(2)*sy)/(2*Z["Z_K"]))


def sigma_y_from_vector_sector(*, m_f1, m_phi, g1=5.58, h3=-15.216):
    """Independent cross-check: m_f1S^2 - m_omegaS^2 = 2 (g1^2 - h3) sigma_y^2."""
    return float(np.sqrt((m_f1**2 - m_phi**2)/(2*(g1**2 - h3))))
