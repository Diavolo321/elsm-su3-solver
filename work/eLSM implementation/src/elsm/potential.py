"""Thermodynamic potential pieces for the SU(3) eLSM.

Units: GeV. The zero-point quark contribution is deliberately separate because
Eq. (27) of arXiv:2512.22198v2 is dimensionally ambiguous as printed and gives
no regulator/subtraction prescription. Do not interpret the default thermal-only
potential as a fully audited reproduction of that equation.
"""
import numpy as np
from scipy.integrate import simpson
from .parameters import SU3Parameters, PolyakovParameters


def mesonic_potential(sigma_x, sigma_y, p: SU3Parameters, T=0.0):
    """Tree-level SU(3) potential, Eq. (14), in GeV^4."""
    sx, sy = float(sigma_x), float(sigma_y)
    m = p.effective_m0_sq(float(T))
    return (
        0.5*m*(sx*sx + sy*sy)
        - p.c_anomaly/(2*np.sqrt(2))*sx*sx*sy
        + 0.5*p.lambda1*sx*sx*sy*sy
        + (2*p.lambda1+p.lambda2)/8*sx**4
        + (p.lambda1+p.lambda2)/4*sy**4
        - p.h_x*sx - p.h_y*sy
    )


def mesonic_gradient_analytic(sigma_x, sigma_y, p: SU3Parameters, T=0.0):
    """Analytic gradient of Eq. (14), ordered (d/d sigma_x, d/d sigma_y)."""
    sx, sy = float(sigma_x), float(sigma_y)
    m = p.effective_m0_sq(float(T))
    dx = (m*sx - p.c_anomaly/np.sqrt(2)*sx*sy
          + p.lambda1*sx*sy**2
          + (2*p.lambda1+p.lambda2)/2*sx**3 - p.h_x)
    dy = (m*sy - p.c_anomaly/(2*np.sqrt(2))*sx**2
          + p.lambda1*sx**2*sy
          + (p.lambda1+p.lambda2)*sy**3 - p.h_y)
    return np.array([dx, dy], dtype=float)


def mesonic_hessian(sigma_x, sigma_y, p: SU3Parameters, T=0.0):
    """Hessian of the scalar mean-field potential with respect to sigma_x,y."""
    sx, sy = float(sigma_x), float(sigma_y)
    m = p.effective_m0_sq(float(T))
    hxx = (m - p.c_anomaly/np.sqrt(2)*sy + p.lambda1*sy**2
           + 1.5*(2*p.lambda1+p.lambda2)*sx**2)
    hyy = (m + p.lambda1*sx**2 + 3*(p.lambda1+p.lambda2)*sy**2)
    hxy = (-p.c_anomaly/np.sqrt(2)*sx + 2*p.lambda1*sx*sy)
    return np.array([[hxx, hxy], [hxy, hyy]], dtype=float)


def polyakov_log_argument(phi, phi_bar):
    phi, phi_bar = float(phi), float(phi_bar)
    return 1 - 6*phi*phi_bar + 4*(phi**3 + phi_bar**3) - 3*(phi*phi_bar)**2


def polyakov_log_potential(phi, phi_bar, T, p: PolyakovParameters):
    """Standard logarithmic Polyakov potential, in GeV^4.

    U/T^4 = -a(T) phi phi_bar/2 + b(T) log(Haar(phi,phi_bar)).
    The physical real mean-field domain is normally 0 <= phi,phi_bar < 1;
    H<=0 is outside the real logarithm domain and returns +infinity.
    """
    T = float(T)
    if T < 0:
        raise ValueError("Temperature T must be non-negative")
    if T == 0:
        # The T^4-scaled logarithmic potential tends to zero at finite fields.
        return 0.0
    p.validate()
    if not (0 <= phi < 1 and 0 <= phi_bar < 1):
        return float("inf")
    x = p.T0/T
    aT = p.a0 + p.a1*x + p.a2*x*x
    bT = p.b3*x**3
    arg = polyakov_log_argument(phi, phi_bar)
    if arg <= 0:
        return float("inf")
    return float(T**4*(-0.5*aT*phi*phi_bar + bT*np.log(arg)))


def quark_masses(sigma_x, sigma_y, g_yukawa=5.5):
    """Return (m_u,m_d,m_s) in GeV, using the paper's Eq. (29) convention."""
    mx = g_yukawa*float(sigma_x)/2
    ms = g_yukawa*float(sigma_y)/np.sqrt(2)
    return np.array([mx, mx, ms], dtype=float)


def _polyakov_logs(E, T, phi, phi_bar, mu):
    """Stable-enough log g+ and log g- for the real mean-field case."""
    xp = np.exp(np.clip(-(E-mu)/T, -745, 50))
    xm = np.exp(np.clip(-(E+mu)/T, -745, 50))
    gp = 1 + 3*(phi + phi_bar*xp)*xp + xp**3
    gm = 1 + 3*(phi_bar + phi*xm)*xm + xm**3
    if np.any(gp <= 0) or np.any(gm <= 0):
        raise ValueError("Quark logarithm left its real domain; check phi, phi_bar and mu")
    return np.log(gp), np.log(gm)


def quark_thermal_potential(T, sigma_x, sigma_y, phi, phi_bar,
                            g_yukawa=5.5, mu=0.0, n_c=3,
                            p_max=None, n_points=1000, thermal_normalization=1.0):
    """Thermal logarithm contribution to Omega_qbarq, in GeV^4.

    Implements the thermal logarithms of Eq. (27). The divergent zero-point
    integral is not included; see vacuum_quark_potential() and README.
    """
    T = float(T)
    if T < 0:
        raise ValueError("Temperature T must be non-negative")
    if T == 0:
        return 0.0
    if n_points < 20:
        raise ValueError("n_points must be >= 20")
    if not np.isfinite(thermal_normalization) or thermal_normalization <= 0:
        raise ValueError("thermal_normalization must be positive and finite")
    if not (0 <= phi < 1 and 0 <= phi_bar < 1):
        raise ValueError("Require 0 <= phi,phi_bar < 1 for real mean-field calculation")
    masses = quark_masses(sigma_x, sigma_y, g_yukawa)
    if p_max is None:
        p_max = max(5.0, 25*T + 2*float(np.max(np.abs(masses))))
    mom = np.linspace(0.0, float(p_max), int(n_points))
    total = 0.0
    for mass in masses:
        E = np.sqrt(mom*mom + mass*mass)
        lp, lm = _polyakov_logs(E, T, phi, phi_bar, mu)
        integral = simpson(mom*mom*(lp+lm), x=mom)/(2*np.pi**2)
        # The Polyakov-modified logarithm already contains the color trace
        # (the 3*phi and 3*phi_bar terms). Only the two spin states remain
        # outside the logarithm; multiplying by N_c here double-counts color.
        total += -2*thermal_normalization*T*integral
    return float(total)


def vacuum_quark_potential(sigma_x, sigma_y, g_yukawa=5.5, n_c=3,
                           cutoff=None, subtract_massless=True, n_points=1600):
    """Sharp-three-momentum-cutoff zero-point term, with explicit convention.

    This is a finite-cutoff model prescription, NOT asserted to be the one used
    in the paper. With subtract_massless=True, subtracts the massless reference
    integral at the same cutoff. A caller must pass cutoff explicitly.
    """
    if cutoff is None or cutoff <= 0:
        raise ValueError("A positive explicit cutoff (GeV) is required")
    if n_points < 20:
        raise ValueError("n_points must be >= 20")
    masses = quark_masses(sigma_x, sigma_y, g_yukawa)
    mom = np.linspace(0.0, float(cutoff), int(n_points))
    total = 0.0
    for mass in masses:
        e = np.sqrt(mom*mom + mass*mass)
        if subtract_massless:
            e = e - mom
        # Standard spin-colour zero-point degeneracy -2*N_c, not an assumed
        # transcription of the ambiguous prefactor printed in Eq. (27).
        total += -2*n_c*simpson(mom*mom*e, x=mom)/(2*np.pi**2)
    return float(total)


def vacuum_quark_potential_unsubtracted(sigma_x, sigma_y, g_yukawa=5.5, n_c=3,
                                         cutoff=None, n_points=1600):
    """Sharp-cutoff zero-point integral without reference subtraction.

    This is one conventional fermion zero-point prescription, not a claim about
    Eq. (27) of arXiv:2512.22198v2. The cutoff is explicit because the integral
    is UV divergent as the cutoff is removed. Units: GeV^4.
    """
    if cutoff is None or cutoff <= 0:
        raise ValueError("A positive explicit cutoff (GeV) is required")
    if n_points < 20:
        raise ValueError("n_points must be >= 20")
    masses = quark_masses(sigma_x, sigma_y, g_yukawa)
    mom = np.linspace(0.0, float(cutoff), int(n_points))
    total = 0.0
    for mass in masses:
        energy = np.sqrt(mom*mom + mass*mass)
        total += -2*n_c*simpson(mom*mom*energy, x=mom)/(2*np.pi**2)
    return float(total)


def grand_potential(T, sigma_x, sigma_y, phi, phi_bar, p,
                    polyakov_parameters=None, mu=0.0,
                    vacuum_prescription="omit", vacuum_cutoff=None,
                    momentum_p_max=None, momentum_n_points=1000,
                    thermal_normalization=1.0):
    """Full mean-field Omega under an explicitly chosen vacuum prescription.

    vacuum_prescription: 'omit', 'sharp_cutoff_subtracted', or
    'sharp_cutoff_unsubtracted'. 'omit' is thermal-only; the two cutoff choices
    are explicit alternative model prescriptions, not asserted source conventions.
    """
    if polyakov_parameters is None:
        raise ValueError("Supply explicit PolyakovParameters before minimization")
    omega = (mesonic_potential(sigma_x, sigma_y, p, T)
             + polyakov_log_potential(phi, phi_bar, T, polyakov_parameters)
             + quark_thermal_potential(T, sigma_x, sigma_y, phi, phi_bar,
                                      p.g_yukawa, mu, p_max=momentum_p_max, n_points=momentum_n_points,
                                      thermal_normalization=thermal_normalization))
    if vacuum_prescription == "omit":
        return float(omega)
    if vacuum_prescription == "sharp_cutoff_subtracted":
        return float(omega + vacuum_quark_potential(
            sigma_x, sigma_y, p.g_yukawa, cutoff=vacuum_cutoff,
            subtract_massless=True))
    if vacuum_prescription == "sharp_cutoff_unsubtracted":
        return float(omega + vacuum_quark_potential_unsubtracted(
            sigma_x, sigma_y, p.g_yukawa, cutoff=vacuum_cutoff))
    raise ValueError("Unknown vacuum_prescription; use 'omit', 'sharp_cutoff_subtracted', or 'sharp_cutoff_unsubtracted'")
