"""Finite-temperature quark-loop correction to meson mass matrices (Eq. 32).

The correction is generic in a meson channel: callers supply derivatives of the
squared quark masses with respect to that channel's fields. This avoids silently
inventing channel-specific derivatives not listed in the main article for SU(3).
All quantities are in GeV units.
"""
import numpy as np
from scipy.integrate import simpson
from .potential import quark_masses


def polyakov_occupations(E, T, phi, phi_bar, mu=0.0):
    """Return n_plus, n_minus, b_plus, b_minus as in Eqs. (33)–(36)."""
    if T <= 0:
        raise ValueError("Thermal occupations require T > 0")
    E = np.asarray(E, dtype=float)
    xp = np.exp(np.clip(-(E-mu)/T, -745, 50))
    xm = np.exp(np.clip(-(E+mu)/T, -745, 50))
    gp = 1 + 3*(phi + phi_bar*xp)*xp + xp**3
    gm = 1 + 3*(phi_bar + phi*xm)*xm + xm**3
    if np.any(gp <= 0) or np.any(gm <= 0):
        raise ValueError("Polyakov distribution denominator is non-positive")
    np_ = (phi*xp + 2*phi_bar*xp**2 + xp**3)/gp
    nm_ = (phi_bar*xm + 2*phi*xm**2 + xm**3)/gm
    cp = (phi*xp + 4*phi_bar*xp**2 + 3*xp**3)/gp
    cm = (phi_bar*xm + 4*phi*xm**2 + 3*xm**3)/gm
    bp = 3*np_**2 - cp
    bm = 3*nm_**2 - cm
    return np_, nm_, bp, bm


def thermal_mass_matrix_correction(T, sigma_x, sigma_y, phi, phi_bar,
                                   d_m2_d_fields, d2_m2_d_fields2,
                                   g_yukawa=5.5, mu=0.0, n_c=3,
                                   p_max=None, n_points=1000):
    """Evaluate the thermal correction matrix from Eq. (32).

    Parameters
    ----------
    d_m2_d_fields : array (n_flavours, n_fields)
        First derivatives d(m_f^2)/d(phi_a), dimensions GeV.
    d2_m2_d_fields2 : array (n_flavours, n_fields, n_fields)
        Second derivatives d^2(m_f^2)/(d(phi_a)d(phi_b)), dimensionless.
    Returns
    -------
    ndarray (n_fields,n_fields), in GeV^2.

    This implements the structure of Eq. (32); source-specific field derivatives
    must be supplied for each meson channel. Flavour order is (u,d,s).
    """
    T = float(T)
    if T <= 0:
        n_fields = np.asarray(d_m2_d_fields).shape[-1]
        return np.zeros((n_fields, n_fields), dtype=float)
    first = np.asarray(d_m2_d_fields, dtype=float)
    second = np.asarray(d2_m2_d_fields2, dtype=float)
    if first.ndim != 2 or first.shape[0] != 3:
        raise ValueError("d_m2_d_fields must have shape (3,n_fields), flavours (u,d,s)")
    nf, n_fields = first.shape
    if second.shape != (nf, n_fields, n_fields):
        raise ValueError("d2_m2_d_fields2 must have shape (3,n_fields,n_fields)")
    if not (0 <= phi < 1 and 0 <= phi_bar < 1):
        raise ValueError("Require 0 <= phi,phi_bar < 1")
    masses = quark_masses(sigma_x, sigma_y, g_yukawa)
    if p_max is None:
        p_max = max(5.0, 25*T + 2*float(np.max(np.abs(masses))))
    mom = np.linspace(0.0, float(p_max), int(n_points))
    result = np.zeros((n_fields, n_fields), dtype=float)
    for f, mass in enumerate(masses):
        E = np.sqrt(mom*mom + mass*mass)
        np_, nm_, bp, bm = polyakov_occupations(E, T, phi, phi_bar, mu)
        nsum = np_ + nm_
        bsum = bp + bm
        # Eq. (32): nu_c * integral dp/(2pi)^3 * 1/(2E) times bracket.
        # The angular integral is p^2 dp/(2 pi^2), so combined prefactor is
        # nu_c * p^2 dp/(4 pi^2 E).
        e_safe = np.maximum(E, np.finfo(float).tiny)
        for a in range(n_fields):
            for b in range(n_fields):
                bracket = (nsum*second[f, a, b]
                           - nsum*first[f, a]*first[f, b]/(2*e_safe**2)
                           + bsum*first[f, a]*first[f, b]/(2*e_safe*T))
                integrand = mom*mom*bracket/(2*e_safe)
                result[a, b] += (2*n_c)*simpson(integrand, x=mom)/(2*np.pi**2)
    return result


def corrected_mass_matrix(vacuum_mass_matrix, thermal_correction, wavefunction_factors=None):
    """Add Eq. (32) to a curvature mass-squared matrix and optionally renormalize.

    `wavefunction_factors` may be a scalar or a diagonal vector Z_i; this helper
    applies Z_i Z_j to the complete matrix. Supply factors only when the source's
    channel-specific convention requires it.
    """
    m0 = np.asarray(vacuum_mass_matrix, dtype=float)
    dt = np.asarray(thermal_correction, dtype=float)
    if m0.shape != dt.shape or m0.ndim != 2 or m0.shape[0] != m0.shape[1]:
        raise ValueError("Vacuum and thermal matrices must be same square shape")
    out = m0 + dt
    if wavefunction_factors is not None:
        z = np.asarray(wavefunction_factors, dtype=float)
        if z.ndim == 0:
            out = z*z*out
        elif z.shape == (out.shape[0],):
            out = z[:, None]*out*z[None, :]
        else:
            raise ValueError("wavefunction_factors must be scalar or vector of matrix dimension")
    return 0.5*(out + out.T)


def mass_eigenvalues(mass_squared_matrix, *, require_positive=False):
    """Return sorted eigenvalues of a symmetric mass-squared matrix (GeV^2)."""
    m2 = np.asarray(mass_squared_matrix, dtype=float)
    if m2.ndim != 2 or m2.shape[0] != m2.shape[1]:
        raise ValueError("Input must be square")
    vals = np.linalg.eigvalsh(0.5*(m2+m2.T))
    if require_positive and np.any(vals <= 0):
        raise ValueError("Mass-squared matrix has non-positive eigenvalues")
    return vals


def sigma_field_mass_derivatives(sigma_x, sigma_y, g_yukawa=5.5):
    """Return derivatives for mean fields (sigma_x,sigma_y) in flavour order u,d,s.

    For m_u=m_d=g*sigma_x/2 and m_s=g*sigma_y/sqrt(2), returns arrays
    suitable for `thermal_mass_matrix_correction`. This covers the scalar
    condensate-field directions only; individual meson channels require their
    own derivatives.
    """
    sx, sy, g = float(sigma_x), float(sigma_y), float(g_yukawa)
    first = np.zeros((3, 2), dtype=float)
    second = np.zeros((3, 2, 2), dtype=float)
    first[0, 0] = first[1, 0] = g*g*sx/2
    first[2, 1] = g*g*sy
    second[0, 0, 0] = second[1, 0, 0] = g*g/2
    second[2, 1, 1] = g*g
    return first, second
