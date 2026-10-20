import numpy as np
from elsm.thermal_masses import (polyakov_occupations, thermal_mass_matrix_correction,
                                 corrected_mass_matrix, mass_eigenvalues)


def test_occupations_are_finite_and_nonnegative_for_mu_zero():
    nplus, nminus, bplus, bminus = polyakov_occupations(
        np.array([0.2, 0.5, 1.0]), 0.16, 0.2, 0.2)
    assert np.all(np.isfinite(nplus + nminus + bplus + bminus))
    assert np.all(nplus >= 0) and np.all(nminus >= 0)


def test_thermal_mass_correction_shape_and_symmetry():
    # Simple synthetic derivatives test the integral machinery, not a specific meson channel.
    d1 = np.zeros((3, 2)); d1[:, 0] = 0.1
    d2 = np.zeros((3, 2, 2)); d2[:, 0, 0] = 1.0
    corr = thermal_mass_matrix_correction(0.15, 0.1, 0.14, 0.2, 0.2,
                                          d1, d2, n_points=120)
    assert corr.shape == (2, 2)
    assert np.allclose(corr, corr.T)
    assert np.all(np.isfinite(corr))


def test_corrected_matrix_eigenvalues():
    out = corrected_mass_matrix(np.diag([0.2, 0.4]), np.diag([0.01, 0.02]), [1.1, 1.2])
    assert np.allclose(mass_eigenvalues(out), np.linalg.eigvalsh(out))


def test_vacuum_quark_prescriptions_are_explicit_and_finite():
    from elsm.potential import vacuum_quark_potential, vacuum_quark_potential_unsubtracted
    sub = vacuum_quark_potential(0.11, 0.14, cutoff=1.0, n_points=300)
    raw = vacuum_quark_potential_unsubtracted(0.11, 0.14, cutoff=1.0, n_points=300)
    assert np.isfinite(sub) and np.isfinite(raw)
    assert raw < sub  # subtracting the massless reference makes the term less negative
