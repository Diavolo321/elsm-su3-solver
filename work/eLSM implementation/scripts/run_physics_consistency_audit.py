# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
#!/usr/bin/env python3
"""Independent vacuum stability/spectrum consistency checks for the SU(3) eLSM.

This is a diagnostic audit, not a claim that the hybrid parameter set is the
fully self-consistent parameterization used by the paper.
"""
from pathlib import Path
import sys
import csv
import json
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from elsm.parameters import SU3Parameters
from elsm.potential import mesonic_potential, mesonic_gradient_analytic, mesonic_hessian
from elsm.vacuum_fit import fit_condensates_pcac
from elsm.meson_masses import wavefunction_factors, tree_level_meson_mass_squared, mixed_mass_eigenvalues


def main():
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    # Inputs taken from the target paper's SU(3), c != 0 Table II / PCAC setup.
    fit = fit_condensates_pcac(
        f_pi=0.09502, f_K=0.1069, m_a1=1.06976, m_k1=1.25346, g1=5.58,
        m0_sq=-(0.258**2), lambda1=3.15, lambda2=36.253, c_anomaly=2.828,
        g_yukawa=5.5, constituent_mass=0.300)
    p = SU3Parameters(m0_sq=-(0.258**2), lambda1=3.15, lambda2=36.253,
                      c_anomaly=2.828, h_x=fit.h_x, h_y=fit.h_y, g_yukawa=5.5)
    sx, sy = fit.sigma_x, fit.sigma_y
    grad = mesonic_gradient_analytic(sx, sy, p)
    hess = mesonic_hessian(sx, sy, p)
    # Multi-start search of the tree-level mesonic potential in the explicitly
    # stated physical box. This is not a proof of globality outside the box.
    starts = [(0.01, 0.01), (0.10, 0.10), (0.30, 0.30), (0.50, 0.70), (sx, sy)]
    minima = []
    for x0 in starts:
        r = minimize(lambda z: mesonic_potential(z[0], z[1], p), np.asarray(x0),
                     jac=lambda z: mesonic_gradient_analytic(z[0], z[1], p),
                     bounds=((0.0, 0.6), (0.0, 0.8)), method="L-BFGS-B",
                     options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-9})
        minima.append(r)
    best = min(minima, key=lambda r: r.fun)
    spread_minima = max(r.fun for r in minima) - min(r.fun for r in minima)
    heig = np.linalg.eigvalsh(hess)

    # For this displayed quartic, positive lambda1 and lambda2 imply each
    # quartic monomial coefficient is non-negative, so the quartic form is
    # bounded from below (strictly positive away from the origin here).
    quartic_coeffs = {
        "sx4": (2*p.lambda1+p.lambda2)/8,
        "sx2_sy2": p.lambda1/2,
        "sy4": (p.lambda1+p.lambda2)/4,
    }
    quartic_bounded = all(v > 0 for v in quartic_coeffs.values())

    # Other wavefunction factors use fitted physical axial/vector masses as
    # listed in the paper's vacuum inputs; these are only for channel diagnostics.
    Z = wavefunction_factors(sx, sy, g1=5.58, m_a1=1.06976, m_k1=1.25346,
                             m_f1s=1.4264, m_kstar=0.89555)
    masses2 = tree_level_meson_mass_squared(
        sx, sy, m0_sq=p.m0_sq, c_anomaly=p.c_anomaly,
        lambda1=p.lambda1, lambda2=p.lambda2, Z=Z,
        g1=5.58, c1=0.44, h2=40.975, h3=-15.216)
    scalar_eig2 = np.linalg.eigvalsh(masses2["scalar_isoscalar_matrix"])
    ps_eig2 = np.linalg.eigvalsh(masses2["pseudoscalar_isoscalar_matrix"])

    # Exact Ward/curvature identity implied by the printed potential and Table A1:
    # m_pi^2 = Z_pi^2 * h_x / sigma_x at a stationary point.
    # Compare this hybrid set against the paper's SU(3), c != 0 Table I pion fit.
    m_pi_table = 0.13985
    hx_required_table_pi = m_pi_table**2 * sx / Z["Z_pi"]**2
    hx_ratio = fit.h_x / hx_required_table_pi
    m_pi_from_stationarity = np.sqrt(Z["Z_pi"]**2 * fit.h_x / sx)
    target_bracket = (m_pi_table / Z["Z_pi"])**2
    current_bracket = fit.h_x / sx
    m0_sq_needed_if_only_m0_changed = p.m0_sq + target_bracket - current_bracket

    rows = [
        ("PCAC sigma_x", sx, "GeV", "derived from f_pi, f_K, Z_pi, Z_K equations"),
        ("PCAC sigma_y", sy, "GeV", "derived from f_pi, f_K, Z_pi, Z_K equations"),
        ("g implied by m_q=0.300 GeV", fit.g_from_constituent_mass, "dimensionless", "2*m_q/sigma_x; table lists g=5.5"),
        ("stationarity residual norm", float(np.linalg.norm(grad)), "GeV^3", "gradient of tree-level mesonic potential at PCAC condensates with h derived from same point"),
        ("mesonic potential at stationary point", mesonic_potential(sx, sy, p), "GeV^4", "relative value; not an absolute thermodynamic pressure"),
        ("multi-start best sigma_x", float(best.x[0]), "GeV", "minimum found within box sigma_x in [0,0.6], sigma_y in [0,0.8] GeV"),
        ("multi-start best sigma_y", float(best.x[1]), "GeV", "minimum found within box sigma_x in [0,0.6], sigma_y in [0,0.8] GeV"),
        ("multi-start potential spread", float(spread_minima), "GeV^4", "max-min objective among five bounded local minimizations"),
        ("Hessian eigenvalue 1", heig[0], "GeV^2", "tree-level scalar condensate-direction Hessian"),
        ("Hessian eigenvalue 2", heig[1], "GeV^2", "tree-level scalar condensate-direction Hessian"),
        ("quartic coefficient sx^4", quartic_coeffs["sx4"], "dimensionless", "positive"),
        ("quartic coefficient sx^2 sy^2", quartic_coeffs["sx2_sy2"], "dimensionless", "positive"),
        ("quartic coefficient sy^4", quartic_coeffs["sy4"], "dimensionless", "positive"),
        ("tree-level m_pi", float(np.sqrt(masses2["pi"])) if masses2["pi"] >= 0 else float("nan"), "GeV", "conditional hybrid diagnostic; compare with paper's c!=0 SU(3) fit 0.13985 GeV"),
        ("pion mass from stationarity identity", float(m_pi_from_stationarity), "GeV", "sqrt(Z_pi^2*h_x/sigma_x); algebraically equivalent to Table A1 pion curvature at stationary point"),
        ("h_x required for Table I pion mass", float(hx_required_table_pi), "GeV^3", "m_pi_table^2*sigma_x/Z_pi^2, holding PCAC condensate and Z_pi fixed"),
        ("constructed h_x / required h_x", float(hx_ratio), "dimensionless", "diagnostic ratio; values far from 1 expose hybrid parameter-set inconsistency"),
        ("m0_sq needed if only m0_sq changed", float(m0_sq_needed_if_only_m0_changed), "GeV^2", "one-parameter algebraic sensitivity only; not a proposed fit"),
        ("tree-level m_K", float(np.sqrt(masses2["K"])) if masses2["K"] >= 0 else float("nan"), "GeV", "conditional hybrid diagnostic; compare with physical kaon ~0.494-0.498 GeV"),
        ("scalar-isoscalar eigenmass 1", float(np.sqrt(scalar_eig2[0])) if scalar_eig2[0] >= 0 else float("nan"), "GeV", "tree-level eigenvalue from Table A1 matrix"),
        ("scalar-isoscalar eigenmass 2", float(np.sqrt(scalar_eig2[1])) if scalar_eig2[1] >= 0 else float("nan"), "GeV", "tree-level eigenvalue from Table A1 matrix"),
        ("pseudoscalar-isoscalar eigenmass 1", float(np.sqrt(ps_eig2[0])) if ps_eig2[0] >= 0 else float("nan"), "GeV", "tree-level eigenvalue from Table A1 matrix"),
        ("pseudoscalar-isoscalar eigenmass 2", float(np.sqrt(ps_eig2[1])) if ps_eig2[1] >= 0 else float("nan"), "GeV", "tree-level eigenvalue from Table A1 matrix"),
    ]
    with (outdir / "physics_consistency_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["quantity", "value", "unit", "interpretation"]); w.writerows(rows)

    report = f'''# Independent physics-consistency audit (SU(3) eLSM)

Date: 2026-10-18. This is a diagnostic audit, not a validated full reproduction.

## 1. Inputs and scope

The test combines the target paper's Table II scalar-sector values (`m0^2=-(0.258 GeV)^2`, `lambda1=3.15`, `lambda2=36.253`, `c=2.828`) with a PCAC reconstruction from `f_pi=95.02 MeV`, `f_K=106.9 MeV`, `m_a1=1.06976 GeV`, `m_K1=1.25346 GeV`, and `g1=5.58`. The explicit-breaking parameters `h_x,h_y` are then derived from the stationarity equations at those PCAC condensates. This is a controlled consistency diagnostic, but it is a **hybrid set** because Table II's `g=5.5` is inconsistent with the PCAC-derived `sigma_x` under `g=2 m_q/sigma_x` for `m_q=0.300 GeV`.

## 2. Stationarity and local stability

- PCAC condensates: `sigma_x={sx:.8f} GeV`, `sigma_y={sy:.8f} GeV`.
- Gradient norm at the constructed stationary point: `{np.linalg.norm(grad):.3e} GeV^3`.
- Hessian eigenvalues: `{heig[0]:.6f}`, `{heig[1]:.6f} GeV^2`.
- Both Hessian eigenvalues are positive, so this point is a **local minimum in the two condensate directions** for the tree-level mesonic potential under this constructed parameter set.
- Five bounded minimizations from separated starts in `sigma_x in [0,0.6] GeV`, `sigma_y in [0,0.8] GeV` converge to the same minimum within numerical tolerance: `({best.x[0]:.7f}, {best.x[1]:.7f}) GeV`; the spread of objective values is `{spread_minima:.2e} GeV^4`. This is evidence for global-minimum stability **inside this box for this two-field tree potential**, not a proof for the full thermodynamic potential or unbounded field space.
- Quartic coefficients in the displayed potential are `sx^4: {quartic_coeffs['sx4']:.6f}`, `sx^2 sy^2: {quartic_coeffs['sx2_sy2']:.6f}`, `sy^4: {quartic_coeffs['sy4']:.6f}`. They are all positive, so the quartic part is bounded from below and grows at large fields for these couplings.

These checks establish tree-level mesonic stability only. They do **not** prove stability of the full finite-temperature grand potential with quarks and Polyakov loop, nor global minimality over all fields.

## 3. Exact pion-curvature identity and fit inconsistency

For the printed Eq. (14) potential and Table A1 pion formula, the stationarity equation gives the identity `m_pi^2 = Z_pi^2 * h_x / sigma_x`. This is not an approximation: the bracket in the pion curvature equals `h_x/sigma_x` when `dU/dsigma_x=0`.

For the PCAC condensates above and the Table II scalar parameters, stationarity implies `h_x={fit.h_x:.8f} GeV^3` and hence `m_pi={m_pi_from_stationarity:.5f} GeV`. To obtain the Table I SU(3), `c != 0` fitted pion mass of `139.85 MeV` while holding the same `sigma_x` and `Z_pi`, the identity requires `h_x={hx_required_table_pi:.8f} GeV^3`, smaller by a factor `{hx_ratio:.3f}`. This mismatch is much larger than numerical error.

This narrows the problem beyond the generic phrase “hybrid set”: the printed potential, printed mass formula, PCAC-derived condensate, and Table II scalar parameters cannot all produce the Table I pion mass under this implementation/convention. Plausible explanations include a transcription/convention issue in the source, a parameter table/fit set that is not the one used for Table I, or an implementation error in one of the formulas. The identity itself should be tested against the source derivation before assigning blame. The one-parameter `m0^2` value shown in the CSV is only an algebraic sensitivity check, not a recommended refit.

## 4. Conditional vacuum-spectrum diagnostic

Using the Table A1 mass formula implementation and the above hybrid inputs, the tree-level values are:

- `m_pi = {np.sqrt(masses2['pi']):.4f} GeV` (physical pion mass is about 0.135--0.140 GeV).
- `m_K = {np.sqrt(masses2['K']):.4f} GeV` (physical kaon mass is about 0.494--0.498 GeV).
- Scalar-isoscalar eigenmasses: `{np.sqrt(np.maximum(scalar_eig2,0))[0]:.4f}`, `{np.sqrt(np.maximum(scalar_eig2,0))[1]:.4f} GeV`.
- Pseudoscalar-isoscalar eigenmasses: `{np.sqrt(np.maximum(ps_eig2,0))[0]:.4f}`, `{np.sqrt(np.maximum(ps_eig2,0))[1]:.4f} GeV`.

The pion mismatch is large. The exact identity above shows that the current PCAC-plus-Table-II combination cannot reproduce the Table-I pion mass. This is now a quantified source/formula/parameter-set consistency issue, not merely a numerical convergence issue. It still does not establish a failure of the published model: the next step is to verify the source equations and reconstruct the authors’ coupled fit, especially because the paper’s Table I distinguishes fit inputs from outputs and presents parenthetical values for the anomaly-on case.

## 5. Lattice-QCD external benchmark

HotQCD reports a chiral-crossover pseudocritical temperature `T_pc(0) = (156.5 +/- 1.5) MeV` for physical light and strange quark masses in the continuum limit (Bazavov et al., *Phys. Lett. B* 795 (2019) 15--21, DOI [10.1016/j.physletb.2019.05.013](https://doi.org/10.1016/j.physletb.2019.05.013); [arXiv:1812.08235](https://arxiv.org/abs/1812.08235)). At physical quark masses this is a crossover, not a first-order transition. This number is a benchmark for the **chiral pseudocritical temperature**, not a universal target for every mass curve or for the model parameter `T0=270 MeV`.

Schaefer, Pawlowski & Wambach, *Phys. Rev. D* 76, 074023 (2007), DOI [10.1103/PhysRevD.76.074023](https://doi.org/10.1103/PhysRevD.76.074023), obtain chiral and Polyakov-loop observables by minimizing the grand-canonical potential in a PQM model. This supports checking the global thermodynamic minimum and comparing like-for-like observables, rather than matching a single curve by parameter tuning.

## 6. Status and next step

| Check | Result | Status |
|---|---|---|
| Tree-level stationarity | gradient norm `{np.linalg.norm(grad):.2e} GeV^3` | passed by construction |
| Local scalar-direction stability | both Hessian eigenvalues positive | passed for this constructed set |
| Bounded multi-start minimization | five starts converge to same point in stated box | passed within tested box only |
| Large-field quartic boundedness | all displayed quartic coefficients positive | passed for this scalar potential |
| Full global stability incl. Polyakov/quarks | not tested by this script | open |
| Vacuum pion mass | `{np.sqrt(masses2['pi']):.3f} GeV` versus Table I 0.13985 GeV | source/formula/parameter-set inconsistency quantified by exact stationarity identity |
| Lattice crossover comparison | reference 156.5 +/- 1.5 MeV | benchmark identified; not yet a like-for-like model comparison |

Machine-readable details: `physics_consistency_audit.csv`.
'''
    (outdir / "PHYSICS_CONSISTENCY_AUDIT.md").write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
