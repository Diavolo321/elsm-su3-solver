# SU(3) eLSM reference solver — audited computational baseline

This repository is a transparent numerical baseline for the SU(3) sector of
“Temperature Dependence of the Masses of Various Meson States: A Comparative
Study in SU(3) and SU(4) extended Linear-Sigma Model”, arXiv:2512.22198v2.
It is **not yet a validated reproduction of every figure/table** in that paper.
The source paper leaves several conventions underspecified or internally
ambiguous; those are made explicit here rather than silently guessed.

## Implemented

- Tree-level mesonic potential, Eq. (14), with analytic gradient and Hessian.
- Stationarity-based calculation of `h_x`, `h_y` from Eqs. (15)–(16), once
  vacuum condensates are supplied.
- New implicit PCAC calibration (`fit_condensates_pcac`) solving Eq. (37) for
  `sigma_x,sigma_y` with the fitted `f_pi,f_K` and `Z_pi,Z_K`; reports the
  implied `g=2 m_q/sigma_x` mismatch rather than hiding it.
- Standard Roessner–Ratti–Weise logarithmic SU(3) Polyakov potential with
  `a0=3.51`, `a1=-2.47`, `a2=15.2`, `b3=-1.75`, and configurable `T0`.
- Thermal quark/antiquark logarithms from Eq. (27), using the mass convention
  in the paper and radial Simpson integration.
- Explicit, optional sharp-cutoff quark zero-point terms in both unsubtracted
  and massless-reference-subtracted forms. These are selectable model
  prescriptions, **neither is claimed to be the paper's prescription**.
- Polyakov-modified occupation functions in Eqs. (33)–(36).
- SU(3) tree-level scalar, pseudoscalar, vector and axial-vector mass-squared
  formulas from Appendix B, Table A1, including the `Z_i` wave-function factors
  and the two isoscalar mixing matrices.
- Finite-temperature mass-squared correction integral from Eq. (32), with the
  Polyakov occupation functions from Eqs. (33)–(36). Channel-specific quark-mass
  derivatives are explicit inputs; a helper is included for the `(sigma_x,sigma_y)`
  mean-field directions.
- Helpers to add thermal corrections to a curvature mass matrix and diagonalize it.
- Multi-start minimization, a four-field mean-field solver and temperature
  continuation helper. Bounds in the convenience solver are numerical guardrails
  and should be documented/varied in a production study.
- Tests for gradients, Hessian, domain restrictions, stationarity, thermal
  correction matrix operations and explicit vacuum-prescription behavior.

## Status after the 2026-10-18 fix (read first)

The model vacuum is now the authors' convention: `sigma_x = 2 m_q/g`, `sigma_y` fitted to the
Table I spectrum (`fit_condensates_yukawa`). With this, Tables I/V (pi, eta, eta', sigma, a0,
K0*, f0, rho, a1) are reproduced to <0.5% and Table III T_chi to within one 2.5 MeV grid step.
Two Table A1 misprints (eta_NS, K0*) are corrected against the Lagrangian. See
`CHANGELOG_FIX.md`; run `PYTHONPATH=src python scripts/reproduce_paper_tables.py`.
The PCAC route (`fit_condensates_pcac`) is diagnostic only. Items 2 and 4 below are resolved;
older numbers in this README and in `results/` that use sigma_x=0.1263 are superseded.

## Important source-audit caveats (historical)

1. **Vacuum quark term:** Eq. (27) as printed contains a prefactor multiplying
   the zero-point integral that appears dimensionally inconsistent and does not
   state a regulator/subtraction prescription. The code therefore does not
   silently include it. `grand_potential(..., vacuum_prescription="omit")` is
   thermal-only. `sharp_cutoff_subtracted` and `sharp_cutoff_unsubtracted` are
   explicit alternatives with a user-specified three-momentum cutoff; neither
   should be described as the article's exact prescription without further
   evidence. Their outputs differ by the chosen reference subtraction.
2. **Vacuum condensates:** the paper gives `g=2 m_q/sigma_x` and PCAC relations
   involving `Z` factors, but the numerical fit and displayed values do not
   unambiguously determine a single internally consistent `sigma_x,sigma_y`
   set from the main text alone. They remain null in `configs/su3.yaml`. Use
   `SU3Parameters.from_vacuum_condensates(...)` to derive `h_x,h_y` after the
   condensates are independently fixed and documented.
3. **Eq. (32):** the generic thermal matrix correction is implemented, but the
   main paper does not print the channel-specific derivatives `d(m_f^2)/d(phi)`
   and `d^2(m_f^2)/d(phi)d(phi)` for each physical meson. The helper covers only
   the `(sigma_x,sigma_y)` directions. We intentionally do not fabricate a
   full channel map; a validated mapping still requires deriving the Yukawa
   mass matrix in the field basis used by Refs. [59,66].
4. **Mass formulas:** the SU(3) vacuum expressions in Appendix B, Table A1, and
   the `Z_i` factors are implemented. The in-medium Eq. (32) correction is a
   generic matrix routine, but it is not automatically wired to every physical
   channel because the paper delegates some channel-specific derivatives to its
   references. A complete reproduction still needs those derivatives audited
   channel by channel and the outputs compared with the paper plots.
5. **Temperature-dependent `F(T)`:** implemented as the separate switch
   `include_FT`, corresponding to Eq. (43). It is not the Eq. (32) thermal
   mass correction.
6. **Polyakov coefficients:** the standard logarithmic set is grounded in the
   Roessner–Ratti–Weise parameterization (Phys. Rev. D 75, 034007 (2007), DOI
   10.1103/PhysRevD.75.034007), also widely reproduced in the PLSM literature.
   The main paper sets `T0=0.27 GeV`. This is a source-based standard choice,
   but full reproduction still requires matching the authors' precise numerical
   implementation and conventions.

## Install and test

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Minimal usage

```python
from elsm import SU3Parameters, PolyakovParameters, grand_potential

from elsm import fit_condensates_yukawa
vac = fit_condensates_yukawa(m0_sq=-0.066564, lambda1=3.15, lambda2=36.253,
                             c_anomaly=2.828)          # sigma_x=0.10909, sigma_y=0.1036
p = SU3Parameters.from_vacuum_condensates(
    m0_sq=-0.066564, lambda1=3.15, lambda2=36.253, c_anomaly=2.828,
    sigma_x=vac.sigma_x, sigma_y=vac.sigma_y, g_yukawa=5.5, Tc=0.270, include_FT=False,
)
U = grand_potential(
    T=0.15, sigma_x=vac.sigma_x, sigma_y=vac.sigma_y,
    phi=0.2, phi_bar=0.2, p=p,
    polyakov_parameters=PolyakovParameters(T0=0.270),
    vacuum_prescription="omit",
)
```

The minimal example uses the reproduction parameter set (Table II, c != 0 row).

## Source references

- Friesen et al., arXiv:2512.22198v2, especially Eqs. (14)–(16), (26)–(36),
  (37), (43), Table II and Appendix B.
- Roessner, Ratti & Weise, “Thermodynamics of the two-flavor Polyakov–Nambu–Jona-Lasinio model”, Phys. Rev. D 75, 034007 (2007), DOI 10.1103/PhysRevD.75.034007.
- Tawfik, Diab & Hussein, “Chiral phase structure of the sixteen meson states in
the SU(3) Polyakov linear-sigma model for finite temperature and chemical
potential in a strong magnetic field”, Chin. Phys. C 43, 034103 (2019),
DOI 10.1088/1674-1137/43/3/034103.

## External literature audit

See [`EXTERNAL_CROSSCHECK.md`](EXTERNAL_CROSSCHECK.md) for a comparison with the
published 2026 version and the related 2025 derivation paper. That comparison
identifies apparent errors in the related paper's printed stationarity and
wave-function-factor equations. The repository follows differentiation of the
stated potential and dimensionally consistent square-root Z factors, rather
than copying those suspect lines literally.

The test suite includes finite-difference checks of the mean-field potential's
gradient/Hessian and of quark-mass derivatives in the diagonal condensate
subspace. These checks do not establish the full set of thermal corrections
for charged/off-diagonal meson channels, nor do they constitute reproduction of
the article's temperature curves. See `AUDIT_NOTES.md` for the remaining gates.

### Sensitivity audit

Run `PYTHONPATH=src python scripts/run_sensitivity_audit.py` to reproduce the
mu=0 symmetric-loop thermal-only scan, momentum-grid/cutoff sensitivity, and
explicit alternative vacuum-prescription checks. The source [59] cited by the
target article (Schaefer & Wagner, arXiv:0808.1491) explicitly neglects the
UV-divergent vacuum contribution in its no-sea mean-field approximation; this
supports, but does not author-confirm, using `vacuum_prescription="omit"` as
the source-consistent baseline. The raster Fig. 1 curve trace is not yet
validated and no RMSE is reported.

## Independent physics checks (not only figure replication)

Run `python scripts/run_physics_consistency_audit.py` to regenerate
`results/PHYSICS_CONSISTENCY_AUDIT.md` and `results/physics_consistency_audit.csv`.
This checks tree-level stationarity, Hessian eigenvalues, quartic boundedness,
a bounded multi-start minimum search, and a conditional vacuum-mass diagnostic.
The spectrum calculation intentionally labels its mixed PCAC/Table-II inputs as
a hybrid diagnostic; it must not be interpreted as a definitive failure of the
published model until the complete vacuum fit is reconstructed consistently.
The report also records the HotQCD continuum crossover benchmark and source links.
