# Fix log — 2026-10-18: vacuum convention and Table A1 misprints

## Root cause of the wrong pion/kaon masses
The solver used the vacuum obtained by inverting PCAC (Eq. 37) with the Table I
f_pi, f_K: sigma_x = 0.1263, sigma_y = 0.0977 GeV. The authors instead fix
sigma_x = 2 m_q / g = 0.6/5.5 = 0.10909 GeV (Sec. III). Evidence: m_rho and m_a1 of
Table I alone give sigma_x = 0.10902 / 0.10905 GeV without any PCAC input.
Because the pion is a pseudo-Goldstone boson (m_pi^2 = Z_pi^2 h_x/sigma_x, a near
cancellation of O(0.3 GeV^2) terms), +16% in sigma_x gave m_pi = 436 instead of 140 MeV.

## Changes
1. `src/elsm/meson_masses.py`
   - `eta_NS`: c/2*sigma_x -> c/sqrt(2)*sigma_x (Lagrangian; reproduces eta/eta').
   - `K0star` anomaly term: +c/(2 sqrt 2)*sigma_y -> +c/2*sigma_x (Lagrangian; reproduces Table V K0*=1.032).
2. `src/elsm/vacuum_fit.py`: new primary `fit_condensates_yukawa`, `vacuum_spectrum`,
   `sigma_y_from_vector_sector`, `TABLE_I_SU3`; `fit_condensates_pcac` marked diagnostic-only.
3. `configs/su3.yaml`: both Table II rows; vacuum filled; quark vacuum = omit (documentation only — no code reads it yet).
4. `tests/test_paper_reproduction.py`: Lagrangian-Hessian check of Table A1, Tables I/V,
   Eq. (32) normalization vs potential, Table III T_chi (c!=0). 27/27 pass.
5. `scripts/reproduce_paper_tables.py`: new end-to-end reproduction; `run_temperature_scan.py`
   switched to the Yukawa vacuum; other PCAC-based scripts marked LEGACY and their hard-coded
   `/mnt/data/...` paths replaced by repository-relative paths.

## Result
| | model | paper |
|---|---|---|
| m_pi (c!=0) | 140.1 | 139.85 |
| eta / eta' | 534.0 / 966.8 | 531.9 / 965.44 |
| sigma (c!=0 / c=0) | 700.6 / 600.5 | 700 / 600 |
| a0 / K0* / f0 | 926.9 / 1034.3 / 1126.4 | 926 / 1032 / 1124 |
| T_chi c!=0, c=0, c!=0+F, c=0+F | 0.2300 0.2150 0.2175 0.2050 | 0.2325 0.2175 0.2175 0.2075 |

## Still open
- Kaon: model 494.9 MeV (exp. 495.6) vs Table I 420.49. Not reproducible consistently.
- Table I f_pi, f_K are not consistent with its masses (model gives 89.7 / 105.1 MeV).
- Eq. (32) is not yet wired to individual meson channels (needed for Figs. 2-4).
- All files in `results/` produced before this fix (PCAC vacuum) are superseded.
