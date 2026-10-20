# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
#!/usr/bin/env python3
"""Audit PCAC wave-function factors and isoscalar mass matrices."""
from pathlib import Path
import csv
import numpy as np
from elsm.vacuum_fit import fit_condensates_pcac
from elsm.meson_masses import wavefunction_factors, tree_level_meson_mass_squared
ROOT = Path(__file__).resolve().parents[1]
p = dict(m0_sq=-0.066564, lambda1=3.15, lambda2=36.253, c_anomaly=2.828)
r = fit_condensates_pcac(f_pi=.09502, f_K=.1069, m_a1=1.06976, m_k1=1.25346, g1=5.58, **p)
z = wavefunction_factors(r.sigma_x,r.sigma_y,g1=5.58,m_a1=1.06976,m_k1=1.25346,m_f1s=1.4264,m_kstar=.89555)
m = tree_level_meson_mass_squared(r.sigma_x,r.sigma_y,**p,Z=z)
ps, sc = m['pseudoscalar_isoscalar_matrix'], m['scalar_isoscalar_matrix']
pe, se = np.linalg.eigvalsh(ps), np.linalg.eigvalsh(sc)
rows = [
('sigma_x_PCAC',r.sigma_x,'GeV','Eq.37 implicit PCAC solution'),('sigma_y_PCAC',r.sigma_y,'GeV','Eq.37 implicit PCAC solution'),
('Z_pi',z['Z_pi'],'1','Eq.A13'),('Z_K',z['Z_K'],'1','Eq.A13'),('Z_etaS',z['Z_etaS'],'1','Eq.A13; m_f1S=1.4264 GeV'),('Z_K0star',z['Z_K0star'],'1','Eq.A13; m_Kstar=0.89555 GeV'),
('m_pi',np.sqrt(m['pi']),'GeV','Table A1 with Table II scalar parameters'),('m_K',np.sqrt(m['K']),'GeV','Table A1 with Table II scalar parameters'),
('trace_eta_mass2',np.trace(ps),'GeV^2','Trace of pseudoscalar isoscalar mass-squared matrix'),('eta_lower_eigenmass',np.sqrt(pe[0]),'GeV','Eigenmass'),('eta_upper_eigenmass',np.sqrt(pe[1]),'GeV','Eigenmass'),
('sigma_N_diagonal_mass',np.sqrt(m['sigma_N']),'GeV','Diagonal curvature mass, not mixed eigenmass'),('sigma_lower_eigenmass',np.sqrt(se[0]),'GeV','Lower scalar-isoscalar eigenmass'),('sigma_upper_eigenmass',np.sqrt(se[1]),'GeV','Upper scalar-isoscalar eigenmass'),('g_from_constituent_mass',r.g_from_constituent_mass,'1','g=2 mq/sigma_x, mq=0.3 GeV')]
with (ROOT/'results/APPENDIX_B_CONVENTION_AUDIT.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['quantity','value','unit','interpretation']); w.writerows(rows)
zt=wavefunction_factors(r.sigma_x,r.sigma_y,g1=5.58,m_a1=1.06976,m_k1=1.25346,m_f1s=1.42477,m_kstar=.89555)['Z_etaS']
tab_trace=.5319**2+.96544**2
report=f'''# Appendix B convention audit: Z factors, eta trace, and sigma mass

## What the source says

- Eq. (37) gives PCAC relations with wave-function factors included.
- Appendix B Eq. (A13) defines `Z_pi=Z_etaN`, `Z_K`, `Z_etaS`, and `Z_K0star`.
- Table A1 provides 2x2 pseudoscalar and scalar isoscalar mass-squared matrices in the non-strange/strange basis.
- Table I reports SU(3) `m_sigma = 600 (700) MeV`; its note says parenthetical values are for `c != 0` if different. The table does not explicitly specify the exact eigenstate/curvature convention for the broad sigma resonance.

## Recalculation using the current implementation

Inputs: Table II SU(3), `c != 0`: `m0^2=-0.066564 GeV^2`, `lambda1=3.15`, `lambda2=36.253`, `c=2.828`. PCAC inputs: `f_pi=0.09502`, `f_K=0.1069`, `m_a1=1.06976`, `m_K1=1.25346`, `g1=5.58`. For `Z_etaS` and `Z_K0star`, central experimental masses `m_f1S=1.4264` and `m_Kstar=0.89555 GeV` are used; this is an explicit convention choice, not clearly fixed by the paper for this calculation.

- Condensates: `sigma_x={r.sigma_x:.9f} GeV`, `sigma_y={r.sigma_y:.9f} GeV`.
- Factors: `Z_pi={z['Z_pi']:.9f}`, `Z_K={z['Z_K']:.9f}`, `Z_etaS={z['Z_etaS']:.9f}`, `Z_K0star={z['Z_K0star']:.9f}`.
- PCAC reconstructed decay constants: `f_pi={r.f_pi_reconstructed:.8f} GeV`, `f_K={r.f_K_reconstructed:.8f} GeV`.
- Table A1 masses: `m_pi={np.sqrt(m['pi']):.6f} GeV`, `m_K={np.sqrt(m['K']):.6f} GeV`.
- Pseudoscalar matrix: `[[{ps[0,0]:.9f}, {ps[0,1]:.9f}], [{ps[1,0]:.9f}, {ps[1,1]:.9f}]] GeV^2`. Trace=`{np.trace(ps):.9f} GeV^2`; eigenmasses=`{np.sqrt(pe[0]):.6f}`, `{np.sqrt(pe[1]):.6f} GeV`.
- Table I anomaly-on eta values are 0.5319 and 0.96544 GeV, whose squared-mass sum is `{tab_trace:.9f} GeV^2`.
- Scalar matrix: `[[{sc[0,0]:.9f}, {sc[0,1]:.9f}], [{sc[1,0]:.9f}, {sc[1,1]:.9f}]] GeV^2`. Eigenmasses=`{np.sqrt(se[0]):.6f}`, `{np.sqrt(se[1]):.6f} GeV`; non-strange diagonal curvature mass=`{np.sqrt(m['sigma_N']):.6f} GeV`. Table I anomaly-on scalar value is 0.700 GeV.

## Findings

1. The implemented `Z_pi` and `Z_K` solve the PCAC equations to numerical precision. `Z_etaS` and `Z_K0star` are calculated directly from Eq. (A13), but their values depend on which axial/vector masses are selected.
2. The eta trace is invariant under mixing and should equal `m_eta^2 + m_eta_prime^2` for a consistently defined 2x2 mass-squared matrix. Here it is `{np.trace(ps):.6f} GeV^2`, compared with `{tab_trace:.6f} GeV^2` from Table I. Using the fitted `m_f1=1.42477 GeV` instead of experimental `1.4264 GeV` changes `Z_etaS` from `{z['Z_etaS']:.6f}` to `{zt:.6f}`, too little to explain the trace mismatch by itself.
3. Sigma interpretation is not fully specified in Table I. However, neither the lower mixed eigenmass nor the non-strange diagonal curvature mass of this hybrid input set is 0.700 GeV.
4. The relation `g=2 mq/sigma_x`, with `mq=0.3 GeV`, gives `g={r.g_from_constituent_mass:.6f}`, whereas Table II lists `g=5.5`.

## Limitations

This is not proof that the paper is inconsistent: the calculation combines rounded Table II parameters with condensates inferred from Table I fit outputs and selected vector/axial masses. The authors may use unrounded parameters or a coupled fitting convention not recoverable from the table alone. The next verification should compare each printed Table A1 formula against an independent derivation from the Lagrangian, especially the scalar/pseudoscalar anomaly coefficients and mixing terms.
'''
(ROOT/'results/APPENDIX_B_CONVENTION_AUDIT.md').write_text(report)
print(report)
