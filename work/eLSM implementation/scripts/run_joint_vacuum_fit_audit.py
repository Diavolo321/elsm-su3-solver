# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
#!/usr/bin/env python3
"""Conditional reconstruction of the paper's anomaly-on SU(3) vacuum fit.

This is a diagnostic with PCAC condensates fixed from the tabulated decay
constants/axial-vector masses. It is not claimed to reproduce the authors'
full fit, because their precise objective/inputs and condensate prescription
are not fully specified in the article text.
"""
from pathlib import Path
import sys, csv
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from elsm.vacuum_fit import fit_condensates_pcac
from elsm.meson_masses import wavefunction_factors, tree_level_meson_mass_squared

def main():
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    pc=fit_condensates_pcac(f_pi=.09502,f_K=.1069,m_a1=1.06976,m_k1=1.25346,g1=5.58,
      m0_sq=-(.258**2),lambda1=3.15,lambda2=36.253,c_anomaly=2.828,g_yukawa=5.5, constituent_mass=.3)
    sx,sy=pc.sigma_x,pc.sigma_y
    Z=wavefunction_factors(sx,sy,g1=5.58,m_a1=1.06976,m_k1=1.25346,m_f1s=1.4264,m_kstar=.89555)
    # Parenthetical anomaly-on Table I values as transcribed from the PDF.
    targets={'pi2':.13985**2,'K2':.42049**2,'eta_sum2':.5319**2+.96544**2,'sigma_light2':.700**2}
    def masses(x):
        m0,c,l1,l2=x
        return tree_level_meson_mass_squared(sx,sy,m0_sq=m0,c_anomaly=c,lambda1=l1,lambda2=l2,Z=Z)
    def residual(x):
        try:
            q=masses(x); se=np.linalg.eigvalsh(q['scalar_isoscalar_matrix'])
            vals=[q['pi'],q['K'],q['eta_N']+q['eta_S'],se[0]]
            scales=[.02**2,.05**2,.1**2,.1**2]
            return np.array([(v-t)/s for v,t,s in zip(vals,targets.values(),scales)])
        except (ValueError, FloatingPointError): return np.ones(4)*1e6
    table=np.array([-(.258**2),2.828,3.15,36.253])
    # Wide but finite bounds; result is a diagnostic weighted least-squares point.
    fit=least_squares(residual,table,bounds=([-2,0,-20,0],[1,10,30,200]),max_nfev=100000,
                      xtol=1e-13,ftol=1e-13,gtol=1e-13)
    rows=[]
    for label,x in [('Table II',table),('conditional least-squares',fit.x)]:
        q=masses(x); se=np.linalg.eigvalsh(q['scalar_isoscalar_matrix'])
        vals=[np.sqrt(max(q['pi'],0)),np.sqrt(max(q['K'],0)),q['eta_N']+q['eta_S'],np.sqrt(max(se[0],0))]
        for name,val,target,unit in zip(['m_pi','m_K','m_eta^2+m_eta_prime^2','light scalar eigenmass'],vals,
              [.13985,.42049,targets['eta_sum2'],.700],['GeV','GeV','GeV^2','GeV']):
            rows.append([label,name,val,target,unit, val-target])
    with (out/'joint_vacuum_fit_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['parameter_set','observable','model_value','target_value','unit','model_minus_target']); w.writerows(rows)
    ptxt=f'''# Conditional joint vacuum-fit audit (SU(3), anomaly on)

## Scope and caveat

The article states that for `c != 0` the fit constrains the pion and kaon masses, the sum `m_eta^2 + m_eta_prime^2`, and the scalar sigma mass. The text does not fully specify the numerical objective, weighting, or every input used in the authors' fit. This script therefore performs a **conditional diagnostic**, not a claimed reproduction of the original optimizer: it fixes `sigma_x,y` and wave-function factors from the PCAC relations and the tabulated `f_pi`, `f_K`, `m_a1`, `m_K1`, `g1`, then fits `(m0^2,c,lambda1,lambda2)` to the four constraints.

The parenthetical anomaly-on Table I values were transcribed as `m_pi=139.85 MeV`, `m_K=420.49 MeV`, `m_eta=531.9 MeV`, `m_eta'=965.44 MeV`, `m_sigma=700 MeV`. The kaon target is notably below the physical kaon mass; it is used here as printed, not silently replaced with the physical value. The scalar target is interpreted as the lower eigenmass of the isoscalar scalar matrix; this interpretation itself is a modelling assumption.

## Fixed PCAC inputs

- `sigma_x = {sx:.8f} GeV`, `sigma_y = {sy:.8f} GeV`
- `Z_pi = {Z['Z_pi']:.8f}`, `Z_K = {Z['Z_K']:.8f}`, `Z_etaS = {Z['Z_etaS']:.8f}`

## Results

Table II anomaly-on parameters: `m0^2=-(0.258 GeV)^2`, `c=2.828`, `lambda1=3.15`, `lambda2=36.253`. The scaled residual vector for constraints `(m_pi^2,m_K^2,eta squared-mass sum,light scalar eigenmass squared)` is `{np.array2string(residual(table),precision=5)}`. It is far from zero in this conditional setup.

Weighted least-squares diagnostic solution:

- `m0^2 = {fit.x[0]:.8g} GeV^2`
- `c = {fit.x[1]:.8g}`
- `lambda1 = {fit.x[2]:.8g}`
- `lambda2 = {fit.x[3]:.8g}`
- scaled residual vector = `{np.array2string(residual(fit.x),precision=5)}`
- optimizer status: `{fit.message}`

The fitted point hits or approaches a parameter bound and leaves substantial residuals, especially in the kaon constraint. This means the four constraints are not jointly reproduced by this **fixed-PCAC-condensate implementation within the tested parameter box**. It does not prove that the paper's fit is impossible: the authors may use different condensates, conventions, mass formulas, input values, or fitting choices. However, it does mean Table II cannot simply be combined with this PCAC reconstruction and these Table I targets and treated as a self-consistent vacuum parameter set.

## Interpretation

The next check is source-level, not temperature tuning: inspect the PDF table alignment and Appendix B formulas, and determine whether the Table I parenthetical values are actually the targets used in the anomaly-on fit or model outputs. The discrepancy between the listed kaon value and physical kaon mass makes that distinction essential. Thermal predictions remain unvalidated until this ambiguity and the vacuum-spectrum mismatch are resolved.

Machine-readable values: `joint_vacuum_fit_audit.csv`.
'''
    (out/'JOINT_VACUUM_FIT_AUDIT.md').write_text(ptxt,encoding='utf-8')
    print(ptxt)
if __name__=='__main__': main()
