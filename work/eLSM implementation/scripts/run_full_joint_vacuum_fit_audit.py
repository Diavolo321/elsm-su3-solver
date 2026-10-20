# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
#!/usr/bin/env python3
"""Joint SU(3), anomaly-on vacuum diagnostic: PCAC + four mass constraints.
This is a transparent conditional fit, not a guaranteed reproduction of authors' optimizer.
"""
from pathlib import Path
import sys, csv
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from elsm.meson_masses import wavefunction_factors, tree_level_meson_mass_squared
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
G1,MA1,MK1,MF1S,MKSTAR=5.58,1.06976,1.25346,1.4264,.89555
TABLE=np.array([.1263103742,.0977103971,-.066564,2.828,3.15,36.253])
TARGETS={
 'table_model_outputs':np.array([.13985**2,.42049**2,.5319**2+.96544**2,.700**2]),
 'experimental_inputs':np.array([.1373**2,.4956**2,.5479**2+.9578**2,.700**2])}
SCALES=np.array([.02**2,.05**2,.1**2,.1**2])
def observables(x):
 sx,sy,m0,c,l1,l2=x
 Z=wavefunction_factors(sx,sy,g1=G1,m_a1=MA1,m_k1=MK1,m_f1s=MF1S,m_kstar=MKSTAR)
 q=tree_level_meson_mass_squared(sx,sy,m0_sq=m0,c_anomaly=c,lambda1=l1,lambda2=l2,Z=Z)
 eigp=np.linalg.eigvalsh(q['pseudoscalar_isoscalar_matrix']); eigs=np.linalg.eigvalsh(q['scalar_isoscalar_matrix'])
 fpi=sx/Z['Z_pi']; fk=(sx+np.sqrt(2)*sy)/(2*Z['Z_K'])
 return np.array([fpi,fk,q['pi'],q['K'],np.trace(q['pseudoscalar_isoscalar_matrix']),eigs[0]]),Z,q,eigp,eigs
def residual(x,targets):
 try:
  v,*_=observables(x)
  return np.r_[(v[0]-.09502)/.002,(v[1]-.1069)/.002,(v[2:]-targets)/SCALES]
 except (ValueError,FloatingPointError): return np.ones(6)*1e4
def main():
 rows=[]; summaries=[]
 bounds=([.03,.03,-2,0,-20,0],[.3,.3,1,10,30,200])
 for name,targ in TARGETS.items():
  fit=least_squares(residual,TABLE,args=(targ,),bounds=bounds,max_nfev=100000,xtol=1e-13,ftol=1e-13,gtol=1e-13)
  for label,x in [('Table II',TABLE),(f'joint fit: {name}',fit.x)]:
   try:
    v,Z,q,ep,es=observables(x)
    model=np.r_[v[2:5],es[0]]
   except Exception:
    model=np.full(4,np.nan)
   target=targ
   for obs,val,t,unit in zip(['m_pi^2','m_K^2','m_eta^2+m_eta_prime^2','light_scalar_eigenvalue^2'],model,target,['GeV^2','GeV^2','GeV^2','GeV^2']): rows.append([name,label,obs,val,t, val-t,unit])
  summaries.append(f"### {name}\n- fit parameters `(sigma_x,sigma_y,m0_sq,c,lambda1,lambda2)` = `{np.array2string(fit.x, precision=8)}`\n- scaled residual vector = `{np.array2string(residual(fit.x,targ), precision=6)}`\n- cost = `{fit.cost:.8g}`; optimizer: {fit.message}\n- parameters at/near bounds: `lambda1={fit.x[4]:.6g}` (bounds -20 to 30), `m0_sq={fit.x[2]:.6g}`\n")
 with (OUT/'full_joint_vacuum_fit_audit.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['target_set','parameter_set','observable','model_value','target_value','residual','unit']); w.writerows(rows)
 text='''# Full joint vacuum-fit diagnostic (SU(3), anomaly on)\n\n## Method\n\nThis extends the prior conditional test: `sigma_x` and `sigma_y` are now free parameters jointly with `m0_sq`, `c`, `lambda1`, and `lambda2`. Six residuals are used: PCAC `f_pi`, `f_K`, plus four mass constraints. The wave-function factors are recalculated from the condensates at every evaluation. The mass constraints use the code's Table A1 formulas and the trace of the pseudoscalar isoscalar mass-squared matrix for `m_eta^2 + m_eta_prime^2`. The light scalar constraint is interpreted as the lower eigenvalue of the scalar isoscalar matrix. That scalar interpretation and the numerical residual scales are assumptions for this diagnostic, not fully specified fit details from the paper.\n\nThe fit is bounded and weighted; results reaching parameter bounds are not accepted as a robust physical fit. Table II's rounded parameter values are used as the initial/reference set. PCAC values use `f_pi=0.09502`, `f_K=0.1069`, `m_a1=1.06976`, `m_K1=1.25346`, `g1=5.58`; `m_f1s=1.4264`, `m_Kstar=0.89555` enter the other wave-function factors.\n\n## Results\n\n'''+ '\n'.join(summaries)+'''\n## Interpretation\n\nThis is a joint consistency test of the current implementation, not an exact replication of the authors' fit. Both target sets should be read cautiously: the paper describes experimental inputs for the anomalous fit, while Table I also displays model outputs; the target sigma mass and the trace/eigenvalue mapping need confirmation against the typeset table and model conventions. A fit that hits bounds (notably `lambda1`) is not evidence that a physical solution exists or does not exist.\n\nThe robust conclusion at this stage is narrower: Table II plus our PCAC convention does not reproduce the tabulated spectrum directly. The joint fit can move condensates and couplings, but the parameter-bound behaviour indicates that the current constraints/conventions or formula implementation must be checked before judging the paper. Thermal predictions remain not physically validated.\n\nMachine-readable details are in `full_joint_vacuum_fit_audit.csv`.\n'''
 (OUT/'FULL_JOINT_VACUUM_FIT_AUDIT.md').write_text(text,encoding='utf-8')
 print(text)
if __name__=='__main__': main()
