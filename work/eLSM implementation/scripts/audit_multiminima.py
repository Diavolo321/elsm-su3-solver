# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
"""Audit competing minima, mu=0 Polyakov symmetry, and vacuum-cutoff sensitivity.

Outputs are diagnostics only; the vacuum prescriptions are not claimed to match
Eq. (27) of arXiv:2512.22198v2.
"""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.potential import grand_potential
from elsm.solver import solve_mean_fields
from elsm.vacuum_fit import fit_condensates_pcac

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
fit = fit_condensates_pcac(f_pi=.09502, f_K=.1069, m_a1=1.06976,
    m_k1=1.25346, g1=5.58, m0_sq=-.066564, lambda1=3.15,
    lambda2=36.253, c_anomaly=2.828, g_yukawa=5.5,
    constituent_mass=.300)
p = SU3Parameters(m0_sq=-.066564, lambda1=3.15, lambda2=36.253,
    c_anomaly=2.828, h_x=fit.h_x, h_y=fit.h_y, g_yukawa=5.5,
    Tc=.270, include_FT=False)
poly = PolyakovParameters(T0=.270)
starts = [(.1263,.0977,.01,.01), (.08,.08,.25,.25),
          (.03,.04,.70,.70), (.01,.02,.95,.95)]
temps = [.18, .195, .20, .205, .22, .25]
rows=[]
for T in temps:
    for vac, cutoff in [('omit', None), ('sharp_cutoff_subtracted', .6),
                        ('sharp_cutoff_subtracted', .8), ('sharp_cutoff_subtracted', 1.0)]:
        best, runs = solve_mean_fields(T,p,poly,starts=starts,
            vacuum_prescription=vac, vacuum_cutoff=cutoff)
        for i,r in enumerate(runs):
            rows.append(dict(T_GeV=T, vacuum_prescription=vac,
                cutoff_GeV='' if cutoff is None else cutoff, start_index=i,
                sigma_x_GeV=r.x[0], sigma_y_GeV=r.x[1], phi=r.x[2],
                phi_bar=r.x[3], omega_GeV4=r.omega, success=r.success,
                grad_norm=r.grad_norm, selected_global_among_starts=(r is best),
                distance_phi_phibar=abs(r.x[2]-r.x[3])))
    # At mu=0 impose the symmetry phi == phi_bar as a controlled diagnostic.
    def f(z):
        return grand_potential(T,z[0],z[1],z[2],z[2],p,
            polyakov_parameters=poly,vacuum_prescription='omit')
    sym=[]
    for x0 in [(0.1263,.0977,.01),(.08,.08,.25),(.03,.04,.70),(.01,.02,.95)]:
        rr=minimize(f,x0,method='L-BFGS-B',bounds=((0,.6),(0,.8),(0,.999)),
                    options={'maxiter':5000,'ftol':1e-12})
        if rr.success and np.isfinite(rr.fun): sym.append(rr)
    sr=min(sym,key=lambda q:q.fun)
    rows.append(dict(T_GeV=T,vacuum_prescription='omit_phi_equals_phibar',
        cutoff_GeV='',start_index='best_symmetric',sigma_x_GeV=sr.x[0],
        sigma_y_GeV=sr.x[1],phi=sr.x[2],phi_bar=sr.x[2],omega_GeV4=sr.fun,
        success=sr.success,grad_norm=float('nan'),selected_global_among_starts=True,
        distance_phi_phibar=0.0))
with (OUT/'multiminima_vacuum_sensitivity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(f'wrote {len(rows)} diagnostic rows to results/multiminima_vacuum_sensitivity.csv')
