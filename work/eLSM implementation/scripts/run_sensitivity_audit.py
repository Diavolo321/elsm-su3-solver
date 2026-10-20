# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
"""Reproducible SU(3) sensitivity audit for the Fig. 1 comparison.

The source plot was inspected, but no RMSE is reported: overlapping red/black
curves in the raster figure make automatic channel assignment ambiguous. This
script instead computes source-consistent no-sea curves and numeric sensitivity
checks. Run with PYTHONPATH=src python scripts/run_sensitivity_audit.py.
"""
from pathlib import Path
import csv
import numpy as np
import cv2
from scipy.optimize import minimize
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.potential import grand_potential
from elsm.vacuum_fit import fit_condensates_pcac

ROOT=Path(__file__).resolve().parents[1]
fit=fit_condensates_pcac(f_pi=.09502,f_K=.1069,m_a1=1.06976,m_k1=1.25346,g1=5.58,m0_sq=-.066564,lambda1=3.15,lambda2=36.253,c_anomaly=2.828,g_yukawa=5.5,constituent_mass=.300)
base=SU3Parameters.from_vacuum_condensates(m0_sq=-.066564,lambda1=3.15,lambda2=36.253,c_anomaly=2.828,sigma_x=fit.sigma_x,sigma_y=fit.sigma_y,g_yukawa=5.5)
poly=PolyakovParameters(T0=.270)
# Enforce the mu=0 mean-field relation phi=phi_bar, consistent with the usual
# real mean-field prescription; compare several starts and integration settings.
def solve(T, n_points=1000, pmax=None, vacuum='omit', cutoff=None):
    def obj(z):
        return grand_potential(T,z[0],z[1],z[2],z[2],base,polyakov_parameters=poly,
            vacuum_prescription=vacuum,vacuum_cutoff=cutoff,momentum_n_points=n_points,momentum_p_max=pmax)
    starts=[(.1263,.0977,.01),(.08,.08,.25),(.03,.04,.70),(.01,.02,.95)]
    rs=[minimize(obj,x0,method='L-BFGS-B',bounds=((0,.6),(0,.8),(0,.999)),
                 options={'maxiter':5000,'ftol':1e-12,'gtol':1e-8}) for x0 in starts]
    valid=[r for r in rs if r.success and np.isfinite(r.fun)]
    if not valid: raise RuntimeError(f'No successful solution at T={T}')
    return min(valid,key=lambda r:r.fun)

# Resolution and momentum-cutoff sensitivity at representative temperatures.
sens=[]
for T in [.18,.20,.22,.25]:
    for n in [400,800,1600]:
        for pmax in [3.0,5.0,8.0]:
            r=solve(T,n,pmax)
            sens.append(dict(T_GeV=T,n_points=n,p_max_GeV=pmax,sigma_x_GeV=r.x[0],sigma_y_GeV=r.x[1],phi=r.x[2],omega_GeV4=r.fun,success=r.success))
with (ROOT/'results'/'numerical_sensitivity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=sens[0].keys());w.writeheader();w.writerows(sens)

# Compare two explicit vacuum prescriptions at a fixed momentum resolution.
vac=[]
for T in [.18,.20,.22,.25]:
    for prescription,cutoff in [('omit',None),('sharp_cutoff_subtracted',.6),('sharp_cutoff_subtracted',.8),('sharp_cutoff_subtracted',1.0)]:
        r=solve(T,1000,None,prescription,cutoff)
        vac.append(dict(T_GeV=T,prescription=prescription,cutoff_GeV='' if cutoff is None else cutoff,sigma_x_GeV=r.x[0],sigma_y_GeV=r.x[1],phi=r.x[2],omega_GeV4=r.fun))
with (ROOT/'results'/'vacuum_sensitivity_symmetric.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=vac[0].keys());w.writeheader();w.writerows(vac)

# Model curve at a common temperature grid. No RMSE is claimed unless the trace
# extraction has been manually inspected; save both for reproducible follow-up.
model=[]
for T in np.linspace(.02,.5,49):
    r=solve(float(T),1000,None)
    model.append(dict(T_GeV=T,sigma_x_over_sigma_x0=r.x[0]/fit.sigma_x,sigma_y_over_sigma_y0=r.x[1]/fit.sigma_y,phi=r.x[2],omega_GeV4=r.fun))
with (ROOT/'results'/'symmetric_thermal_only_curve.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=model[0].keys());w.writeheader();w.writerows(model)
print(f'PCAC sx0={fit.sigma_x:.8f} sy0={fit.sigma_y:.8f}; g from mq={fit.g_from_constituent_mass:.6f} vs Table-II g=5.5')
print(f'Wrote {len(sens)} numerical sensitivity cases, {len(vac)} vacuum cases and {len(model)} symmetric thermal-only curve points. No RMSE emitted because raster curve identity is ambiguous.')
