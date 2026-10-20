# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
"""Coarse post-normalization-fix scan; not a final validation run."""
from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.potential import grand_potential
from elsm.vacuum_fit import fit_condensates_pcac
ROOT=Path(__file__).resolve().parents[1]
fit=fit_condensates_pcac(f_pi=.09502,f_K=.1069,m_a1=1.06976,m_k1=1.25346,g1=5.58,m0_sq=-.066564,lambda1=3.15,lambda2=36.253,c_anomaly=2.828,g_yukawa=5.5,constituent_mass=.300)
p=SU3Parameters(m0_sq=-.066564,lambda1=3.15,lambda2=36.253,c_anomaly=2.828,h_x=fit.h_x,h_y=fit.h_y,g_yukawa=5.5)
poly=PolyakovParameters(T0=.270)
temps=np.linspace(.10,.30,25); rows=[]; prev=None
for T in temps:
    def obj(z):
        return grand_potential(float(T),z[0],z[1],z[2],z[2],p,polyakov_parameters=poly,
            vacuum_prescription='omit',momentum_n_points=60,momentum_p_max=2.5)
    starts=[(.12631,.09771,.01),(.08,.08,.25)]
    if prev is not None: starts.insert(0,prev)
    rs=[minimize(obj,x,method='L-BFGS-B',bounds=((0,.6),(0,.8),(0,.999)),
        options={'maxiter':1500,'ftol':1e-10}) for x in starts]
    rs=[r for r in rs if r.success and np.isfinite(r.fun)]
    if not rs: raise RuntimeError(f'No converged solution at T={T}')
    r=min(rs,key=lambda x:x.fun); prev=r.x.copy()
    rows.append({'T_GeV':T,'sigma_x_GeV':r.x[0],'sigma_y_GeV':r.x[1],
                 'phi':r.x[2],'omega_GeV4':r.fun,'optimizer_success':r.success})
path=ROOT/'results'/'corrected_coarse_scan.csv'
with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
deriv=np.gradient([r['sigma_x_GeV'] for r in rows],temps); idx=int(np.argmax(np.abs(deriv)))
print(f'rows={len(rows)}, Tchi_coarse={temps[idx]:.5f} GeV, max_slope={deriv[idx]:.6f} GeV/GeV')
