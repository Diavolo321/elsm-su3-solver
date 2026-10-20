# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
"""Controlled comparison: standard traced-log normalization vs literal Eq. 27 prefactor.

Both runs use identical PCAC-derived condensate sources, Yukawa coupling, T0,
mesh, optimizer, temperature grid, and no-sea vacuum prescription. The literal
scenario multiplies the traced-log thermal term by nu_c=2*N_c=6, as printed.
This is a transcription scenario, not an endorsed physical convention.
"""
from pathlib import Path
import csv, time
import numpy as np
from scipy.optimize import minimize
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.potential import grand_potential
from elsm.vacuum_fit import fit_condensates_pcac

ROOT = Path(__file__).resolve().parents[1]
fit = fit_condensates_pcac(f_pi=.09502, f_K=.1069, m_a1=1.06976,
    m_k1=1.25346, g1=5.58, m0_sq=-.066564, lambda1=3.15,
    lambda2=36.253, c_anomaly=2.828, g_yukawa=5.5,
    constituent_mass=.300)
g = 2*.300/fit.sigma_x
p = SU3Parameters(m0_sq=-.066564, lambda1=3.15, lambda2=36.253,
    c_anomaly=2.828, h_x=fit.h_x, h_y=fit.h_y, g_yukawa=g)
poly = PolyakovParameters(T0=.270)
temps = np.round(np.arange(.100, .3601, .004), 6)
all_rows=[]; summaries=[]; t0=time.time()
for label, normalization in [('standard_traced_log', 1.0), ('literal_eq27_prefactor', 6.0)]:
    prev=None; rows=[]
    for T in temps:
        def obj(z):
            return grand_potential(float(T), z[0], z[1], z[2], z[2], p,
                polyakov_parameters=poly, vacuum_prescription='omit',
                momentum_n_points=180, momentum_p_max=3.0,
                thermal_normalization=normalization)
        starts=[]
        if prev is not None: starts.append(prev)
        starts += [(.12631,.09771,.01),(.08,.08,.25)]
        rs=[minimize(obj, x, method='L-BFGS-B',
             bounds=((0,.6),(0,.8),(0,.999)),
             options={'maxiter':600,'ftol':1e-9}) for x in starts]
        rs=[r for r in rs if r.success and np.isfinite(r.fun)]
        if not rs:
            raise RuntimeError(f'No converged minimum: {label} T={T:.3f}')
        r=min(rs,key=lambda q:q.fun); prev=r.x.copy()
        rows.append([T,*r.x,r.fun])
    arr=np.asarray(rows); d=np.gradient(arr[:,1],arr[:,0]); i=int(np.argmax(np.abs(d)))
    for row, slope in zip(arr,d): all_rows.append([label,normalization,*row,slope])
    summaries.append([label,normalization,len(arr),arr[i,0],d[i],arr[0,1],arr[-1,1]])
    print(label, 'n=',len(arr),'Tchi=',arr[i,0], 'slope=',d[i],
          'sx(Tmin,Tmax)=',arr[0,1],arr[-1,1])
with (ROOT/'results'/'thermal_normalization_comparison.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['scenario','thermal_normalization','T_GeV','sigma_x_GeV','sigma_y_GeV','phi','omega_GeV4','d_sigma_x_dT']);w.writerows(all_rows)
with (ROOT/'results'/'thermal_normalization_summary.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['scenario','thermal_normalization','n_T','Tchi_sampled_GeV','max_abs_dsigma_x_dT','sigma_x_at_Tmin_GeV','sigma_x_at_Tmax_GeV']);w.writerows(summaries)
print('elapsed_sec=',round(time.time()-t0,1),'g=',g,'T0=.270, n_p=180, pmax=3, no-sea')
