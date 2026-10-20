# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
import sys,csv,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.optimize import minimize
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.potential import grand_potential
from elsm.vacuum_fit import fit_condensates_pcac
root=str(Path(__file__).resolve().parents[1])
fit=fit_condensates_pcac(f_pi=.09502,f_K=.1069,m_a1=1.06976,m_k1=1.25346,g1=5.58,m0_sq=-.066564,lambda1=3.15,lambda2=36.253,c_anomaly=2.828,g_yukawa=5.5,constituent_mass=.300)
g=2*.300/fit.sigma_x
p=SU3Parameters(m0_sq=-.066564,lambda1=3.15,lambda2=36.253,c_anomaly=2.828,h_x=fit.h_x,h_y=fit.h_y,g_yukawa=g)
poly=PolyakovParameters(T0=.270)
temps=np.arange(.10,.361,.004)
prev=None; rows=[]; start=time.time()
for T in temps:
 def obj(z): return grand_potential(float(T),z[0],z[1],z[2],z[2],p,polyakov_parameters=poly,vacuum_prescription='omit',momentum_n_points=180,momentum_p_max=3.)
 starts=[]
 if prev is not None: starts.append(prev)
 starts += [(.12631,.09771,.01),(.08,.08,.25)]
 rs=[minimize(obj,x,method='L-BFGS-B',bounds=((0,.6),(0,.8),(0,.999)),options={'maxiter':600,'ftol':1e-9}) for x in starts]
 rs=[r for r in rs if r.success and np.isfinite(r.fun)]
 if not rs: print('FAIL',T); continue
 r=min(rs,key=lambda q:q.fun); prev=r.x.copy(); rows.append([T,*r.x,r.fun])
arr=np.array(rows); d=np.gradient(arr[:,1],arr[:,0]); i=int(np.argmax(np.abs(d)))
with open(root+'/results/wide_convergence_pcac_g.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['T_GeV','sigma_x_GeV','sigma_y_GeV','phi','omega_GeV4','d_sigma_x_dT']);w.writerows([list(x)+[y] for x,y in zip(arr,d)])
print('n',len(arr),'Tchi',arr[i,0],'slope',d[i],'range endpoints sx',arr[0,1],arr[-1,1],'secs',time.time()-start)
