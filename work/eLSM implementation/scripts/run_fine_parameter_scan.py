# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
import sys,time,csv
from pathlib import Path
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.optimize import minimize
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.potential import grand_potential
from elsm.vacuum_fit import fit_condensates_pcac
root=Path(__file__).resolve().parents[1]
fit=fit_condensates_pcac(f_pi=.09502,f_K=.1069,m_a1=1.06976,m_k1=1.25346,g1=5.58,m0_sq=-.066564,lambda1=3.15,lambda2=36.253,c_anomaly=2.828,g_yukawa=5.5,constituent_mass=.300)
scenarios=[('paper_g_T027',5.5,.270),('pcac_g_T027',2*.300/fit.sigma_x,.270),('paper_g_T022',5.5,.220),('paper_g_T020',5.5,.200)]
allrows=[]
for name,g,T0 in scenarios:
 p=SU3Parameters(m0_sq=-.066564,lambda1=3.15,lambda2=36.253,c_anomaly=2.828,h_x=fit.h_x,h_y=fit.h_y,g_yukawa=g)
 poly=PolyakovParameters(T0=T0)
 temps=np.linspace(.20,.28,41); prev=None; rows=[]; t0=time.time()
 for T in temps:
  def obj(z):
   return grand_potential(float(T),z[0],z[1],z[2],z[2],p,polyakov_parameters=poly,vacuum_prescription='omit',momentum_n_points=100,momentum_p_max=2.5)
  starts=[(.12631,.09771,.01),(.08,.08,.25)]
  if prev is not None: starts.insert(0,prev)
  rs=[minimize(obj,x,method='L-BFGS-B',bounds=((0,.6),(0,.8),(0,.999)),options={'maxiter':800,'ftol':2e-9}) for x in starts]
  rs=[r for r in rs if r.success and np.isfinite(r.fun)]
  if not rs: print('FAIL',name,T); break
  r=min(rs,key=lambda x:x.fun); prev=r.x.copy()
  rows.append((T,*r.x,r.fun))
 arr=np.array(rows); d=np.gradient(arr[:,1],arr[:,0]); i=int(np.argmax(np.abs(d)))
 print(name,'g=',g,'T0=',T0,'n=',len(rows),'Tchi=',arr[i,0],'maxslope=',d[i],'elapsed=',round(time.time()-t0,1))
 for j,row in enumerate(arr): allrows.append({'scenario':name,'g':g,'T0':T0,'T_GeV':row[0],'sigma_x_GeV':row[1],'sigma_y_GeV':row[2],'phi':row[3],'omega_GeV4':row[4],'d_sigma_x_dT':d[j]})
with (root/'results'/'fine_parameter_scan.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=allrows[0].keys());w.writeheader();w.writerows(allrows)
