"""T_chi scan with the reconstructed, internally consistent vacuum
(sigma_x = 2 m_q / g = 0.10909, sigma_y fitted to c=0 spectrum) vs the paper's Table III."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / 'src'))
from scipy.optimize import minimize
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.potential import grand_potential

SX, SY = 0.6/5.5, 0.103591
ROWS = {'c!=0': (-(0.258**2), 3.15, 36.253, 2.828), 'c=0': (-(0.282**2), -10.515, 55.576, 0.0)}
TARGET = {('c!=0', False): 0.2325, ('c=0', False): 0.2175, ('c!=0', True): 0.2175, ('c=0', True): '2.075 (typo? 0.2075)'}
poly = PolyakovParameters(T0=0.270)
temps = np.arange(0.120, 0.3201, 0.0025)
for (row, FT) in TARGET:
    m0, l1, l2, c = ROWS[row]
    p = SU3Parameters.from_vacuum_condensates(m0_sq=m0, lambda1=l1, lambda2=l2, c_anomaly=c,
                                              sigma_x=SX, sigma_y=SY, g_yukawa=5.5, Tc=0.270, include_FT=FT)
    prev = None; rows = []
    for T in temps:
        obj = lambda z: grand_potential(float(T), z[0], z[1], z[2], z[2], p, polyakov_parameters=poly,
                                        vacuum_prescription='omit', momentum_n_points=160, momentum_p_max=3.)
        starts = ([prev] if prev is not None else []) + [(SX, SY, 0.01), (0.03, 0.08, 0.5), (0.005, 0.05, 0.9)]
        rs = [minimize(obj, x, method='L-BFGS-B', bounds=((0, .6), (0, .8), (0, .999)),
                       options={'maxiter': 800, 'ftol': 1e-12}) for x in starts]
        rs = [r for r in rs if np.isfinite(r.fun)]
        r = min(rs, key=lambda q: q.fun); prev = r.x.copy(); rows.append([T, *r.x])
    a = np.array(rows); d = np.gradient(a[:, 1], a[:, 0]); i = int(np.argmax(np.abs(d)))
    dphi = np.gradient(a[:, 3], a[:, 0]); j = int(np.argmax(np.abs(dphi)))
    half = a[np.argmin(np.abs(a[:, 1]/SX-0.5)), 0]
    print(f'{row:5s} F(T)={str(FT):5s}: T_chi={a[i,0]:.4f} GeV  T_d(phi)={a[j,0]:.4f}  '
          f'half-height={half:.4f}  paper Table III T_chi={TARGET[(row, FT)]}', flush=True)
    np.savetxt(f'scan_{row.replace("!","n")}_{FT}.csv', a, delimiter=',', header='T,sx,sy,phi', comments='')
