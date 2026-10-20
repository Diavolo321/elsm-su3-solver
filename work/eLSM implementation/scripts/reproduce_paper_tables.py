"""Reproduce Tables I, V (SU(3) vacuum spectrum) and III (T_chi) of arXiv:2512.22198.

Run:  PYTHONPATH=src python scripts/reproduce_paper_tables.py [--fast]
Writes results/PAPER_REPRODUCTION.md and results/tchi_scan_<case>.csv.
"""
from pathlib import Path
import sys, csv
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.potential import grand_potential
from elsm.vacuum_fit import fit_condensates_yukawa, TABLE_I_SU3

FAST = "--fast" in sys.argv
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
ROWS = {"c_finite": dict(m0_sq=-0.258**2, lambda1=3.15, lambda2=36.253, c_anomaly=2.828),
        "c_zero": dict(m0_sq=-0.282**2, lambda1=-10.515, lambda2=55.576, c_anomaly=0.0)}
TABLE_III = {("c_zero", False): 0.2175, ("c_finite", False): 0.2325,
             ("c_zero", True): 0.2075, ("c_finite", True): 0.2175}  # "2.075" read as 0.2075
c1, h2, h3, g1 = 0.44, 40.975, -15.216, 5.58
lines = ["# Paper reproduction (SU(3))", "",
         "Vacuum: sigma_x = 2 m_q/g = 0.109091 GeV (m_q=0.3, g=5.5); sigma_y fitted to Table I "
         "(pi, eta, eta', sigma). h_x, h_y from Eqs. (15)-(16). Thermal: -2T traced log, no-sea, phi=phi_bar.", ""]

lines += ["## Tables I / V (MeV)", "", "| row | sigma_y | pi | K | eta | eta' | sigma | a0 | K0* | f0 | rho | a1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
vac = {}
for name, row in ROWS.items():
    v = fit_condensates_yukawa(**row); vac[name] = v; m = v.masses
    rho = np.sqrt(c1 + 0.5*(h2+h3)*v.sigma_x**2); a1 = np.sqrt(c1 + (g1**2+0.5*(h2-h3))*v.sigma_x**2)
    lines.append(f"| {name} | {v.sigma_y:.5f} | " + " | ".join(f"{1e3*x:.1f}" for x in (
        m["pi"], m["K"], m["eta"], m["eta_prime"], m["sigma"], m["a0"], m["K0star"], m["f0"], rho, a1)) + " |")
lines += ["| paper c!=0 | | 139.85 | 420.49 | 531.9 | 965.44 | 700 | 926 | 1032 | 1124 | 770.11 | 1069.76 |",
          "| paper c=0 | | 139.85 | 420.49 | 140.0 | 644.93 | 600 | | | | 770.11 | 1069.76 |", "",
          "Kaon: the consistent model gives ~495 MeV (experiment 495.6); Table I's 420.49 is not reproduced "
          "by any consistent convention (Z_K=1 -> 407, Z_K0* used for K -> 410).", ""]

lines += ["## Table III: T_chi = argmax |d sigma_x/dT| (GeV)", "", "| row | F(T) | model | paper | T_d (Phi) |", "|---|---|---|---|---|"]
poly = PolyakovParameters(T0=0.270)
step = 0.005 if FAST else 0.0025
temps = np.arange(0.120, 0.3201, step)
for (name, FT), paper in TABLE_III.items():
    v = vac[name]; row = ROWS[name]
    p = SU3Parameters.from_vacuum_condensates(sigma_x=v.sigma_x, sigma_y=v.sigma_y, g_yukawa=5.5,
                                              Tc=0.270, include_FT=FT, **row)
    prev = (v.sigma_x, v.sigma_y, 0.01); data = []
    for T in temps:
        obj = lambda z: grand_potential(float(T), z[0], z[1], z[2], z[2], p, polyakov_parameters=poly,
                                        momentum_n_points=160, momentum_p_max=3.0)
        starts = [prev, (v.sigma_x, v.sigma_y, 0.01), (0.03, 0.08, 0.5), (0.005, 0.05, 0.9)]
        r = min((minimize(obj, x0, method="L-BFGS-B", bounds=((0, .6), (0, .8), (0, .999)),
                          options={"maxiter": 800, "ftol": 1e-12}) for x0 in starts), key=lambda q: q.fun)
        prev = r.x.copy(); data.append([T, *r.x])
    a = np.array(data)
    t_chi = a[np.argmax(np.abs(np.gradient(a[:, 1], a[:, 0]))), 0]
    t_d = a[np.argmax(np.abs(np.gradient(a[:, 3], a[:, 0]))), 0]
    lines.append(f"| {name} | {FT} | {t_chi:.4f} | {paper} | {t_d:.4f} |")
    with (OUT / f"tchi_scan_{name}_FT{int(FT)}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["T_GeV", "sigma_x_GeV", "sigma_y_GeV", "phi"]); w.writerows(a.tolist())
    print(lines[-1], flush=True)
lines += ["", f"Temperature grid step {step*1e3:.1f} MeV; paper values lie on a 2.5 MeV grid."]
(OUT / "PAPER_REPRODUCTION.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
