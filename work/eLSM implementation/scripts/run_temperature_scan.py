"""Reproducible exploratory SU(3) mean-field temperature scan.

This is a diagnostic run, not a validated reproduction of Fig. 1: the article's
vacuum quark prescription is unresolved and its tabulated g conflicts with the
PCAC-derived sigma_x. Outputs CSV data and a PNG plot with those caveats recorded.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from elsm.parameters import SU3Parameters, PolyakovParameters
from elsm.solver import solve_mean_fields
from elsm.vacuum_fit import fit_condensates_yukawa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
# FIX: authors' vacuum (sigma_x = 2 m_q/g), not the PCAC inversion.
fit = fit_condensates_yukawa(m0_sq=-0.066564, lambda1=3.15, lambda2=36.253,
                             c_anomaly=2.828, g_yukawa=5.5, constituent_mass=0.300)
p = SU3Parameters(m0_sq=-0.066564, lambda1=3.15, lambda2=36.253,
                  c_anomaly=2.828, h_x=fit.h_x, h_y=fit.h_y,
                  g_yukawa=5.5, Tc=0.270, include_FT=False)
poly = PolyakovParameters(T0=0.270)
# Dense enough to reveal branch issues without implying precision unsupported by source.
temperatures = np.linspace(0.02, 0.30, 57)
rows = []
previous = None
for T in temperatures:
    starts = [(fit.sigma_x, fit.sigma_y, 0.01, 0.01), (0.08, 0.08, 0.25, 0.25),
              (0.03, 0.04, 0.70, 0.70), (0.01, 0.02, 0.95, 0.95)]
    if previous is not None:
        starts.insert(0, previous)
    best, runs = solve_mean_fields(float(T), p, poly, starts=starts,
                                   vacuum_prescription="omit")
    sx, sy, phi, phibar = best.x
    rows.append({"T_GeV": T, "sigma_x_GeV": sx, "sigma_y_GeV": sy,
                 "phi": phi, "phi_bar": phibar, "omega_GeV4": best.omega,
                 "optimizer_success": best.success, "grad_norm": best.grad_norm,
                 "nfev": best.nfev, "vacuum_prescription": "omit",
                 "parameter_set": "Yukawa vacuum sigma_x=2m_q/g, g=5.5"})
    previous = best.x.copy()
with (OUT / "exploratory_temperature_scan.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
fig, ax = plt.subplots()
ax.plot([r["T_GeV"] for r in rows], [r["sigma_x_GeV"] for r in rows], label="sigma_x")
ax.plot([r["T_GeV"] for r in rows], [r["sigma_y_GeV"] for r in rows], label="sigma_y")
ax.set(xlabel="T [GeV]", ylabel="Condensate [GeV]", title="Exploratory SU(3) scan — not yet validated")
ax.grid(True); ax.legend(); fig.tight_layout()
fig.savefig(OUT / "exploratory_condensates.png", dpi=160); plt.close(fig)
(OUT / "SCAN_METADATA.md").write_text(f'''# Exploratory scan metadata\n\n- Temperatures: {len(rows)} points from {temperatures[0]:.3f} to {temperatures[-1]:.3f} GeV.\n- Vacuum (Yukawa convention): sigma_x={fit.sigma_x:.9f} GeV, sigma_y={fit.sigma_y:.9f} GeV.\n- Derived h_x={fit.h_x:.9f} GeV^3, h_y={fit.h_y:.9f} GeV^3.\n- g = 5.5 = 2 m_q/sigma_x (consistent).\n- Vacuum quark term: omitted. This is a stated convention for this diagnostic, not a claim about the paper.\n- Polyakov logarithmic potential: standard RRW coefficients, T0=0.270 GeV.\n- Important: the abrupt branch changes / boundary hits must be investigated; this scan is not evidence of reproduction.\n- Comparison metrics against paper figures are intentionally not reported because a verified digitized reference dataset has not yet been assembled.\n''')
print(f"wrote {len(rows)} rows; sigma_x={fit.sigma_x:.6f}, sigma_y={fit.sigma_y:.6f}")
