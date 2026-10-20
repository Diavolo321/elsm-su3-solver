# LEGACY (superseded 2026-10-18): uses the PCAC vacuum sigma_x=0.1263 GeV, which is NOT the
# authors' vacuum. Kept only to reproduce old audit files. Use scripts/reproduce_paper_tables.py.
"""Audit normalization, parameter consistency, thermal prefactor and Fig. 1 shape.

This script does not fit parameters. It reads the existing model scans and the
manually selected raster anchors, computes transparent shape diagnostics, and
records the known parameter inconsistency. The table-g scan inherits PCAC-derived
h_x,h_y, so it is a hybrid diagnostic and is explicitly not a self-consistent
paper parameter set.
"""
from pathlib import Path
import sys
import csv
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from elsm.vacuum_fit import fit_condensates_pcac
RES = ROOT / "results"
fit = fit_condensates_pcac(
    f_pi=.09502, f_K=.1069, m_a1=1.06976, m_k1=1.25346, g1=5.58,
    m0_sq=-.066564, lambda1=3.15, lambda2=36.253,
    c_anomaly=2.828, g_yukawa=5.5, constituent_mass=.300)

anchors = pd.read_csv(RES / "fig1_sigmax_manual_anchors.csv")

def crossing(x, y, level):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ix = np.flatnonzero(y <= level)
    if len(ix) == 0:
        return float("nan")
    i = int(ix[0])
    if i == 0:
        return float(x[0])
    if y[i] == y[i-1]:
        return float(x[i])
    return float(x[i-1] + (level-y[i-1])*(x[i]-x[i-1])/(y[i]-y[i-1]))

# The observable is normalized by the same PCAC vacuum sigma_x(0), not by each
# scan's first sampled point. This keeps the normalization convention explicit.
sigma0 = fit.sigma_x
rows = []
def add(label, frame, col="sigma_x_GeV", g=None, thermal_factor=None, caveat=""):
    x = frame.T_GeV.to_numpy(float)
    y = frame[col].to_numpy(float) / sigma0
    inside = (anchors.T_GeV.to_numpy() >= x.min()) & (anchors.T_GeV.to_numpy() <= x.max())
    ax = anchors.T_GeV.to_numpy()[inside]
    ay = anchors.sigma_x_over_sigma_x0_digitized.to_numpy()[inside]
    my = np.interp(ax, x, y)
    rmse = float(np.sqrt(np.mean((my-ay)**2))) if len(ax) else float('nan')
    rows.append({
        "case": label, "g_used": g if g is not None else "",
        "thermal_normalization": thermal_factor if thermal_factor is not None else "",
        "normalization_sigma_x0_GeV": sigma0,
        "n_anchor_points_in_overlap": int(len(ax)),
        "digitized_half_height_T_GeV": crossing(anchors.T_GeV, anchors.sigma_x_over_sigma_x0_digitized, .5),
        "model_half_height_T_GeV": crossing(x, y, .5),
        "model_10pct_T_GeV": crossing(x, y, .1),
        "shape_RMSE_on_manual_anchors": rmse,
        "interpretation": caveat,
    })

g5 = pd.read_csv(RES / "corrected_scan_g_5.500.csv")
pcac = pd.read_csv(RES / "wide_convergence_pcac_g.csv")
therm = pd.read_csv(RES / "thermal_normalization_comparison.csv")
add("table_g_5.5_hxhy_from_PCAC_hybrid", g5, g=5.5,
    caveat="Not self-consistent: g=5.5 scan reuses PCAC-derived hx,hy and sigma_x0; shape diagnostic only.")
add("PCAC_g_standard_traced_log", pcac, g=fit.g_from_constituent_mass, thermal_factor=1,
    caveat="PCAC-consistent Yukawa relation; no-sea baseline; approximate manual raster anchors.")
for name, df in therm.groupby("scenario"):
    norm = float(df.thermal_normalization.iloc[0])
    add(name, df, g=fit.g_from_constituent_mass, thermal_factor=norm,
        caveat="Same PCAC-derived hx,hy and optimizer; factor 6 is a literal-prefactor sensitivity experiment, not endorsed physics.")

out = RES / "replication_gap_audit.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

summary = f'''# Replication gap audit (2026-10-18)\n\n## Observable and normalization\n\nThe target is the dimensionless left-panel observable $\\sigma_x(T)/\\sigma_x(0)$. All model traces in `replication_gap_audit.csv` are divided by the same independently PCAC-derived vacuum value $\\sigma_x(0)={sigma0:.8f}$ GeV, rather than by each scan's first sampled value. This removes a small normalization inconsistency but does not repair parameter inconsistency. The figure anchors remain manual raster estimates with several-pixel uncertainty.\n\nThe manual anchors give an approximate half-height crossing at $T={crossing(anchors.T_GeV, anchors.sigma_x_over_sigma_x0_digitized, .5):.4f}$ GeV. This is an approximate digitization diagnostic, not a precise measurement from the paper.\n\n## Parameter consistency\n\nThe PCAC fit gives $\\sigma_x(0)={fit.sigma_x:.8f}$ GeV and $\\sigma_y(0)={fit.sigma_y:.8f}$ GeV, reconstructs the input decay constants, and implies $g=2m_q/\\sigma_x(0)={fit.g_from_constituent_mass:.6f}$ for $m_q=0.300$ GeV. The Table II value $g=5.5$ would instead imply $\\sigma_x(0)=2(0.300)/5.5={2*.300/5.5:.8f}$ GeV. Thus these cannot simultaneously satisfy the stated Yukawa relation. Also, the existing `corrected_scan_g_5.500.csv` reuses $h_x,h_y$ derived from the PCAC condensates; it is a hybrid sensitivity scan, not a self-consistent independent parameter set.\n\n## Thermal normalization\n\n`replication_gap_audit.csv` compares the standard color-traced-log convention (external spin factor only) against the literal Eq. (27) prefactor scenario. The latter is retained only as a transcription sensitivity test. It must not be selected merely because its curve is closer to the figure. The vacuum zero-point term remains omitted under the source-motivated no-sea baseline; the target equation's printed vacuum term lacks a sufficiently clear regulator/subtraction prescription.\n\n## Quantitative status\n\nShape RMSE is reported only as an explicitly labelled diagnostic against manual anchors, not as a publication-grade validation metric. It is sensitive to the manual line trace and line overlap. The target curve and model remain discrepant in crossover location. No parameter fit was performed.\n'''
(RES / "REPLICATION_GAP_AUDIT.md").write_text(summary)
print(f"PCAC sigma_x0={fit.sigma_x:.8f} GeV, sigma_y0={fit.sigma_y:.8f} GeV, implied g={fit.g_from_constituent_mass:.6f}")
print(f"Table g=5.5 implies sigma_x0={2*.300/5.5:.8f} GeV")
print(pd.DataFrame(rows)[["case","model_half_height_T_GeV","model_10pct_T_GeV","shape_RMSE_on_manual_anchors"]].to_string(index=False))
print('Wrote', out, 'and REPLICATION_GAP_AUDIT.md')
