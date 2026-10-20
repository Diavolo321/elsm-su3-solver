"""Reconstruct which vacuum (sigma_x, sigma_y) the authors actually used,
using Table II parameters + Table A1 (corrected) formulas vs Table I outputs."""
import numpy as np
from scipy.optimize import least_squares, brentq
r2 = np.sqrt(2)
g1, c1, h2, h3, delta_x = 5.58, 0.44, 40.975, -15.216, 0.0
ROWS = {'c!=0': (-(0.258**2), 3.15, 36.253, 2.828), 'c=0': (-(0.282**2), -10.515, 55.576, 0.0)}
TAB = dict(fpi=.09502, fK=.1069, mpi=.13985, mK=.42049, mrho=.77011, ma1=1.06976,
           mK1=1.25346, mphi=1.01919, mf1=1.42477, eta=.5319, etap=.96544, etap_c0=.64493,
           sig_c0=.600, sig_c=.700)

def Zs(sx, sy, ma1=TAB['ma1'], mK1=TAB['mK1'], mf1S=TAB['mf1'], mKs=0.851):
    return dict(pi=ma1/np.sqrt(ma1**2-g1**2*sx**2),
                K=2*mK1/np.sqrt(4*mK1**2-g1**2*(sx+r2*sy)**2),
                etaS=mf1S/np.sqrt(mf1S**2-2*g1**2*sy**2),
                K0s=2*mKs/np.sqrt(4*mKs**2-g1**2*(sx-r2*sy)**2))

def spectrum(sx, sy, row, Z=None, etaNS_coef=1/r2):
    m0, l1, l2, c = ROWS[row]; Z = Z or Zs(sx, sy)
    pi = Z['pi']**2*(m0 - c/r2*sy + (l1+l2/2)*sx**2 + l1*sy**2)
    K = Z['K']**2*(m0 - c/2*sx + (l1+l2/2)*sx**2 + (l1+l2)*sy**2 - l2/r2*sx*sy)
    eN = Z['pi']**2*(m0 + c/r2*sy + (l1+l2/2)*sx**2 + l1*sy**2)
    eS = Z['etaS']**2*(m0 + l1*sx**2 + (l1+l2)*sy**2)
    eNS = Z['pi']*Z['etaS']*etaNS_coef*c*sx
    sN = m0 - c/r2*sy + 1.5*(2*l1+l2)*sx**2 + l1*sy**2
    sS = m0 + 3*(l1+l2)*sy**2 + l1*sx**2
    sNS = -c/r2*sx + 2*l1*sx*sy
    ev = np.linalg.eigvalsh([[eN, eNS], [eNS, eS]]); sv = np.linalg.eigvalsh([[sN, sNS], [sNS, sS]])
    out = dict(mpi=np.sqrt(pi), mK=np.sqrt(max(K, 0)), eta=np.sqrt(max(ev[0], 0)), etap=np.sqrt(ev[1]),
               sigma=np.sqrt(max(sv[0], 0)), fpi=sx/Z['pi'], fK=(sx+r2*sy)/(2*Z['K']))
    out['rho'] = np.sqrt(c1 + 0.5*(h2+h3)*sx**2)
    out['a1'] = np.sqrt(c1 + (g1**2+0.5*(h2-h3))*sx**2)
    out['f1S-phi'] = 2*(g1**2-h3)*sy**2  # m_f1S^2 - m_omegaS^2, independent of delta_y
    return out

if __name__ == '__main__':
    print('=== 1) sigma_x from vector sector alone (no PCAC, no Z):')
    sx_rho = np.sqrt((TAB['mrho']**2 - c1)/(0.5*(h2+h3)))
    sx_a1 = np.sqrt((TAB['ma1']**2 - c1)/(g1**2+0.5*(h2-h3)))
    print(f'  m_rho -> sigma_x = {sx_rho:.5f};  m_a1 -> sigma_x = {sx_a1:.5f};  2*mq/g = {0.6/5.5:.5f}')
    sy_f1 = np.sqrt((TAB['mf1']**2-TAB['mphi']**2)/(2*(g1**2-h3)))
    print(f'  m_f1 - m_phi -> sigma_y = {sy_f1:.5f}')

    print('\n=== 2) Fit (sx,sy) to c=0 row: m_pi, m_eta\' (=etaS), m_sigma')
    def res0(v):
        s = spectrum(*v, 'c=0')
        return [s['mpi']/TAB['mpi']-1, s['etap']/TAB['etap_c0']-1, s['sigma']/TAB['sig_c0']-1]
    f0 = least_squares(res0, [0.11, 0.10]); print('  sx,sy =', f0.x, ' rel.resid =', np.round(f0.fun, 4))

    print('\n=== 3) Spectrum at authors-like vacuum vs code (PCAC) vacuum, c!=0 row')
    for label, (sx, sy) in {'authors-like (0.10909, %.4f)' % f0.x[1]: (0.6/5.5, f0.x[1]),
                            'code PCAC (0.12631, 0.09771)': (0.1263104, 0.0977104)}.items():
        for row in ('c!=0', 'c=0'):
            s = spectrum(sx, sy, row)
            print(f'  {label:30s} {row:5s} mpi={s["mpi"]*1e3:7.1f} mK={s["mK"]*1e3:6.1f} '
                  f'eta={s["eta"]*1e3:6.1f} eta\'={s["etap"]*1e3:6.1f} sigma={s["sigma"]*1e3:6.1f} '
                  f'fpi={s["fpi"]*1e3:5.1f} fK={s["fK"]*1e3:5.1f} rho={s["rho"]*1e3:6.1f} a1={s["a1"]*1e3:6.1f}')
    print('  Table I target          c!=0  mpi= 139.9 mK= 420.5 eta= 531.9 eta\'= 965.4 sigma= 700  fpi=95.0 fK=106.9 rho=770.1 a1=1069.8')
    print('  Table I target          c=0   mpi= 139.9 mK= 420.5 eta= 140.0 eta\'= 644.9 sigma= 600')

    print('\n=== 4) eta-eta\' with printed c/2 vs Lagrangian c/sqrt2 off-diagonal (authors-like vacuum)')
    for coef, nm in [(0.5, 'printed c/2'), (1/r2, 'Lagrangian c/sqrt2')]:
        s = spectrum(0.6/5.5, f0.x[1], 'c!=0', etaNS_coef=coef)
        print(f'  {nm:20s}: eta={s["eta"]*1e3:.1f}  eta\'={s["etap"]*1e3:.1f}   (Table I: 531.9 / 965.4)')

    print('\n=== 5) Kaon hypotheses at authors-like vacuum (target 420.49 MeV)')
    sx, sy = 0.6/5.5, f0.x[1]; Z = Zs(sx, sy)
    base = spectrum(sx, sy, 'c!=0')['mK']/Z['K']
    print(f'  bracket only (Z_K=1)          : {base*1e3:.1f}')
    print(f'  Z_K from A13 (K1)  Z={Z["K"]:.4f}: {base*Z["K"]*1e3:.1f}')
    print(f'  Z_K^1 instead of Z_K^2        : {base*np.sqrt(Z["K"])*1e3:.1f}')
    print(f'  Z_K0* formula used  Z={Z["K0s"]:.4f}: {base*Z["K0s"]*1e3:.1f}')
    print(f'  required Z_K for 420.49       : {TAB["mK"]/base:.4f}')
    # sigma_y needed for kaon alone
    f = lambda y: spectrum(sx, y, 'c!=0')['mK']-TAB['mK']
    ys = np.linspace(0.05, 0.16, 400); vals = [f(y) for y in ys]
    roots = [brentq(f, ys[i], ys[i+1]) for i in range(len(ys)-1) if vals[i]*vals[i+1] < 0]
    print(f'  sigma_y that gives mK=420.49 with A13 Z_K: {np.round(roots,5)}')
    for y in roots:
        s0 = spectrum(sx, y, 'c=0'); s1 = spectrum(sx, y, 'c!=0')
        print(f'     at sy={y:.4f}: mpi(c!=0)={s1["mpi"]*1e3:.1f}, mpi(c=0)={s0["mpi"]*1e3:.1f}, eta\'(c=0)={s0["etap"]*1e3:.1f}, sigma(c=0)={s0["sigma"]*1e3:.1f}')

    print('\n=== 6) lambda2 from Eq.(42) with Table I fit values')
    fpi, fK, mpi, mK = TAB['fpi'], TAB['fK'], TAB['mpi'], TAB['mK']
    print('  (mK^2-mpi^2)/((2fK-fpi)(fK-fpi)) =', (mK**2-mpi**2)/((2*fK-fpi)*(fK-fpi)))
    print('  with exp. values                   =', (.4956**2-.1373**2)/((2*.1104-.0922)*(.1104-.0922)))
