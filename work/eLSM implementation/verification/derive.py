"""Independent numerical derivation of SU(3) (pseudo)scalar curvature masses
from the Lagrangian Eq.(2),(5):  V = m0^2 Tr(P'P) + l1 Tr(P'P)^2 + l2 Tr((P'P)^2)
                                    - c (det P + det P') - Tr[H (P + P')]
with P = sum_a T_a (sigma_a + i pi_a), T_a = lambda_a/2, T_0 = 1/sqrt(6) * 1.
Compared against Table A1 printed formulas (as coded in meson_masses.py)."""
import numpy as np

l = np.zeros((9, 3, 3), complex)
l[0] = np.sqrt(2/3)*np.eye(3)
l[1][0,1]=l[1][1,0]=1
l[2][0,1]=-1j; l[2][1,0]=1j
l[3][0,0]=1; l[3][1,1]=-1
l[4][0,2]=l[4][2,0]=1
l[5][0,2]=-1j; l[5][2,0]=1j
l[6][1,2]=l[6][2,1]=1
l[7][1,2]=-1j; l[7][2,1]=1j
l[8]=np.diag([1,1,-2])/np.sqrt(3)
T = l/2

def Phi(s, p):
    return np.einsum('a,aij->ij', s+1j*p, T)

def V(s, p, par):
    m0, l1, l2, c, hx, hy = par
    P = Phi(s, p); PP = P.conj().T @ P
    H = np.diag([hx/2, hx/2, hy/np.sqrt(2)])
    return (m0*np.trace(PP) + l1*np.trace(PP)**2 + l2*np.trace(PP@PP)
            - c*(np.linalg.det(P)+np.linalg.det(P.conj().T))
            - np.trace(H@(P+P.conj().T))).real

def vac(sx, sy):
    # sigma_x = sqrt(2/3) s0 + sqrt(1/3) s8 ; sigma_y = sqrt(1/3) s0 - sqrt(2/3) s8  (Eq.12)
    M = np.array([[np.sqrt(2), 1], [1, -np.sqrt(2)]])/np.sqrt(3)
    s0, s8 = np.linalg.solve(M, [sx, sy])
    s = np.zeros(9); s[0], s[8] = s0, s8
    return s

def hessian(f, x0, h=1e-4):
    n = len(x0); Hm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i = np.zeros(n); e_j = np.zeros(n); e_i[i]=h; e_j[j]=h
            Hm[i,j] = Hm[j,i] = (f(x0+e_i+e_j)-f(x0+e_i-e_j)-f(x0-e_i+e_j)+f(x0-e_i-e_j))/(4*h*h)
    return Hm

def curvature_masses(sx, sy, m0, l1, l2, c):
    hx = m0*sx - c/np.sqrt(2)*sx*sy + l1*sx*sy**2 + (2*l1+l2)/2*sx**3
    hy = m0*sy - c/(2*np.sqrt(2))*sx**2 + l1*sx**2*sy + (l1+l2)*sy**3
    par = (m0, l1, l2, c, hx, hy)
    s0 = vac(sx, sy)
    f_pi = lambda p: V(s0, p, par)
    f_si = lambda s: V(s, np.zeros(9), par)
    Hp = hessian(f_pi, np.zeros(9)); Hs = hessian(f_si, s0)
    # check stationarity in sigma
    g = [(f_si(s0+e*1e-6)-f_si(s0-e*1e-6))/2e-6 for e in np.eye(9)]
    R = np.array([[np.sqrt(2), 1], [1, -np.sqrt(2)]])/np.sqrt(3)  # (N,S) <- (0,8)
    def NS(Hm):
        B = Hm[np.ix_([0, 8], [0, 8])]
        return R @ B @ R.T
    return dict(grad=np.max(np.abs(g)), pi=Hp[1,1], K=Hp[4,4], a0=Hs[1,1], K0s=Hs[4,4],
                etaNS=NS(Hp), sigNS=NS(Hs), hx=hx, hy=hy)

def tableA1(sx, sy, m0, l1, l2, c):
    r2 = np.sqrt(2)
    return dict(
        pi=m0 - c/r2*sy + (l1+l2/2)*sx**2 + l1*sy**2,
        K=m0 - c/2*sx + (l1+l2/2)*sx**2 + (l1+l2)*sy**2 - l2/r2*sx*sy,
        a0=m0 + c/r2*sy + (l1+1.5*l2)*sx**2 + l1*sy**2,
        K0s=m0 + c/(2*r2)*sy + (l1+l2/2)*sx**2 + l2/r2*sx*sy + (l1+l2)*sy**2,
        etaN=m0 + c/r2*sy + (l1+l2/2)*sx**2 + l1*sy**2,
        etaS=m0 + l1*sx**2 + (l1+l2)*sy**2,
        etaNS=c/2*sx,
        sigN=m0 - c/r2*sy + 1.5*(2*l1+l2)*sx**2 + l1*sy**2,
        sigS=m0 + 3*(l1+l2)*sy**2 + l1*sx**2,
        sigNS=-c/r2*sx + 2*l1*sx*sy)

if __name__ == '__main__':
    sx, sy = 0.10909, 0.1035
    for name, par in [('c!=0', (-(0.258**2), 3.15, 36.253, 2.828)),
                      ('c=0', (-(0.282**2), -10.515, 55.576, 0.0))]:
        d = curvature_masses(sx, sy, *par); t = tableA1(sx, sy, *par)
        print(f'--- {name}: stationarity |grad| = {d["grad"]:.2e}')
        rows = [('pi', d['pi'], t['pi']), ('K', d['K'], t['K']), ('a0', d['a0'], t['a0']),
                ('K0*', d['K0s'], t['K0s']),
                ('etaN', d['etaNS'][0,0], t['etaN']), ('etaS', d['etaNS'][1,1], t['etaS']),
                ('etaNS', d['etaNS'][0,1], t['etaNS']),
                ('sigN', d['sigNS'][0,0], t['sigN']), ('sigS', d['sigNS'][1,1], t['sigS']),
                ('sigNS', d['sigNS'][0,1], t['sigNS'])]
        for n, a, b in rows:
            flag = '' if abs(a-b) < 1e-5 else '   <-- MISMATCH (ratio %.4f)' % (a/b if b else np.nan)
            print(f'{n:6s} Lagrangian={a: .6f}  TableA1={b: .6f}{flag}')
        print('Goldstone K check: (hx+sqrt2 hy)/(sx+sqrt2 sy) =',
              (d['hx']+np.sqrt(2)*d['hy'])/(sx+np.sqrt(2)*sy), ' hx/sx =', d['hx']/sx)
