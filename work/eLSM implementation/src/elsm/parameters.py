"""Parameter objects for the SU(3) eLSM reference implementation.

All dimensional quantities are in GeV or powers of GeV.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SU3Parameters:
    m0_sq: float
    lambda1: float
    lambda2: float
    c_anomaly: float
    h_x: float
    h_y: float
    g_yukawa: float = 5.5
    Tc: float = 0.270
    include_FT: bool = False

    def effective_m0_sq(self, T: float) -> float:
        if T < 0:
            raise ValueError("Temperature T must be non-negative")
        if not self.include_FT:
            return self.m0_sq
        if self.Tc <= 0:
            raise ValueError("Tc must be positive")
        return self.m0_sq * (1.0 - (T / self.Tc) ** 2)

    @classmethod
    def from_vacuum_condensates(
        cls, *, m0_sq: float, lambda1: float, lambda2: float,
        c_anomaly: float, sigma_x: float, sigma_y: float,
        g_yukawa: float = 5.5, Tc: float = 0.270,
        include_FT: bool = False,
    ) -> "SU3Parameters":
        """Construct h_x,h_y from the T=0 stationarity equations (Eqs. 15–16).

        Condensates must be in GeV. This computes the source-defined tree-level
        explicit-breaking parameters; it does not fit/validate the condensates.
        """
        import numpy as np
        sx, sy = float(sigma_x), float(sigma_y)
        hx = (m0_sq*sx - c_anomaly/np.sqrt(2)*sx*sy
              + lambda1*sx*sy**2 + (2*lambda1+lambda2)/2*sx**3)
        hy = (m0_sq*sy - c_anomaly/(2*np.sqrt(2))*sx**2
              + lambda1*sx**2*sy + (lambda1+lambda2)*sy**3)
        return cls(m0_sq=m0_sq, lambda1=lambda1, lambda2=lambda2,
                   c_anomaly=c_anomaly, h_x=float(hx), h_y=float(hy),
                   g_yukawa=g_yukawa, Tc=Tc, include_FT=include_FT)


@dataclass(frozen=True)
class PolyakovParameters:
    """Roessner–Ratti–Weise logarithmic SU(3) Polyakov potential.

    Coefficients are the standard logarithmic set cited by the main paper's
    Ref. [25]: a0=3.51, a1=-2.47, a2=15.2, b3=-1.75. T0 remains configurable.
    """
    T0: float = 0.270
    a0: Optional[float] = 3.51
    a1: Optional[float] = -2.47
    a2: Optional[float] = 15.2
    b3: Optional[float] = -1.75

    def validate(self):
        if self.T0 <= 0:
            raise ValueError("T0 must be positive")
        missing = [k for k in ("a0", "a1", "a2", "b3")
                   if getattr(self, k) is None]
        if missing:
            raise ValueError("Polyakov coefficients not configured: " + ", ".join(missing))
