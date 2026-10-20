"""Numerical minimization utilities for the SU(3) eLSM mean fields."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class MinimumResult:
    x: np.ndarray
    omega: float
    success: bool
    message: str
    grad_norm: float
    nfev: int
    method: str


def minimize_multistart(objective, starts, bounds=None, method="L-BFGS-B",
                        grad=None, grad_tol=1e-5):
    """Run a local minimizer from multiple initial points and retain diagnostics."""
    results = []
    for x0 in starts:
        res = minimize(objective, np.asarray(x0, dtype=float), jac=grad,
                       method=method, bounds=bounds,
                       options={"maxiter": 5000, "ftol": 1e-12, "gtol": grad_tol})
        g = np.asarray(grad(res.x), dtype=float) if grad is not None else np.full_like(res.x, np.nan)
        results.append(MinimumResult(
            x=np.asarray(res.x), omega=float(res.fun), success=bool(res.success),
            message=str(res.message),
            grad_norm=float(np.linalg.norm(g)) if grad is not None else float("nan"),
            nfev=int(res.nfev), method=method))
    good = [r for r in results if r.success and np.isfinite(r.omega)]
    if not good:
        raise RuntimeError("No successful minimization run: " +
                           "; ".join(f"{r.message}, Omega={r.omega}" for r in results))
    best = min(good, key=lambda r: r.omega)
    return best, results


def minimize_temperature_scan(objective_factory, temperatures, starts_factory,
                              bounds=None, grad_factory=None, method="L-BFGS-B"):
    """Minimize sequentially over T, seeding each T with the previous solution.

    objective_factory(T) returns objective(x); grad_factory(T) optionally returns
    gradient(x). starts_factory(T, previous_x) returns a list of starts and should
    include physically motivated alternatives to reduce branch-tracking failures.
    """
    output = []
    previous = None
    for T in temperatures:
        starts = list(starts_factory(float(T), previous))
        if previous is not None:
            starts.insert(0, previous.copy())
        grad = grad_factory(float(T)) if grad_factory is not None else None
        best, runs = minimize_multistart(objective_factory(float(T)), starts,
                                         bounds=bounds, grad=grad, method=method)
        output.append((float(T), best, runs))
        previous = best.x.copy()
    return output


def solve_mean_fields(T, p, polyakov_parameters, *, mu=0.0, starts=None,
                     bounds=((0.0, 0.6), (0.0, 0.8), (0.0, 0.999), (0.0, 0.999)),
                     vacuum_prescription="omit", vacuum_cutoff=None,
                     method="L-BFGS-B"):
    """Minimize Omega(T) over (sigma_x, sigma_y, phi, phi_bar).

    Bounds are numerical guardrails, not source-derived physical constraints;
    adjust and document them for the target calculation. The Polyakov logarithm
    domain and all model terms are evaluated by `grand_potential`.
    """
    from .potential import grand_potential
    if starts is None:
        starts = [
            (0.10, 0.14, 0.05, 0.05),
            (0.08, 0.12, 0.25, 0.25),
            (0.12, 0.16, 0.60, 0.60),
            (0.05, 0.10, 0.85, 0.85),
        ]
    def objective(x):
        try:
            value = grand_potential(
                T, x[0], x[1], x[2], x[3], p,
                polyakov_parameters=polyakov_parameters, mu=mu,
                vacuum_prescription=vacuum_prescription,
                vacuum_cutoff=vacuum_cutoff)
            return float(value) if np.isfinite(value) else 1e100
        except (ValueError, FloatingPointError, OverflowError):
            return 1e100
    return minimize_multistart(objective, starts, bounds=bounds, method=method)
