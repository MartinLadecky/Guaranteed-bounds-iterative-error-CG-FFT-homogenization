"""
Preconditioned conjugate gradients (Algorithm 1 of the paper) with estimates of
the energy norm of the iterative error

    e_k = ||u_{h,k} - u_h||_K^2   ( = U_k - U_h, the error in the effective energy ).

At every iteration k the history stores
    rr       squared residual norm             ||r_k||^2
    rz       squared preconditioned residual   ||r_k||_G^2 = (r_k, G r_k)
    GR       Gauss-Radau upper bound           mu_k ||r_k||_G^2
    Delta    alpha_k ||r_k||_G^2, the decrease  e_k - e_{k+1}
and, after the run, the PT lower bound LB[l] of e_l with the adaptive delay
(Meurant, Papez, Tichy 2021) together with the iteration at which it became
available, LB_at[l].  The PT upper estimate is LB[l] / (1 - tau).
"""
import numpy as np


class AdaptiveDelay:
    """PT lower bound e_l >= Delta_l + ... + Delta_{k-1} with the adaptive delay d = k - l.

    The bound for the iterate l is accepted at iteration k when the estimated
    relative error of the bound,  S Delta_{k-1} / (Delta_l + ... + Delta_{k-2}),
    is below tau (Appendix B of the paper), and, if d_min > 0, the delay k - l is
    at least d_min."""

    def __init__(self, tau=0.25, d_min=0):
        self.tau, self.d_min = tau, d_min
        self.Delta = []            # Delta_0, Delta_1, ...
        self.tail = []             # tail[j] = Delta_j + ... + Delta_{k-1}
        self.l = 0                 # next iterate without a bound
        self.LB, self.LB_at = {}, {}

    def update(self, Delta):
        self.Delta.append(Delta)
        self.tail = [t + Delta for t in self.tail] + [Delta]
        k = len(self.Delta)
        if k < 3:
            return
        tail = np.array(self.tail)
        # safety factor S from the iterations with the last four orders of decrease
        recent = np.flatnonzero(tail[self.l] / tail <= 1e-4)
        m = min(recent[-1], k - 2) if recent.size else 0
        S = np.max(tail[m:-1] / np.array(self.Delta[m:-1]))
        while self.l <= k - 3:                              # delay k - l >= 3
            den = tail[self.l] - Delta                      # Delta_l + ... + Delta_{k-2}
            if not (den > 0 and S * Delta / den <= self.tau and k - self.l >= self.d_min):
                break
            self.LB[self.l] = tail[self.l]                  # Delta_l + ... + Delta_{k-1}
            self.LB_at[self.l] = k
            self.l += 1


def pcg(K, b, G, lam_min, x0=None, tau=0.25, d_min=0, tol=1e-10,
        stop_on=("rr", "trivial", "GR", "PT_U"), maxiter=10000, u_ref=None):
    """PCG for K u = b with the preconditioner G.

    K, G     functions acting on vectors
    lam_min  guaranteed lower bound on the smallest eigenvalue of G K
    stop_on  PCG stops when every listed quantity is <= tol:
             'rr' ||r_k||^2, 'trivial' ||r_k||_G^2 / lam_min, 'GR', 'PT_U'
    u_ref    reference solution u_h; if given, the iterative error e_k is stored
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - K(x)
    z = G(r)
    p = z.copy()
    rz = np.vdot(r, z)
    mu = 1 / lam_min                                  # Gauss-Radau: mu_0 = 1 / mu_min
    pt = AdaptiveDelay(tau, d_min)
    h = {key: [] for key in ("rr", "rz", "GR", "Delta", "error")}

    def record(x, r, rz, mu):
        h["rr"].append(np.vdot(r, r))
        h["rz"].append(rz)
        h["GR"].append(mu * rz)
        if u_ref is not None:
            e = x - u_ref
            h["error"].append(np.vdot(e, K(e)))

    def converged():
        current = {"rr": h["rr"][-1], "trivial": h["rz"][-1] / lam_min, "GR": h["GR"][-1],
                   "PT_U": pt.LB[pt.l - 1] / (1 - tau) if pt.l > 0 else np.inf}
        return all(current[c] <= tol for c in stop_on)

    record(x, r, rz, mu)
    for k in range(maxiter):
        Kp = K(p)
        alpha = rz / np.vdot(p, Kp)                   # step length
        x += alpha * p                                # update solution
        r -= alpha * Kp                               # update residual
        z = G(r)                                      # apply preconditioner
        rz_new = np.vdot(r, z)
        beta = rz_new / rz
        p = z + beta * p                              # new search direction

        h["Delta"].append(alpha * rz)                 # e_k - e_{k+1}
        pt.update(alpha * rz)
        mu = (mu - alpha) / (lam_min * (mu - alpha) + beta)   # Gauss-Radau recursion
        rz = rz_new
        record(x, r, rz, mu)
        if converged():
            break

    h = {key: np.array(v) for key, v in h.items()}
    h["LB"], h["LB_at"] = pt.LB, pt.LB_at
    h["UE"] = {l: v / (1 - tau) for l, v in pt.LB.items()}
    return x, h
