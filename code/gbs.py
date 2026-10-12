"""
Core library for the Paris-coupled generalised Birnbaum-Saunders (GBS)
degradation process and its sequential Monte Carlo inference.

Contents
--------
* Birnbaum-Saunders and BS-Student-t densities, samplers, quantiles
* Increment laws (BS, BS-t, Wiener, Gamma, Inverse Gaussian) behind one
  interface, so that every competing degradation process can be driven by the
  *same* Liu-West particle filter -- a controlled comparison
* Liu-West particle filter with joint state/parameter learning
* RUL prediction: exact single-step, IG approximation, Monte Carlo
* Epistemic / aleatory variance decomposition
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special, stats

SQRT2PI = np.sqrt(2.0 * np.pi)
LOG_SQRT2PI = 0.5 * np.log(2.0 * np.pi)


# ══════════════════════════════════════════════════════════════════════════
# 1.  Birnbaum-Saunders distribution
# ══════════════════════════════════════════════════════════════════════════
def xi(u):
    """xi(u) = sqrt(u) - 1/sqrt(u)."""
    s = np.sqrt(u)
    return s - 1.0 / s


def xi_prime(u):
    """d/du xi(u) = (u^{-1/2} + u^{-3/2}) / 2."""
    return 0.5 * (u ** -0.5 + u ** -1.5)


def bs_pdf(t, alpha, beta):
    """f(t) = xi'(t/beta) / (alpha*beta) * phi(xi(t/beta)/alpha).

    NOTE the normalising constant is alpha*beta, not 2*alpha*beta: xi is a
    strictly increasing bijection (0,inf) -> R, so the change of variable has a
    single branch.  Combined with xi'(u) = (u^{-1/2}+u^{-3/2})/2 this recovers
    the textbook form  [(t/b)^{-1/2}+(t/b)^{-3/2}] / (2*a*b*sqrt(2pi)) * exp(.).
    """
    t = np.asarray(t, dtype=float)
    u = t / beta
    return xi_prime(u) / (alpha * beta) * np.exp(-0.5 * (xi(u) / alpha) ** 2) / SQRT2PI


def bs_logpdf(t, alpha, beta):
    """log f_BS(t; alpha, beta); -inf outside the support."""
    t = np.asarray(t, dtype=float)
    out = np.full(np.broadcast(t, alpha, beta).shape, -np.inf, dtype=float)
    ok = t > 0
    if not np.any(ok):
        return out
    u = np.where(ok, t, 1.0) / beta
    z = xi(u) / alpha
    out = np.where(
        ok,
        np.log(xi_prime(u)) - np.log(alpha * beta) - 0.5 * z ** 2 - LOG_SQRT2PI,
        -np.inf,
    )
    return out


def bs_cdf(t, alpha, beta):
    t = np.asarray(t, dtype=float)
    return np.where(t > 0, stats.norm.cdf(xi(np.where(t > 0, t, 1.0) / beta) / alpha), 0.0)


def bs_ppf(p, alpha, beta):
    """Closed-form quantile function (Prop. 'Closed-Form Quantile Function')."""
    z = stats.norm.ppf(p)
    return beta / 4.0 * (alpha * z + np.sqrt(4.0 + (alpha * z) ** 2)) ** 2


def bs_rvs(alpha, beta, size=None, rng=None):
    """O(1) exact sampler:  T = (beta/4)[aZ + sqrt(4 + a^2 Z^2)]^2."""
    rng = rng or np.random.default_rng()
    z = rng.standard_normal(size if size is not None else np.shape(alpha))
    return beta / 4.0 * (alpha * z + np.sqrt(4.0 + (alpha * z) ** 2)) ** 2


def bs_mean(alpha, beta):
    return beta * (1.0 + alpha ** 2 / 2.0)


def bs_var(alpha, beta):
    return beta ** 2 * alpha ** 2 * (1.0 + 1.25 * alpha ** 2)


# ── BS-Student-t (generalised BS with g = t_nu) ─────────────────────────────
def bst_logpdf(t, alpha, beta, nu):
    t = np.asarray(t, dtype=float)
    ok = t > 0
    u = np.where(ok, t, 1.0) / beta
    z = xi(u) / alpha
    lc = (special.gammaln((nu + 1) / 2.0) - special.gammaln(nu / 2.0)
          - 0.5 * np.log(nu * np.pi))
    lp = (np.log(xi_prime(u)) - np.log(alpha * beta) + lc
          - (nu + 1) / 2.0 * np.log1p(z ** 2 / nu))
    return np.where(ok, lp, -np.inf)


def bst_rvs(alpha, beta, nu, size=None, rng=None):
    """Scale-mixture sampler:  U ~ Gamma(nu/2, nu/2);  T | U ~ BS(alpha/sqrt(U), beta)."""
    rng = rng or np.random.default_rng()
    shape = size if size is not None else np.shape(alpha)
    u = rng.gamma(nu / 2.0, 2.0 / nu, size=shape)
    return bs_rvs(alpha / np.sqrt(u), beta, size=shape, rng=rng)


# ── Inverse Gaussian helpers (RUL approximation) ────────────────────────────
def ig_cdf(t, mu, lam):
    t = np.asarray(t, dtype=float)
    ok = t > 0
    tt = np.where(ok, t, 1.0)
    a = np.sqrt(lam / tt) * (tt / mu - 1.0)
    b = -np.sqrt(lam / tt) * (tt / mu + 1.0)
    # exp(2 lam/mu) * Phi(b) computed in log space to avoid overflow
    logterm = 2.0 * lam / mu + stats.norm.logcdf(b)
    return np.where(ok, stats.norm.cdf(a) + np.exp(np.clip(logterm, -700, 700)), 0.0)


def ig_ppf(p, mu, lam, lo=1e-9, hi=None, iters=80):
    """Quantiles of IG by monotone bisection (vectorised over mu, lam)."""
    mu = np.asarray(mu, dtype=float)
    lam = np.asarray(lam, dtype=float)
    hi = hi if hi is not None else 50.0 * mu + 1.0
    lo = np.full_like(mu, lo)
    hi = np.broadcast_to(np.asarray(hi, dtype=float), mu.shape).copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = ig_cdf(mid, mu, lam)
        lo = np.where(c < p, mid, lo)
        hi = np.where(c < p, hi, mid)
    return 0.5 * (lo + hi)


def ig_rvs(mu, lam, size=None, rng=None):
    """Michael-Schucany-Haas sampler (numerically guarded)."""
    rng = rng or np.random.default_rng()
    shape = size if size is not None else np.shape(mu)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-300, 1e300)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-300, 1e300)
    y = rng.standard_normal(shape) ** 2
    # x = mu + mu^2 y/(2 lam) - (mu/(2 lam)) sqrt(4 mu lam y + mu^2 y^2), rewritten
    # as mu*(1 + c - sqrt(c(c+2))) with c = mu y/(2 lam) to avoid overflow
    c = np.clip(mu * y / (2.0 * lam), 0.0, 1e150)
    x = mu * np.maximum(1.0 + c - np.sqrt(c * (c + 2.0)), 1e-300)
    z = rng.random(shape)
    out = np.where(z <= mu / (mu + x), x, mu ** 2 / np.maximum(x, 1e-300))
    return np.where(np.isfinite(out), out, mu)


# ══════════════════════════════════════════════════════════════════════════
# 2.  Increment laws -- one interface, five processes
# ══════════════════════════════════════════════════════════════════════════
class IncrementLaw:
    """Degradation increment  dX ~ Law(theta, scale)  where `scale` absorbs
    the time step and (for the Paris-coupled variant) the current state."""

    name = "abstract"
    param_names: tuple[str, ...] = ()

    def sample(self, th, scale, rng):
        raise NotImplementedError

    def mean(self, th, scale):
        raise NotImplementedError

    def var(self, th, scale):
        raise NotImplementedError

    def logpdf(self, dx, th, scale):
        """log density of the increment; required by the guided proposal."""
        raise NotImplementedError


def bs_consistent_params(alpha, beta, s):
    """Grid-consistent (moment-invariant) BS increment parameters.

    The obvious scaling  dX ~ BS(alpha, beta*s)  is NOT a valid increment law:
    BS is a scale family, so BS(alpha, beta*s) = s * BS(alpha, beta) and hence
        E[dX] = s * m,     Var[dX] = s^2 * v.
    Summing n = T/s such increments gives Var[X(T)] = n s^2 v = T v s, which
    vanishes as the inspection grid is refined -- the process depends on how
    often one happens to measure.

    Instead solve for (alpha_s, beta_s) making BOTH moments additive,
        beta_s (1 + alpha_s^2/2)                     = m s,
        beta_s^2 alpha_s^2 (1 + 5 alpha_s^2/4)       = v s,
    with m, v the reference (s = 1) mean and variance.  Writing A = alpha_s^2 and
    r = v / (m^2 s), the ratio of the two equations is the quadratic

        (5/4 - r/4) A^2 + (1 - r) A - r = 0,

    whose discriminant collapses to 16(1 + 3r), giving the closed root

        A = 2 r (u + 4) / [ (5 - r)(u + 1) ],    u = sqrt(1 + 3r),

    written in the form that avoids cancellation at small r.  Then
    beta_s = m s / (1 + A/2).  s = 1
    returns (alpha, beta) exactly.  The first-passage results depend on the
    increments only through m and v, so they become grid-invariant as well.
    """
    m = bs_mean(alpha, beta)
    v = bs_var(alpha, beta)
    s = np.asarray(s, dtype=float)
    r = v / np.maximum(m ** 2 * s, 1e-300)
    # g(A) = A(1+5A/4)/(1+A/2)^2 increases from 0 to 5, so a solution exists only
    # for r < 5: no BS law has a squared coefficient of variation of 5 or more.
    # Steps so short (or couplings so negative) that they would demand one are
    # clamped to the admissible boundary rather than returning a complex root.
    r = np.minimum(r, 5.0 - 1e-9)
    # The quadratic (5/4 - r/4)A^2 + (1-r)A - r = 0 has discriminant
    # 16(1-r)^2 + 16r(5-r) = 16(1+3r) -- the r^2 terms cancel -- so
    #     A = 2(r - 1 + sqrt(1+3r)) / (5 - r).
    # Written that way it subtracts two near-equal quantities and loses six per
    # cent of the answer at r ~ 1e-9, as does solving the quadratic directly.
    # Rationalising with u = sqrt(1+3r) gives r - 1 + u = r(u+4)/(u+1), which
    # has no such subtraction and shows A ~ r as r -> 0:
    u = np.sqrt(1.0 + 3.0 * r)
    A = 2.0 * r * (u + 4.0) / ((5.0 - r) * (u + 1.0))
    A = np.clip(np.nan_to_num(A, nan=1e-12), 1e-12, 1e12)
    return np.sqrt(A), np.maximum(m * s / (1.0 + A / 2.0), 1e-300)


class BSLaw(IncrementLaw):
    """Birnbaum-Saunders increments in the grid-consistent parameterisation."""
    name = "BS"
    param_names = ("alpha", "beta")

    def __init__(self, consistent: bool = True):
        self.consistent = consistent

    def _ab(self, th, scale):
        if self.consistent:
            return bs_consistent_params(th["alpha"], th["beta"], scale)
        return th["alpha"], th["beta"] * scale

    def sample(self, th, scale, rng):
        a, b = self._ab(th, scale)
        return bs_rvs(a, b, size=np.shape(th["alpha"]), rng=rng)

    def mean(self, th, scale):
        a, b = self._ab(th, scale)
        return bs_mean(a, b)

    def var(self, th, scale):
        a, b = self._ab(th, scale)
        return bs_var(a, b)

    def logpdf(self, dx, th, scale):
        a, b = self._ab(th, scale)
        return bs_logpdf(dx, a, b)


class BSTLaw(IncrementLaw):
    """BS-Student-t increments (the Diaz-Garcia & Leiva generalised BS family
    with g = t_nu), in the same grid-consistent parameterisation."""
    name = "BS-t"
    param_names = ("alpha", "beta")

    def __init__(self, nu: float = 4.0, consistent: bool = True):
        self.nu = nu
        self.consistent = consistent
        # E[U^{-1}] = nu/(nu-2) with U ~ Gamma(nu/2, nu/2): the scale mixture
        # inflates the effective BS shape by this factor (finite for nu > 2)
        self.infl = np.sqrt(nu / (nu - 2.0))

    def _ab(self, th, scale):
        a = th["alpha"] * self.infl
        if self.consistent:
            return bs_consistent_params(a, th["beta"], scale)
        return a, th["beta"] * scale

    def sample(self, th, scale, rng):
        a, b = self._ab(th, scale)
        return bst_rvs(a / self.infl, b, self.nu, size=np.shape(th["alpha"]), rng=rng)

    def mean(self, th, scale):
        a, b = self._ab(th, scale)
        return bs_mean(a, b)

    def var(self, th, scale):
        a, b = self._ab(th, scale)
        return bs_var(a, b)

    def logpdf(self, dx, th, scale):
        a, b = self._ab(th, scale)
        return bst_logpdf(dx, a / self.infl, b, self.nu)


class WienerLaw(IncrementLaw):
    """dX ~ N(mu*scale, sigma^2*scale) -- non-monotone."""
    name = "Wiener"
    param_names = ("mu", "sigma")

    def sample(self, th, scale, rng):
        return th["mu"] * scale + th["sigma"] * np.sqrt(scale) * rng.standard_normal(
            np.shape(th["mu"]))

    def mean(self, th, scale):
        return th["mu"] * scale

    def var(self, th, scale):
        return th["sigma"] ** 2 * scale

    def logpdf(self, dx, th, scale):
        sd = np.maximum(th["sigma"] * np.sqrt(scale), 1e-300)
        return -0.5 * ((dx - th["mu"] * scale) / sd) ** 2 - np.log(sd) - LOG_SQRT2PI


class GammaLaw(IncrementLaw):
    """dX ~ Gamma(shape = eta*scale, rate = lam) -- monotone."""
    name = "Gamma"
    param_names = ("eta", "lam")

    def sample(self, th, scale, rng):
        return rng.gamma(np.maximum(th["eta"] * scale, 1e-8), 1.0 / th["lam"])

    def mean(self, th, scale):
        return th["eta"] * scale / th["lam"]

    def var(self, th, scale):
        return th["eta"] * scale / th["lam"] ** 2

    def logpdf(self, dx, th, scale):
        k = np.maximum(th["eta"] * scale, 1e-8)
        lam = np.maximum(th["lam"], 1e-300)
        ok = dx > 0
        x = np.where(ok, dx, 1.0)
        return np.where(ok, k * np.log(lam) + (k - 1) * np.log(x) - lam * x
                        - special.gammaln(k), -np.inf)


class IGLaw(IncrementLaw):
    """dX ~ IG(mu*scale, lam*scale^2) -- monotone."""
    name = "IGproc"
    param_names = ("mu", "lam")

    def sample(self, th, scale, rng):
        return ig_rvs(th["mu"] * scale, th["lam"] * scale ** 2, rng=rng)

    def mean(self, th, scale):
        return th["mu"] * scale

    def var(self, th, scale):
        return (th["mu"] * scale) ** 3 / (th["lam"] * scale ** 2)

    def logpdf(self, dx, th, scale):
        mu = np.maximum(th["mu"] * scale, 1e-300)
        lam = np.maximum(th["lam"] * scale ** 2, 1e-300)
        ok = dx > 0
        x = np.where(ok, dx, 1.0)
        return np.where(ok, 0.5 * np.log(lam) - LOG_SQRT2PI - 1.5 * np.log(x)
                        - lam * (x - mu) ** 2 / (2.0 * mu ** 2 * x), -np.inf)


LAWS = {"GBS": BSLaw, "GBS-t": BSTLaw, "Wiener": WienerLaw,
        "Gamma": GammaLaw, "IGproc": IGLaw}


# ══════════════════════════════════════════════════════════════════════════
# 3.  Liu-West particle filter
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class PFConfig:
    n_particles: int = 1000
    delta: float = 0.98              # Liu-West discount
    ess_frac: float = 0.5
    gamma_coupling: bool = False     # Paris-law state coupling  beta_k ∝ (x/x_ref)^gamma
    x_ref: float = 0.1
    seed: int = 0
    sigma_floor: float = 1e-4
    obs: str = "gauss"               # "gauss" or "t" (Student-t observation noise)
    nu_obs: float = 4.0              # d.o.f. of the Student-t observation density
    x_max: float = np.inf            # cap on the latent state
    x_couple_max: float = np.inf     # damage level beyond which the coupling saturates
    s_max: float = 1e6               # cap on the state-dependent scale factor
    jitter_floor: float = 0.05       # roughening floor, as a fraction of prior sd
    proposal: str = "guided"         # "bootstrap" or "guided" (see _guided)


@dataclass
class PFState:
    x: np.ndarray                    # (N,)   latent degradation
    logth: np.ndarray                # (N,P)  log-parameters
    logw: np.ndarray                 # (N,)   log-weights (normalised)
    names: tuple                     # parameter names in logth column order
    ess_hist: list = field(default_factory=list)
    x_hist: list = field(default_factory=list)


def systematic_resample(w, rng):
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), pos).clip(0, n - 1)


class ParticleFilter:
    """Bootstrap particle filter with Liu-West kernel smoothing on log-parameters.

    State model  : X_k = X_{k-1} + dX_k,   dX_k ~ Law(theta, s_k)
    Observation  : Y_k = X_k + eps_k,      eps_k ~ N(0, sigma_eps^2)
    where s_k = (dt_k / dt_ref) * (max(X_{k-1}, x_ref) / x_ref)^gamma.
    """

    def __init__(self, law: IncrementLaw, cfg: PFConfig):
        self.law = law
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.names = tuple(law.param_names) + ("sigma_eps",)
        if cfg.gamma_coupling:
            self.names = self.names + ("gamma",)

    # ── initialisation ────────────────────────────────────────────────────
    def init(self, prior_logmean: dict, prior_logsd: dict, x0: float) -> PFState:
        n = self.cfg.n_particles
        cols = []
        for nm in self.names:
            cols.append(prior_logmean[nm] + prior_logsd[nm] * self.rng.standard_normal(n))
        logth = np.column_stack(cols)
        self._prior_sd = np.array([prior_logsd[nm] for nm in self.names], dtype=float)
        return PFState(x=np.full(n, float(x0)), logth=logth,
                       logw=np.full(n, -np.log(n)), names=self.names)

    def _theta(self, logth):
        th = {nm: np.exp(np.clip(logth[:, j], -30, 30))
              for j, nm in enumerate(self.names)}
        if self.cfg.gamma_coupling:
            # gamma is a *signed* exponent, stored as log(1 + gamma).  It is
            # bounded to a physically meaningful range: gamma = m/2 with the
            # Paris exponent m in (0, 16] covers every engineering alloy, and an
            # unbounded gamma lets q(x)^gamma overflow and destroys the Lamperti
            # transform long before it fits anything.
            g = np.exp(np.clip(logth[:, self.names.index("gamma")], -30, 30)) - 1.0
            th["gamma"] = np.clip(g, -0.9, 8.0)
        return th

    def _scale(self, x_prev, dt, dt_ref, th):
        """Time- and state-dependent scale  s_k = (dt/dt_ref) * q(x)^gamma  with
        q(x) = 1 + x/x_ref.  gamma = 0 recovers the homogeneous BS process;
        gamma = m/2 is the Paris-Erdogan coupling (da/dN proportional to a^{m/2})."""
        s = dt / dt_ref
        if self.cfg.gamma_coupling:
            # For gamma > 1 the coupled ODE blows up in finite time, which is the
            # intended physics but overflows in floating point; the factor is
            # capped so that a diverging particle degenerates gracefully into an
            # "already failed" particle instead of producing NaN.
            # The coupling saturates at the failure threshold. For gamma > 1 the
            # coupled drift blows up in finite time, which is the intended
            # physics of unstable spall growth but is meaningless beyond D: the
            # first-passage time to D never depends on the dynamics past D. Left
            # unsaturated, a bootstrap proposal that overshoots D by orders of
            # magnitude gives every particle an equally hopeless weight and the
            # filter loses the state altogether.
            q = 1.0 + np.clip(x_prev, 0.0, self.cfg.x_couple_max) / self.cfg.x_ref
            s = s * np.clip(np.power(q, th["gamma"]), 1e-12, self.cfg.s_max)
        return s

    # ── guided proposal ───────────────────────────────────────────────────
    def _guided(self, x_prev, y, th, s, sig):
        """Sample the increment from an observation-aware proposal.

        A bootstrap filter proposes dX from the transition alone. With strictly
        positive increments and a flat health indicator the cloud drifts above
        the data within a few hundred steps and cannot come back: every particle
        is then equally implausible and resampling cannot repair it. We instead
        match the transition's first two moments with a Gaussian, multiply by the
        Gaussian likelihood -- a conjugate product -- and sample from the result
        truncated to dX > 0:

            v* = (1/v_t + 1/sigma^2)^{-1},
            m* = v* ( m_t/v_t + (y - x_{k-1})/sigma^2 ),
            q  = N(m*, v*) restricted to (0, inf).

        The proposal is only an approximation, so the exact transition density is
        restored in the importance weight; the filter remains unbiased for any
        increment law.
        """
        m_t = np.maximum(self.law.mean(th, s), 1e-300)
        v_t = np.maximum(self.law.var(th, s), 1e-300)
        v_star = 1.0 / (1.0 / v_t + 1.0 / sig ** 2)
        m_star = v_star * (m_t / v_t + (y - x_prev) / sig ** 2)
        sd = np.sqrt(v_star)

        # inverse-CDF sampling from N(m*, v*) truncated to (0, inf)
        a = -m_star / sd
        lo = special.ndtr(a)                      # faster than stats.norm.cdf
        u = lo + (1.0 - lo) * self.rng.random(np.shape(m_star))
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        dx = np.maximum(m_star + sd * special.ndtri(u), 1e-300)

        log_q = (-0.5 * ((dx - m_star) / sd) ** 2 - np.log(sd) - LOG_SQRT2PI
                 - np.log(np.maximum(1.0 - lo, 1e-300)))
        return dx, log_q

    # ── one filtering step ────────────────────────────────────────────────
    def step(self, st: PFState, y, dt, dt_ref) -> PFState:
        cfg = self.cfg
        w = np.exp(st.logw - st.logw.max())
        w /= w.sum()

        # Liu-West kernel shrinkage in log-parameter space
        a = (3 * cfg.delta - 1) / (2 * cfg.delta)
        h2 = 1 - a ** 2
        mbar = w @ st.logth
        dev = st.logth - mbar
        V = (dev * w[:, None]).T @ dev
        # Roughening floor.  The Liu-West kernel jitters by h^2 V, so once
        # repeated resampling drives V to zero the collapse is absorbing: no
        # further jitter is injected and the parameter particles freeze.  A floor
        # proportional to the prior spread and decaying as k^{-1/2} keeps the
        # cloud alive while still allowing the posterior to contract.
        k = len(st.ess_hist) + 1
        floor = (cfg.jitter_floor * getattr(self, "_prior_sd", 1.0)) ** 2 / np.sqrt(k)
        V = V + np.diag(np.atleast_1d(floor) * np.ones(V.shape[0])) \
              + 1e-12 * np.eye(V.shape[0])
        m = a * st.logth + (1 - a) * mbar
        try:
            L = np.linalg.cholesky(h2 * V)
        except np.linalg.LinAlgError:
            L = np.sqrt(h2 * np.diag(V))[None, :] * np.eye(V.shape[0])
        logth_new = m + self.rng.standard_normal(st.logth.shape) @ L.T
        logth_new = np.clip(logth_new, -30, 30)

        th = self._theta(logth_new)
        s = self._scale(st.x, dt, dt_ref, th)
        sig = np.maximum(th["sigma_eps"], cfg.sigma_floor)

        if cfg.proposal == "guided":
            dx, log_q = self._guided(st.x, y, th, s, sig)
        else:
            dx = self.law.sample(th, s, self.rng)
            log_q = None
        dx = np.where(np.isfinite(dx), dx, 0.0)
        x_new = np.clip(st.x + dx, -np.inf, cfg.x_max)

        r = (y - x_new) / sig
        if cfg.obs == "t":
            nu = cfg.nu_obs
            loglik = (special.gammaln((nu + 1) / 2.0) - special.gammaln(nu / 2.0)
                      - 0.5 * np.log(nu * np.pi) - np.log(sig)
                      - (nu + 1) / 2.0 * np.log1p(r ** 2 / nu))
        else:
            loglik = -0.5 * r ** 2 - np.log(sig) - LOG_SQRT2PI
        if log_q is not None:
            # importance weight for the guided proposal:
            #   w ∝ f_transition(dx) * g(y | x_{k-1}+dx) / q(dx)
            lp = self.law.logpdf(dx, th, s)
            loglik = loglik + np.where(np.isfinite(lp), lp, -1e10) - log_q
        logw = st.logw + loglik
        logw -= logw.max()
        wn = np.exp(logw)
        wn /= wn.sum()

        ess = 1.0 / np.sum(wn ** 2)
        if ess < cfg.ess_frac * cfg.n_particles:
            idx = systematic_resample(wn, self.rng)
            x_new, logth_new = x_new[idx], logth_new[idx]
            wn = np.full(cfg.n_particles, 1.0 / cfg.n_particles)

        st2 = PFState(x=x_new, logth=logth_new,
                      logw=np.log(np.maximum(wn, 1e-300)), names=self.names,
                      ess_hist=st.ess_hist + [ess], x_hist=st.x_hist)
        return st2

    def run(self, y, t, dt_ref, prior_logmean, prior_logsd, x0=None):
        st = self.init(prior_logmean, prior_logsd, x0 if x0 is not None else y[0])
        states = []
        for k in range(1, len(y)):
            st = self.step(st, y[k], t[k] - t[k - 1], dt_ref)
            states.append(st)
        return states


# ══════════════════════════════════════════════════════════════════════════
# 4.  RUL prediction
# ══════════════════════════════════════════════════════════════════════════
def lamperti(x, gamma, x_ref):
    """psi(x) = int_0^x q(u)^{-gamma} du  with  q(u) = 1 + u/x_ref.

    Under the Paris-coupled process dX_k ~ BS(alpha, beta*(dt/dt_ref)*q(X)^gamma)
    the transformed increment satisfies  dZ_k = psi(X_k) - psi(X_{k-1})
                                              = BS(alpha, beta*dt/dt_ref) + O(dX^2),
    so the homogeneous first-passage theory applies on the psi-scale with
    distance-to-failure  d_Z = psi(D) - psi(x_k).
    """
    g = np.asarray(gamma, dtype=float)
    q = 1.0 + np.asarray(x, dtype=float) / x_ref
    near1 = np.abs(g - 1.0) <= 1e-8
    out = np.where(near1,
                   x_ref * np.log(q),
                   x_ref * (np.power(q, 1.0 - np.where(near1, 0.0, g)) - 1.0)
                   / np.where(near1, 1.0, 1.0 - g))
    return out


def lamperti_var_distance(x, D, gamma, x_ref):
    """Effective distance governing the SPREAD of the first-passage time.

    psi maps the coupled process to one with constant drift, but not to one with
    constant step variance.  Under the grid-consistent coupling the increment at
    state x has Var[dX] = q(x)^gamma * var1, so on the Lamperti scale

        Var[dZ] = psi'(x)^2 Var[dX] = q^{-2 gamma} q^{gamma} var1
                = q(x)^{-gamma} var1 ,

    which still depends on x.  Accumulating by the renewal CLT along the path,
    Var[tau] = (var1/mu1^3) int_x^D q(u)^{-2 gamma} du, so the variance has its
    own effective distance -- the same Lamperti integral taken at 2*gamma:

        d_V = psi_{2 gamma}(D) - psi_{2 gamma}(x) .

    At gamma = 0 this equals d_Z and the homogeneous formulas are recovered
    exactly.  Using d_V = d_Z instead, as a first-order treatment does,
    over-predicts the RUL standard deviation by up to 40% at gamma = 1.5.
    """
    out = lamperti(D, 2.0 * gamma, x_ref) - lamperti(x, 2.0 * gamma, x_ref)
    fallback = np.maximum(D - np.asarray(x, dtype=float), 0.0)
    return np.where(np.isfinite(out), out, fallback)


def lamperti_distance(x, D, mu1, var1, gamma, x_ref, second_order=True):
    """Effective distance-to-failure on the Lamperti scale.

    `mu1`, `var1` are the mean and variance of the increment at unit scale, so
    the construction applies to ANY increment law, not only Birnbaum-Saunders.

    First order:   d_Z = psi(D) - psi(x),  psi(x) = int_0^x q(u)^{-gamma} du.

    Second order:  psi is curved, so E[dZ] = psi'(x) E[dX] + (1/2) psi''(x) E[dX^2]
    with psi'(x) = q^{-gamma} and psi''(x) = -gamma q^{-gamma-1}/x_ref.  At state
    x the increment has scale s = q(x)^gamma, and grid consistency makes the mean
    proportional to s but the variance ALSO proportional to s -- not to s^2.  So

        E[dX]   = mu1 q^gamma
        E[dX^2] = Var + mean^2 = var1 q^gamma + mu1^2 q^{2 gamma}

    and the two terms carry different powers of q:

        E[dZ] = mu1 - (gamma/(2 x_ref)) [ var1 q^{-1} + mu1^2 q^{gamma-1} ].

    Giving both terms q^{2 gamma} -- i.e. letting the variance inherit the SQUARE
    of the scale -- is the same mistake the grid-consistency theorem exists to
    rule out.  It agrees at x = 0 and drifts as q(x) grows.

    The transformed drift therefore decays as damage accumulates.  Integrating
    dz / E[dZ] along the path, with int q^{-1} du = x_ref ln q and
    int q^{-gamma-1} du = (x_ref/gamma)[q(x)^-gamma - q(D)^-gamma],

        E[tau] = (1/mu1) [ d_Z + kappa ],
        kappa  = var1/(2 mu1) [ q(x)^-gamma - q(D)^-gamma ]
                 + (gamma mu1/2) ln( q(D)/q(x) ),

    the factor x_ref cancelling from both integrals.  kappa vanishes at
    gamma = 0, recovering the homogeneous process exactly.
    """
    dz = lamperti(D, gamma, x_ref) - lamperti(x, gamma, x_ref)
    fallback = np.maximum(D - np.asarray(x, dtype=float), 0.0)
    dz = np.where(np.isfinite(dz), dz, fallback)
    if not second_order:
        return dz
    qD = 1.0 + D / x_ref
    qx = 1.0 + np.maximum(x, 0.0) / x_ref
    safe_mu = np.maximum(mu1, 1e-300)
    kappa = (0.5 * var1 / safe_mu * (np.power(qx, -gamma) - np.power(qD, -gamma))
             + 0.5 * gamma * mu1 * np.log(qD / qx))
    out = dz + np.where(np.isfinite(kappa), kappa, 0.0)
    return np.where(np.isfinite(out), out, fallback)


def rul_ig_params(d, th, law: IncrementLaw, dt_ref, scale=1.0, d_var=None):
    """IG approximation parameters for the first-passage time to distance d.

    With per-step (length dt_ref) increment mean mu_D and variance sig2_D, the
    renewal CLT gives  E[RUL] = d*dt_ref/mu_D  and  Var[RUL] = d*dt_ref^2*sig2_D/mu_D^3.
    Matching to IG(mu, lam), whose variance is mu^3/lam, yields

        mu_RUL  = d * dt_ref / mu_D ,        lam_RUL = d^2 * dt_ref / sig2_D .

    Both carry units of time (lam_RUL is LINEAR, not quadratic, in dt_ref).

    `d_var` is the distance governing the VARIANCE, which differs from `d` once
    the process is state-coupled: the Lamperti map removes the drift's state
    dependence but not the step variance's, so mean and spread accumulate over
    different effective distances (see lamperti_var_distance).  Then

        lam_RUL = d^3 * dt_ref / (sig2_D * d_var) ,

    which reduces to the formula above when d_var = d -- the homogeneous case,
    and the default.
    """
    mu_d = np.maximum(law.mean(th, scale), 1e-300)
    var_d = np.maximum(law.var(th, scale), 1e-300)
    mu_rul = d * dt_ref / mu_d
    if d_var is None:
        lam_rul = d ** 2 * dt_ref / var_d
    else:
        lam_rul = d ** 3 * dt_ref / (var_d * np.maximum(d_var, 1e-300))
    return mu_rul, lam_rul


def mc_first_passage(st: PFState, pf: ParticleFilter, idx, D, dt_ref,
                     n_mc=64, n_max=4000):
    """Continuous-time first passage by simulation, vectorised over particles.

    Paths are advanced in steps of dt_ref; the crossing instant inside the
    terminal step is recovered by linear interpolation, so tau is continuous
    rather than quantised to multiples of dt_ref.
    Returns an (len(idx), n_mc) array; paths that do not cross within n_max
    steps are right-censored at n_max*dt_ref.
    """
    rng = pf.rng
    th_all = pf._theta(st.logth)
    m = len(idx)
    if m * n_mc > 8_000:             # keep the terminal-phase branch affordable
        n_mc = max(8, 8_000 // m)
    th = {k: np.repeat(v[idx], n_mc).reshape(m, n_mc) for k, v in th_all.items()}
    x = np.repeat(st.x[idx], n_mc).reshape(m, n_mc)
    tau = np.zeros((m, n_mc))
    alive = np.ones((m, n_mc), dtype=bool)

    for _ in range(n_max):
        if not alive.any():
            break
        s = pf._scale(x, dt_ref, dt_ref, th)
        dx = pf.law.sample(th, s, rng)
        dx = np.where(np.isfinite(dx), dx, 0.0)
        x_new = x + dx
        crossed = alive & (x_new >= D)
        # linear interpolation of the crossing instant within the step
        frac = np.where(crossed, (D - x) / np.maximum(dx, 1e-300), 1.0)
        tau = np.where(alive, tau + dt_ref * np.clip(frac, 0.0, 1.0), tau)
        x = np.where(alive, x_new, x)
        alive = alive & ~crossed
    return tau, alive          # `alive` flags right-censored paths


def _rul_ig(st: PFState, pf: ParticleFilter, D, dt_ref):
    """Per-particle IG parameters of the RUL, on the Lamperti scale when the
    Paris coupling is active (there the transformed process is homogeneous)."""
    th = pf._theta(st.logth)
    if pf.cfg.gamma_coupling:
        one = np.ones_like(st.x)
        mu1 = pf.law.mean(th, one)
        var1 = pf.law.var(th, one)
        d = np.maximum(lamperti_distance(st.x, D, mu1, var1, th["gamma"],
                                         pf.cfg.x_ref), 1e-12)
        d_var = np.maximum(lamperti_var_distance(st.x, D, th["gamma"],
                                                 pf.cfg.x_ref), 1e-12)
        s0 = one
    else:
        d = np.maximum(D - st.x, 1e-12)
        d_var = None                      # homogeneous: the two coincide
        s0 = pf._scale(st.x, dt_ref, dt_ref, th)
    mu_rul, lam_rul = rul_ig_params(d, th, pf.law, dt_ref, scale=s0, d_var=d_var)
    n_d = d / np.maximum(pf.law.mean(th, s0), 1e-300)
    return th, mu_rul, lam_rul, n_d


def predict_rul(st: PFState, pf: ParticleFilter, D, dt_ref, nd_min=10.0,
                n_mc=64, n_draw=8, n_max=4000):
    """Posterior predictive RUL from the particle cloud.

    Per particle the conditional RUL law is taken from the IG approximation when
    the expected number of remaining increments n_d >= nd_min, and from direct
    Monte-Carlo path simulation otherwise (accurate near end of life).

    Returns dict with the posterior mean, weighted predictive samples, the
    epistemic/aleatory split and diagnostics.
    """
    th, mu_rul, lam_rul, n_d = _rul_ig(st, pf, D, dt_ref)

    w = np.exp(st.logw - st.logw.max())
    w /= w.sum()

    rul_mean_i = mu_rul.copy()                      # E[RUL | particle i]
    draws = np.empty((len(w), n_draw))

    ig_idx = np.flatnonzero((n_d >= nd_min) & np.isfinite(mu_rul))
    mc_idx = np.flatnonzero(~((n_d >= nd_min) & np.isfinite(mu_rul)))
    if ig_idx.size:
        draws[ig_idx] = ig_rvs(np.repeat(mu_rul[ig_idx][:, None], n_draw, 1),
                               np.repeat(lam_rul[ig_idx][:, None], n_draw, 1),
                               rng=pf.rng)
    censored = 0.0
    if mc_idx.size:
        sims, alive = mc_first_passage(st, pf, mc_idx, D, dt_ref,
                                       n_mc=n_mc, n_max=n_max)
        censored = float(alive.mean())
        rul_mean_i[mc_idx] = sims.mean(axis=1)
        take = pf.rng.integers(0, sims.shape[1], size=(len(mc_idx), n_draw))
        draws[mc_idx] = np.take_along_axis(sims, take, axis=1)
    rul_mean_i = np.where(np.isfinite(rul_mean_i), rul_mean_i, 0.0)
    draws = np.where(np.isfinite(draws), draws, 0.0)

    # predictive sample: resample particles by weight, then take their draws
    sel = systematic_resample(w, pf.rng)
    samples = draws[sel].ravel()
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        samples = np.array([float(np.sum(w * rul_mean_i))])

    v_al = float(np.sum(w * mu_rul ** 3 / np.maximum(lam_rul, 1e-300)))
    m1 = float(np.sum(w * mu_rul))
    v_ep = max(float(np.sum(w * mu_rul ** 2)) - m1 ** 2, 0.0)
    rho = v_ep / max(v_ep + v_al, 1e-300)

    return {
        "mean": float(np.sum(w * rul_mean_i)),
        "samples": samples,
        "v_aleatory": v_al,
        "v_epistemic": v_ep,
        "rho": rho,
        "n_d_mean": float(np.sum(w * n_d)),
        "frac_mc": float(len(mc_idx) / len(w)),
        "frac_censored": censored,
        "mu_rul": mu_rul,
        "lam_rul": lam_rul,
        "w": w,
    }


def uncertainty_decomposition(st: PFState, pf: ParticleFilter, D, dt_ref):
    """Law-of-total-variance split of Var(RUL | Y_{1:k}) under the IG approximation."""
    _, mu, lam, _ = _rul_ig(st, pf, D, dt_ref)

    w = np.exp(st.logw - st.logw.max())
    w /= w.sum()
    v_al = float(np.sum(w * mu ** 3 / np.maximum(lam, 1e-300)))       # E[Var(RUL|th)]
    m1 = float(np.sum(w * mu))
    m2 = float(np.sum(w * mu ** 2))
    v_ep = max(m2 - m1 ** 2, 0.0)                                     # Var[E(RUL|th)]
    rho = v_ep / max(v_ep + v_al, 1e-300)
    return v_al, v_ep, rho
