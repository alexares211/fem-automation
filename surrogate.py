"""Data-driven / surrogate-model ply orientation optimisation.

This is the project's optimisation strategy: the part's geometry, supports
(Lager) and loads are arbitrary and change from part to part, so there is no
closed-form stiffness expression to search -- every evaluation is one real
CalculiX solve of the actual model. This module accepts that cost and works
with it directly instead of approximating the physics away:

    1. SAMPLE    Design of Experiments over the free ply angles (Latin
                 Hypercube, not a full Cartesian sweep)
    2. EVALUATE  the real (expensive) objective at each sampled point --
                 one full ccx solve, every time
    3. FIT       a Gaussian Process surrogate f_hat(theta) with predictive
                 uncertainty (not just a point estimate)
    4. OPTIMISE  search the cheap surrogate, not the real evaluator, via
                 either sequential Bayesian Optimization (Expected
                 Improvement) or a genetic-algorithm-search-then-verify
                 pattern

The evaluator is a plain callable angles -> float (maximised). The only one
this module ships, make_ccx_evaluator, reuses the exact full_inp ->
run_ccx_blocking -> dat_parser.read_reaction_force chain already used
elsewhere in this repo, reading the .dat History Output -- never .frd,
which CalculiX 2.22 zeroes for multi-ply composite shells -- so it stays
correct on whatever geometry, supports and loads the current part has, not
just one example part.

No scipy / scikit-learn dependency: the GP below is a minimal from-scratch
implementation (RBF kernel, Cholesky solve) since this project's
environment only has numpy.
"""
import math
import os

import numpy as np

_ANGLE_PERIOD = 180.0


class SurrogateError(Exception):
    pass


class SurrogateCancelled(SurrogateError):
    """Raised by an evaluator to signal that the search should stop at the
    next safe point (a user-requested cancellation) rather than be treated
    as a failed evaluation."""
    pass


# ---------------------------------------------------------------------------
# free/full angle expansion, respecting the same symmetry convention as
# clt.py: for a symmetric laminate only ceil(N/2) plies are free, the rest
# are mirror reflections.
# ---------------------------------------------------------------------------

def _half(n_plies, symmetric):
    return (n_plies + 1) // 2 if symmetric else n_plies


def _free_index(k, n_plies, half, symmetric):
    return k if (not symmetric or k < half) else (n_plies - 1 - k)


def snap_to_grid(angles, step):
    """Round `angles` to the nearest multiple of `step`, wrapped into the
    [0, 180) period (a ply at 180 deg is physically the same orientation as
    0 deg -- see expand_free/_ANGLE_PERIOD). `step=None` or `0` disables
    snapping (the default -- continuous search, unchanged behaviour).

    Applied everywhere a candidate angle is *generated* (DoE, GA population
    init, mutation, the Bayesian candidate pool) so every angle that ever
    reaches a real CalculiX solve sits on the user's requested grid (e.g.
    0/15/30/.../165 for a 15 deg step) instead of the raw continuous values
    LHS/GA/EI would otherwise produce.
    """
    if not step:
        return angles
    arr = np.asarray(angles, dtype=float)
    return np.round(arr / float(step)) * float(step) % _ANGLE_PERIOD


def expand_free(free_angles, n_plies, symmetric):
    """A length-`half` vector of free-ply angles -> the full length-N
    stacking sequence (mirrored if symmetric)."""
    half = _half(n_plies, symmetric)
    free_angles = list(free_angles)
    if len(free_angles) != half:
        raise SurrogateError(f"Expected {half} free angles, got {len(free_angles)}.")
    return [float(free_angles[_free_index(k, n_plies, half, symmetric)]) for k in range(n_plies)]


# ---------------------------------------------------------------------------
# 1. the evaluator -- angles (length n_plies) -> scalar S, maximised
# ---------------------------------------------------------------------------

def make_ccx_evaluator(model, ccx_exe, output_folder, reduce="sum", cancel_event=None):
    """The evaluator: writes one .inp per candidate sequence, solves it with
    CalculiX, and reads reaction force back from the .dat History Output
    (never .frd -- confirmed CalculiX 2.22 bug: RF reads as 0 there for
    multi-ply composite shells). Built to drop straight into a Flask
    background job (see app.py's run_bo / /bo_start).

    model: same dict shape app.py's /bo_start endpoint builds -- nodes,
    elements, element_type, material_name, constants, thickness,
    symmetric, lagers, loads. Nothing here assumes any particular geometry,
    support layout or load case -- whatever the current part's model dict
    contains is what gets solved.
    reduce: "sum" (default) or "max" of the per-lager reaction-force norm.
    cancel_event: an optional threading.Event; checked before every solve
        and passed through to run_ccx_blocking so a Stop button can end a
        long search between -- or during -- individual solves. A cancelled
        call raises SurrogateCancelled, which bayesian_optimise and
        genetic_search_then_verify treat as "stop and return what's been
        found so far", not as a failed evaluation.

    Every call costs one real ccx solve (several seconds or more) -- this
    is the expensive f(theta) the rest of this module is built to search
    sample-efficiently instead of brute-forcing.
    """
    from ccx_runner import run_ccx_blocking
    from dat_parser import read_reaction_force, DatParseError
    from inp_writer import full_inp, InpWriteError, LAGER_ALL

    if reduce not in ("sum", "max"):
        raise SurrogateError("reduce must be 'sum' or 'max'.")
    os.makedirs(output_folder, exist_ok=True)
    call_count = [0]

    def f(angles):
        if cancel_event is not None and cancel_event.is_set():
            raise SurrogateCancelled("Stopped by user.")
        call_count[0] += 1
        suffix = "_".join(
            str(round(float(a), 1)).replace("-", "m").replace(".", "p") for a in angles
        )
        inp_path = os.path.join(output_folder, f"bo_{call_count[0]:04d}_{suffix}deg.inp")
        try:
            text = full_inp(
                model["nodes"], model["elements"], model["element_type"],
                model["material_name"], model["constants"],
                list(angles), model["thickness"], model["lagers"],
                model["symmetric"], "auto", model["loads"],
            )
        except InpWriteError as e:
            raise SurrogateError(f"Could not write .inp for angles={angles}: {e}")
        with open(inp_path, "w") as fh:
            fh.write(text)

        run_result = run_ccx_blocking(inp_path, ccx_exe, cancel_event=cancel_event)
        if run_result["stopped"]:
            raise SurrogateCancelled("Stopped by user.")
        if run_result["has_error"] or not run_result["converged"]:
            raise SurrogateError(f"CalculiX did not converge for angles={angles}.")

        dat_path = os.path.splitext(inp_path)[0] + ".dat"
        try:
            rf = read_reaction_force(dat_path, LAGER_ALL)
        except DatParseError as e:
            raise SurrogateError(f"Could not read reaction force for angles={angles}: {e}")
        norms = [math.sqrt(fx ** 2 + fy ** 2 + fz ** 2) for fx, fy, fz in rf["per_node"].values()]
        if not norms:
            raise SurrogateError(f"No reaction-force rows read back for angles={angles}.")
        return float(sum(norms)) if reduce == "sum" else float(max(norms))

    return f


# ---------------------------------------------------------------------------
# 2. Design of Experiments -- Latin Hypercube sampling
# ---------------------------------------------------------------------------

def latin_hypercube(n_samples, n_dims, low=0.0, high=_ANGLE_PERIOD, rng=None):
    """n_samples points spread one-per-stratum in each of n_dims dimensions
    (Latin Hypercube), each dimension independently permuted -- a small,
    well-spread sample instead of a full Cartesian grid."""
    if n_samples <= 0 or n_dims <= 0:
        return np.empty((0, max(n_dims, 0)))
    rng = rng or np.random.default_rng()
    out = np.empty((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        jitter = rng.uniform(size=n_samples)
        out[:, d] = (perm + jitter) / n_samples
    return low + out * (high - low)


# ---------------------------------------------------------------------------
# 3. surrogate -- minimal from-scratch Gaussian Process (RBF kernel)
# ---------------------------------------------------------------------------

class GaussianProcess:
    """GP regression with an RBF kernel, fit by Cholesky solve. Reports
    predictive mean *and* standard deviation, which is the whole reason
    this module uses a GP rather than a plain regression: the report is
    "trust me" plus a number for how much.
    """

    def __init__(self, length_scale=30.0, signal_var=1.0, noise_var=1e-6):
        self.length_scale = float(length_scale)
        self.signal_var = float(signal_var)
        self.noise_var = float(noise_var)
        self.X = self.y = self._L = self._alpha = None
        self._y_mean, self._y_std = 0.0, 1.0

    def _kernel(self, X1, X2):
        d2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
        return self.signal_var * np.exp(-0.5 * d2 / self.length_scale ** 2)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        yn = (y - self._y_mean) / self._y_std
        K = self._kernel(X, X) + self.noise_var * np.eye(len(X))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, yn))
        self.X, self.y = X, y
        return self

    def predict(self, Xs):
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = self._kernel(Xs, self.X)
        mean = (Ks @ self._alpha) * self._y_std + self._y_mean
        v = np.linalg.solve(self._L, Ks.T)
        var = np.diag(self._kernel(Xs, Xs)) - np.sum(v ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 1e-12)) * self._y_std
        return mean, std

    def log_marginal_likelihood(self):
        yn = (self.y - self._y_mean) / self._y_std
        n = len(self.y)
        return float(-0.5 * yn @ self._alpha - np.sum(np.log(np.diag(self._L)))
                     - 0.5 * n * math.log(2 * math.pi))


def fit_gp(X, y, length_scale_candidates=None, noise_var=1e-6):
    """Fit a GP, choosing length_scale by a coarse grid search over log
    marginal likelihood. There is no scipy optimizer available in this
    project's environment, so this is deliberately a grid search, not a
    full continuous hyperparameter optimisation -- good enough for the
    handful of dimensions (free ply angles) this module deals with.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if length_scale_candidates is None:
        spread = np.ptp(X, axis=0) if X.shape[0] > 1 else np.array([_ANGLE_PERIOD])
        base = float(np.mean(spread)) or _ANGLE_PERIOD
        length_scale_candidates = [base * f for f in (0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5)]
    best = None
    for ls in length_scale_candidates:
        gp = GaussianProcess(length_scale=max(ls, 1e-3), noise_var=noise_var).fit(X, y)
        lml = gp.log_marginal_likelihood()
        if best is None or lml > best[0]:
            best = (lml, gp)
    return best[1]


# ---------------------------------------------------------------------------
# 4a. optimise the surrogate -- sequential Bayesian Optimization (EI)
# ---------------------------------------------------------------------------

def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def expected_improvement(mean, std, f_best, xi=0.01):
    """EI for maximisation: how much a candidate is expected to beat the
    current best by, balancing high mean (exploit) against high
    uncertainty (explore)."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    imp = mean - f_best - xi
    safe_std = np.where(std > 1e-12, std, 1.0)
    z = imp / safe_std
    cdf = np.array([_norm_cdf(zz) for zz in z.ravel()]).reshape(z.shape)
    pdf = np.array([_norm_pdf(zz) for zz in z.ravel()]).reshape(z.shape)
    ei = imp * cdf + std * pdf
    return np.where(std > 1e-12, ei, 0.0)


def bayesian_optimise(evaluator, n_plies, symmetric=False, n_init=8, n_iter=15,
                       xi=0.01, pool_size=2000, seed_points=None, rng=None,
                       on_eval=None, angle_step=None):
    """Sequential Bayesian Optimization: sample a DoE, evaluate the real
    (expensive) `evaluator` on each, fit a GP, then repeatedly pick the
    single most-informative next angle combination (max Expected
    Improvement) and evaluate it for real.

    evaluator: callable(full_angle_sequence) -> float, MAXIMISED. May raise
        SurrogateCancelled to stop the search early -- whatever has been
        evaluated so far is still returned, with "stopped": True.
    seed_points: optional full-length angle sequences (e.g. a known-good
        layup) evaluated before the random DoE -- an informed rather than a
        cold start. Not snapped to angle_step -- the caller supplied these
        exactly, so they are trusted as-is.
    angle_step: optional grid spacing in degrees (e.g. 15) -- every
        generated candidate (DoE and the per-round EI pool) is rounded to
        the nearest multiple of this before it's ever handed to the real
        evaluator, so results only ever land on angles the user actually
        asked for. None/0 (default) searches continuously, unchanged.
    on_eval: optional callable(dict) invoked after every successful real
        evaluation with {"angles", "S", "source", "n_evaluations", ...} --
        the hook a caller uses to stream live progress (e.g. into a
        background job's status dict) while the search is still running.
    """
    n_plies = int(n_plies)
    half = _half(n_plies, symmetric)
    rng = rng or np.random.default_rng()

    X, y, history = [], [], []
    stopped = False

    def try_eval(free_vec, full, source, iteration=None):
        nonlocal stopped
        try:
            val = float(evaluator(full))
        except SurrogateCancelled:
            stopped = True
            return False
        X.append(list(free_vec))
        y.append(val)
        entry = {"angles": full, "S": val, "source": source}
        if iteration is not None:
            entry["iteration"] = iteration
        history.append(entry)
        if on_eval is not None:
            on_eval(dict(entry, n_evaluations=len(y)))
        return True

    if seed_points:
        for pt in seed_points:
            pt = list(pt)
            if len(pt) != n_plies:
                raise SurrogateError(f"seed point must have length {n_plies}.")
            if not try_eval([pt[k] for k in range(half)], pt, "seed"):
                break

    if not stopped:
        doe = snap_to_grid(latin_hypercube(max(0, int(n_init)), half, rng=rng), angle_step)
        for row in doe:
            if not try_eval(row, expand_free(row, n_plies, symmetric), "init"):
                break

    if not X:
        if stopped:
            return {"history": [], "best_angles": None, "best_S": None,
                    "n_evaluations": 0, "stopped": True}
        raise SurrogateError("Need at least one initial sample: set n_init > 0 or pass seed_points.")

    if not stopped:
        for it in range(int(n_iter)):
            gp = fit_gp(np.array(X), np.array(y))
            pool = rng.uniform(0.0, _ANGLE_PERIOD, size=(pool_size, half))
            best_idx = int(np.argmax(y))
            n_jitter = max(1, pool_size // 10)
            jitter = (np.array(X[best_idx]) + rng.normal(0, 10.0, size=(n_jitter, half))) % _ANGLE_PERIOD
            pool = np.vstack([pool, jitter])
            pool = snap_to_grid(pool, angle_step)
            mean, std = gp.predict(pool)
            ei = expected_improvement(mean, std, f_best=max(y), xi=xi)
            next_x = pool[int(np.argmax(ei))]
            full = expand_free(next_x, n_plies, symmetric)
            if not try_eval(next_x, full, "EI", iteration=it + 1):
                break

    best_idx = int(np.argmax(y))
    return {
        "history": history,
        "best_angles": expand_free(X[best_idx], n_plies, symmetric),
        "best_S": float(y[best_idx]),
        "n_evaluations": len(y),
        "stopped": stopped,
    }


# ---------------------------------------------------------------------------
# 4b. optimise the surrogate -- search-then-verify (GA against the GP)
# ---------------------------------------------------------------------------

def genetic_search_then_verify(evaluator, n_plies, symmetric=False, n_init=20,
                                pop_size=200, n_generations=60, elite_frac=0.1,
                                mutation_std=15.0, top_k=3, rng=None,
                                on_eval=None, angle_step=None):
    """Fit a GP on an initial DoE sample, then run a genetic algorithm
    directly against the (now cheap) surrogate -- millions of surrogate
    evaluations cost nothing -- and confirm only the top_k winners with the
    real, expensive evaluator.

    evaluator / on_eval / angle_step: see bayesian_optimise -- same contract,
        including SurrogateCancelled support (checked around both the
        initial DoE and the final top_k verification, the only two points
        real solves run). angle_step is applied to the DoE, the initial
        population and every mutated child, so the population never drifts
        off the requested grid across generations.
    """
    n_plies = int(n_plies)
    half = _half(n_plies, symmetric)
    rng = rng or np.random.default_rng()

    doe = snap_to_grid(latin_hypercube(int(n_init), half, rng=rng), angle_step)
    if len(doe) == 0:
        raise SurrogateError("n_init must be > 0.")

    X, y = [], []
    stopped = False
    for row in doe:
        full = expand_free(row, n_plies, symmetric)
        try:
            val = float(evaluator(full))
        except SurrogateCancelled:
            stopped = True
            break
        X.append(list(row))
        y.append(val)
        if on_eval is not None:
            on_eval({"angles": full, "S": val, "source": "init", "n_evaluations": len(y)})

    if not X:
        if stopped:
            return {"surrogate_training_points": 0, "ga_population": pop_size,
                    "ga_generations": n_generations, "verified": [], "best": None,
                    "stopped": True}
        raise SurrogateError("All initial evaluations failed before any completed.")

    gp = fit_gp(np.array(X), np.array(y))
    verified = []

    if not stopped:
        pop = snap_to_grid(rng.uniform(0.0, _ANGLE_PERIOD, size=(pop_size, half)), angle_step)
        n_elite = max(1, int(pop_size * elite_frac))
        for _ in range(int(n_generations)):
            mean, _std = gp.predict(pop)
            order = np.argsort(mean)[::-1]
            elite = pop[order[:n_elite]]
            children = []
            while len(children) < pop_size - n_elite:
                p1 = elite[rng.integers(n_elite)]
                p2 = elite[rng.integers(n_elite)]
                mask = rng.random(half) < 0.5
                child = np.where(mask, p1, p2)
                child = (child + rng.normal(0, mutation_std, size=half)) % _ANGLE_PERIOD
                child = snap_to_grid(child, angle_step)
                children.append(child)
            pop = np.vstack([elite, np.array(children)]) if children else elite

        mean, std = gp.predict(pop)
        order = np.argsort(mean)[::-1][:max(1, int(top_k))]
        for idx in order:
            full = expand_free(pop[idx], n_plies, symmetric)
            try:
                S = float(evaluator(full))
            except SurrogateCancelled:
                stopped = True
                break
            entry = {"angles": full, "S": S,
                     "surrogate_mean": float(mean[idx]), "surrogate_std": float(std[idx])}
            verified.append(entry)
            if on_eval is not None:
                on_eval(dict(entry, source="verify", n_evaluations=len(X) + len(verified)))
        verified.sort(key=lambda e: e["S"], reverse=True)

    return {
        "surrogate_training_points": len(X),
        "ga_population": pop_size,
        "ga_generations": n_generations,
        "verified": verified,
        "best": verified[0] if verified else None,
        "stopped": stopped,
    }
