"""Classical Laminate Theory (CLT) -- analytic Stage-1 stiffness optimisation.

Stage 1 of the Loesungsprinzip: for a FIXED number of plies N, find the
stacking sequence  [theta_1 .. theta_N]  (each theta drawn from a discrete
candidate set) that maximises a scalar stiffness measure S read off the
laminate's ABD matrix. Everything here is closed form -- no CalculiX, no
FE mesh, no iteration.

Pipeline (Loesungsprinzip sections 02 and 04):

  1. Q            per-ply reduced stiffness in the fibre / matrix axes,
                  from E1, E2, G12, nu12. One matrix, computed once.
  2. Qbar(theta)  Q rotated into the laminate x-y axes. Only as many
                  distinct Qbar matrices exist as there are candidate
                  angles -- all precomputed once.
  3. A, B, D      assembled by summing each ply's Qbar, weighted by its
                  through-thickness position z, over the stack:
                      A_ij = sum_k  Qbar_ij(k) * (z_k - z_{k-1})
                      B_ij = 1/2 sum_k Qbar_ij(k) * (z_k^2 - z_{k-1}^2)
                      D_ij = 1/3 sum_k Qbar_ij(k) * (z_k^3 - z_{k-1}^3)
                  with z measured from the mid-plane.
  4. S            one number pulled from A / B / D -- an entry (A11), or a
                  combination such as the effective modulus Ex. This is
                  the objective being maximised.

Angle domain: a ply's stiffness is periodic in 180 degrees (a fibre is a
line, not a vector -- Qbar(theta) == Qbar(theta + 180)). So the candidate
set spans 0..180 exclusive; theta and 180-theta are genuinely different
plies (they differ in the sign of the shear-coupling terms Qbar16/Qbar26).
Angles are folded into [0, 180) and de-duplicated on input, so negative
angles may be entered directly ( -45 -> 135 ).

Solver strategy (Loesungsprinzip section 4.3):

  * If the candidate count na**(free plies) fits under CLT_MAX_CANDIDATES,
    every sequence is enumerated and scored with numpy in one vectorised
    pass. Exact, and it also yields a ranked list.
  * If it does not fit AND the objective is a single ABD entry, the plies
    are independent (each term in the A/B/D sum depends only on that one
    ply's angle and position) so the optimum is chosen layer by layer with
    no search at all -- for any N.
  * If it does not fit and the objective couples several ABD entries,
    the caller is asked to enable symmetry, use coarser angle steps, cut
    the ply count, or pick a single-entry objective.

What "symmetric" buys: a laminate mirrored about its mid-plane has B == 0
exactly (no extension-bending coupling), and only ceil(N/2) plies are free
-- the rest are reflections -- so the search space shrinks from na**N to
na**ceil(N/2).
"""
import numpy as np

# Default candidate angles: 0..180 in 15 degree steps, 180 omitted (== 0).
DEFAULT_ANGLES = tuple(float(a) for a in range(0, 180, 15))

# Above this a full enumeration is refused (memory + time). The headline
# case -- a 12-ply symmetric laminate with the 12-angle default set --
# is 12**6 = 2 985 984 candidates and fits under this ceiling.
CLT_MAX_CANDIDATES = 3_500_000

_ANGLE_PERIOD = 180.0


class CLTError(Exception):
    pass


def normalise_angles(angles):
    """Fold every angle into [0, 180) and drop duplicates; returns a sorted list."""
    out = set()
    for a in angles:
        try:
            a = float(a)
        except (TypeError, ValueError):
            raise CLTError(f"Candidate angle {a!r} is not a number.")
        out.add(round(a % _ANGLE_PERIOD, 6))
    if not out:
        raise CLTError("Provide at least one candidate angle.")
    return sorted(out)


# ---------------------------------------------------------------------------
# 1. per-ply stiffness
# ---------------------------------------------------------------------------

def reduced_stiffness(E1, E2, G12, nu12):
    """Q, the 3x3 plane-stress reduced stiffness in the ply's own
    fibre (1) / transverse (2) axes. Loesungsprinzip eq 2.1.
    """
    try:
        E1, E2, G12, nu12 = float(E1), float(E2), float(G12), float(nu12)
    except (TypeError, ValueError):
        raise CLTError("E1, E2, G12 and nu12 must all be numbers.")
    if min(E1, E2, G12) <= 0:
        raise CLTError("E1, E2 and G12 must be positive.")

    nu21 = nu12 * E2 / E1
    denom = 1.0 - nu12 * nu21
    if denom <= 0:
        raise CLTError(
            "1 - nu12*nu21 <= 0 -- nu12 is too large for this E1/E2 "
            "(needs nu12 < sqrt(E1/E2))."
        )

    Q = np.zeros((3, 3))
    Q[0, 0] = E1 / denom
    Q[1, 1] = E2 / denom
    Q[0, 1] = Q[1, 0] = nu12 * E2 / denom
    Q[2, 2] = G12
    return Q


def transformed_stiffness(Q, theta_deg):
    """Qbar(theta): Q rotated from the ply axes into the laminate x-y axes.

    Uses the standard  Qbar = T^-1 . Q . R . T . R^-1  form (Jones,
    "Mechanics of Composite Materials"), which is less error-prone to code
    than the cos/sin power expansion and gives the full 3x3 including the
    Qbar16 / Qbar26 shear-coupling terms.
    """
    t = np.radians(float(theta_deg))
    c, s = np.cos(t), np.sin(t)
    T = np.array([
        [c * c,      s * s,       2 * c * s],
        [s * s,      c * c,      -2 * c * s],
        [-c * s,     c * s,       c * c - s * s],
    ])
    R = np.diag([1.0, 1.0, 2.0])          # Reuter matrix (engineering shear strain)
    Tinv = np.linalg.inv(T)
    Rinv = np.diag([1.0, 1.0, 0.5])
    return Tinv @ Q @ R @ T @ Rinv


def _interface_z(n_plies, ply_t):
    """z_0 .. z_N, the through-thickness interface coordinates, mid-plane at 0."""
    h = n_plies * ply_t
    return -0.5 * h + ply_t * np.arange(n_plies + 1), h


def abd_matrices(Q, angles_deg, ply_t):
    """Assemble (A, B, D, h) for one explicit stacking sequence.
    Loesungsprinzip eq 2.3.
    """
    ply_t = float(ply_t)
    if ply_t <= 0:
        raise CLTError("Ply thickness must be positive.")
    angles_deg = list(angles_deg)
    if not angles_deg:
        raise CLTError("Need at least one ply.")

    z, h = _interface_z(len(angles_deg), ply_t)
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for k, ang in enumerate(angles_deg):
        Qb = transformed_stiffness(Q, ang)
        A += Qb * (z[k + 1] - z[k])
        B += Qb * (z[k + 1] ** 2 - z[k] ** 2) / 2.0
        D += Qb * (z[k + 1] ** 3 - z[k] ** 3) / 3.0
    return A, B, D, h


# ---------------------------------------------------------------------------
# 2. objectives -- the scalar S pulled from A / B / D
# ---------------------------------------------------------------------------
# Each entry: key -> (human label, scalar function of (A, B, D, h)).
# The batch equivalent lives in _score_batch below; keep the two in step.

def _eff_in_plane(A, h):
    a11, a22, a12, a66 = A[0, 0], A[1, 1], A[0, 1], A[2, 2]
    det_in = a11 * a22 - a12 ** 2
    return {
        "Ex": det_in / (a22 * h),
        "Ey": det_in / (a11 * h),
        "Gxy": a66 / h,
        "nu_xy": a12 / a22,
    }


OBJECTIVES = {
    "Ex":      ("effective Young's modulus along x  [MPa]",   lambda A, B, D, h: _eff_in_plane(A, h)["Ex"]),
    "Ey":      ("effective Young's modulus along y  [MPa]",   lambda A, B, D, h: _eff_in_plane(A, h)["Ey"]),
    "Gxy":     ("effective in-plane shear modulus  [MPa]",    lambda A, B, D, h: _eff_in_plane(A, h)["Gxy"]),
    "A11":     ("in-plane stiffness along x  [N/mm]",         lambda A, B, D, h: A[0, 0]),
    "A22":     ("in-plane stiffness along y  [N/mm]",         lambda A, B, D, h: A[1, 1]),
    "A11+A22": ("biaxial in-plane stiffness  [N/mm]",         lambda A, B, D, h: A[0, 0] + A[1, 1]),
    "A66":     ("in-plane shear stiffness  [N/mm]",           lambda A, B, D, h: A[2, 2]),
    "D11":     ("bending stiffness about y  [N.mm]",          lambda A, B, D, h: D[0, 0]),
}

# Objectives that are a single ABD entry -> plies are independent, so the
# optimum can be built layer by layer with no enumeration.  value = (which
# matrix, row, col).
_SINGLE_ENTRY = {
    "A11": ("A", 0, 0),
    "A22": ("A", 1, 1),
    "A66": ("A", 2, 2),
    "D11": ("D", 0, 0),
}


def _score_batch(objective, a11, a22, a12, a66, d11, h):
    """Vectorised objective over (M,) ABD-entry arrays -> (M,)."""
    det_in = a11 * a22 - a12 ** 2
    return {
        "Ex": det_in / (a22 * h),
        "Ey": det_in / (a11 * h),
        "Gxy": a66 / h,
        "A11": a11,
        "A22": a22,
        "A11+A22": a11 + a22,
        "A66": a66,
        "D11": d11,
    }[objective]


# ---------------------------------------------------------------------------
# 3. the Stage-1 solver
# ---------------------------------------------------------------------------

def _per_position_contributions(Q, angles_deg, n_plies, ply_t):
    """cA[k, a], cD[k, a] = ply k's contribution to A resp. D if it takes
    angle index a.  Shapes (n_plies, na, 3, 3).  Built once (Loesungsprinzip
    section 4.1) then reused for every candidate sequence.
    """
    z, h = _interface_z(n_plies, ply_t)
    Qb = np.stack([transformed_stiffness(Q, a) for a in angles_deg])   # (na, 3, 3)
    cA = np.empty((n_plies, len(angles_deg), 3, 3))
    cD = np.empty_like(cA)
    for k in range(n_plies):
        cA[k] = Qb * (z[k + 1] - z[k])
        cD[k] = Qb * (z[k + 1] ** 3 - z[k] ** 3) / 3.0
    return cA, cD, h


def _free_index(k, n_plies, half, symmetric):
    """Which free (left-half) ply position controls full position k."""
    return k if (not symmetric or k < half) else (n_plies - 1 - k)


def _solve_enumerate(Q, angles, n_plies, ply_t, objective, symmetric, top):
    cA, cD, h = _per_position_contributions(Q, angles, n_plies, ply_t)
    na = len(angles)
    half = (n_plies + 1) // 2 if symmetric else n_plies
    M = na ** half
    rows = np.arange(M)

    def sel_at(k):
        """Angle index chosen at full ply position k, for all M candidates."""
        f = _free_index(k, n_plies, half, symmetric)
        return (rows // (na ** (half - 1 - f))) % na

    # Accumulate only the ABD entries the objectives actually use -- keeps
    # peak memory at ~10 x M x 8 bytes instead of materialising (M, 3, 3).
    a11 = np.zeros(M); a22 = np.zeros(M); a12 = np.zeros(M)
    a66 = np.zeros(M); d11 = np.zeros(M)
    for k in range(n_plies):
        s = sel_at(k)
        a11 += cA[k][s, 0, 0]
        a22 += cA[k][s, 1, 1]
        a12 += cA[k][s, 0, 1]
        a66 += cA[k][s, 2, 2]
        d11 += cD[k][s, 0, 0]

    scores = _score_batch(objective, a11, a22, a12, a66, d11, h)
    order = np.argsort(scores)[::-1]

    def decode(row):
        left = [int((row // (na ** (half - 1 - f))) % na) for f in range(half)]
        return [angles[left[_free_index(k, n_plies, half, symmetric)]]
                for k in range(n_plies)]

    ranked = [
        {"rank": r + 1, "angles": decode(int(row)), "S": float(scores[row])}
        for r, row in enumerate(int(x) for x in order[:top])
    ]
    return decode(int(order[0])), ranked, int(M), "full enumeration"


def _solve_layerwise(Q, angles, n_plies, ply_t, objective, symmetric):
    """Single-entry objective: pick each ply's angle independently."""
    which, i, j = _SINGLE_ENTRY[objective]
    cA, cD, _ = _per_position_contributions(Q, angles, n_plies, ply_t)
    contrib = (cA if which == "A" else cD)[:, :, i, j]   # (n_plies, na)

    half = (n_plies + 1) // 2 if symmetric else n_plies
    chosen = [0] * n_plies
    for k in range(half):
        if symmetric:
            partner = n_plies - 1 - k
            pair = contrib[k] + (contrib[partner] if partner != k else 0.0)
            a = int(np.argmax(pair))
            chosen[k] = chosen[partner] = a
        else:
            chosen[k] = int(np.argmax(contrib[k]))

    seq = [angles[a] for a in chosen]
    return seq, n_plies * len(angles), "layer-by-layer (analytic, no enumeration)"


def optimise(E1, E2, G12, nu12, n_plies, ply_t,
             objective="Ex", angles=DEFAULT_ANGLES, symmetric=False, top=10):
    """Stage 1: best orientation sequence for a fixed ply count N.

    Returns a JSON-friendly dict: the winning sequence, its ABD matrices and
    effective engineering constants, the objective value, and a ranked list
    of the top candidates (enumeration path only).
    """
    if objective not in OBJECTIVES:
        raise CLTError(f"Unknown objective {objective!r}. "
                       f"Choose one of: {', '.join(OBJECTIVES)}.")
    try:
        n_plies = int(n_plies)
    except (TypeError, ValueError):
        raise CLTError("Number of plies must be a whole number.")
    if n_plies < 1:
        raise CLTError("Number of plies must be at least 1.")
    try:
        ply_t = float(ply_t)
    except (TypeError, ValueError):
        raise CLTError("Ply thickness must be a number.")
    if ply_t <= 0:
        raise CLTError("Ply thickness must be positive.")

    angles = normalise_angles(angles)
    na = len(angles)
    top = max(1, int(top))

    Q = reduced_stiffness(E1, E2, G12, nu12)

    half = (n_plies + 1) // 2 if symmetric else n_plies
    count = na ** half

    ranked = None
    if count <= CLT_MAX_CANDIDATES:
        best_seq, ranked, n_eval, method = _solve_enumerate(
            Q, angles, n_plies, ply_t, objective, symmetric, top
        )
    elif objective in _SINGLE_ENTRY:
        best_seq, n_eval, method = _solve_layerwise(
            Q, angles, n_plies, ply_t, objective, symmetric
        )
    else:
        raise CLTError(
            f"{count:,} candidate sequences ({na} angles ^ {half} free plies) "
            f"exceeds the {CLT_MAX_CANDIDATES:,} cap for a coupled objective "
            f"({objective}). Enable symmetry, use coarser angle steps, reduce "
            f"the ply count, or pick a single-entry objective "
            f"(A11 / A22 / A66 / D11), which is solved layer by layer without "
            f"enumeration."
        )

    # canonical report values: recompute the winner exactly from its sequence
    A, B, D, h = abd_matrices(Q, best_seq, ply_t)
    best_S = float(OBJECTIVES[objective][1](A, B, D, h))
    eff = _eff_in_plane(A, h)

    return {
        "objective": objective,
        "objective_label": OBJECTIVES[objective][0],
        "method": method,
        "num_plies": n_plies,
        "num_free_plies": half,
        "num_candidates": int(n_eval),
        "angles_considered": angles,
        "symmetric": bool(symmetric),
        "ply_thickness": ply_t,
        "laminate_thickness": float(h),
        "best_sequence": [float(a) for a in best_seq],
        "best_S": best_S,
        "A": A.tolist(),
        "B": B.tolist(),
        "D": D.tolist(),
        "eff": {k: float(v) for k, v in eff.items()},
        "ranked": ranked or [{"rank": 1, "angles": [float(a) for a in best_seq], "S": best_S}],
    }
