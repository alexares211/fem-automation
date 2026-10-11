"""Puck inter-fiber-failure (IFF) + fiber-failure (FF) check for a single UD
ply under plane stress (sigma33 = sigma23 = sigma13 = 0), which is what a
CalculiX shell element gives per ply/section point.

Equations (1), (4), (5), (6), (8), (9) transcribed directly from:
  Findeisen, Schober, Hohe, Weiss -- "User's Manual of Puck Analysis Tool",
  Fraunhofer IWM Report V1195/2022 (European Space Agency, Project 425168),
  Chapter 2 ("Puck's FRP failure criterion").
No coefficient here is from memory -- every one was read off that report's
equations 1 and 4-9 at full zoom before being typed in, specifically because
getting a strength criterion subtly wrong is a bad way to find out.

Stress sign/axis convention (matches the report and this project's
*Orientation-per-ply setup): sigma11 = fiber direction, sigma22 = transverse
in-plane, sigma12 = in-plane shear -- all in the PLY's local material frame,
which is exactly what CalculiX reports once each ply has its own
*Orientation (see inp_writer.section_block).
"""
import math
from dataclasses import dataclass

from dat_parser import read_ply_stresses, DatParseError


class PuckInputError(Exception):
    pass


@dataclass(frozen=True)
class PuckStrength:
    """The independent material parameters of Table in Section 2.4 of the
    report. S11t/S11c/S22t/S22c/S21s are strengths (all positive by
    convention, including the compressive ones). p21_plus/p21_minus are the
    Mode A / Mode B slope parameters -- Puck's own guidance is that a smooth
    A-to-B transition is usually wanted, i.e. p21_plus == p21_minus, so that
    is the default if only one is given.
    """
    S11t: float
    S11c: float
    S22t: float
    S22c: float
    S21s: float
    p21_plus: float
    p21_minus: float = None

    def __post_init__(self):
        if self.p21_minus is None:
            object.__setattr__(self, "p21_minus", self.p21_plus)
        for name in ("S11t", "S11c", "S22t", "S22c", "S21s"):
            v = getattr(self, name)
            if v is None or v <= 0:
                raise PuckInputError(f"Puck strength {name!r} must be a positive number.")
        if self.p21_plus <= 0 or self.p21_minus <= 0:
            raise PuckInputError("Puck slope parameters p21_plus/p21_minus must be positive.")


# Typical CFRP default slope parameters (Puck & Schuermann's own commonly
# cited guidance values) -- a starting point only. Use PuckStrength's
# p21_plus/p21_minus directly once you have real values for your material;
# don't ship a real failure assessment on these defaults alone.
TYPICAL_CFRP_P21_PLUS = 0.30


def p22_minus(p21_minus, S22c, S21s):
    """Equation (6): the Mode B/C transition parameter, derived from
    p21_minus (not an independent material constant)."""
    return 0.5 * (math.sqrt(1 + 2 * p21_minus * S22c / S21s) - 1)


def fiber_failure(sigma11, strength: PuckStrength):
    """Equation (1): max-stress check along the fiber direction.
    Returns f_FF; f_FF >= 1 means the fiber has broken."""
    if sigma11 >= 0:
        return sigma11 / strength.S11t
    return abs(sigma11) / strength.S11c


def inter_fiber_failure(sigma22, sigma12, strength: PuckStrength):
    """Equation (4)+(5): plane-stress Puck IFF check (closed form -- no
    angle search needed, Puck already solved that analytically for the
    plane-stress case; see Section 2.2 of the report).

    Returns (f_IFF, mode) where mode is 'A', 'B', or 'C', and f_IFF >= 1
    means the matrix has cracked (fiber unaffected).
    """
    S22t, S22c, S21s = strength.S22t, strength.S22c, strength.S21s
    p21p, p21m = strength.p21_plus, strength.p21_minus

    if sigma22 > 0:
        # Mode A
        term = math.sqrt(
            (sigma12 / S21s) ** 2
            + (1 - p21p * S22t / S21s) ** 2 * (sigma22 / S22t) ** 2
        )
        f = term + p21p * (sigma22 / S21s)
        return f, "A"

    # sigma22 <= 0: Mode B vs Mode C is decided by Equation (5)'s stress
    # ratio. The report states it as "0 <= |sigma22/sigma21| <= R22A/|sigma21c|"
    # for Mode B and "0 <= sigma21/sigma22 <= |sigma21c|/R22A" for Mode C --
    # algebraically those are the same threshold on |sigma12|/|sigma22|,
    # just written as reciprocal ratios in each case. Using |sigma12|/|sigma22|
    # directly (shear-to-normal ratio) avoids that reciprocal flip and, just
    # as importantly, avoids dividing by sigma22 when sigma22 == 0 (pure
    # shear, which must fall in Mode B, continuous with Mode A at sigma22=0 --
    # verified: both give f=1 at sigma22=0, sigma12=S21s).
    p22m = p22_minus(p21m, S22c, S21s)
    R22A = S22c / (2 * (1 + p22m))
    sigma21c = S21s * math.sqrt(1 + 2 * p22m)
    threshold = sigma21c / R22A

    ratio = math.inf if sigma22 == 0 else abs(sigma12 / sigma22)

    if ratio >= threshold:
        # Mode B
        f = math.sqrt((sigma12 / S21s) ** 2 + (p21m * sigma22 / S21s) ** 2) \
            + p21m * (sigma22 / S21s)
        return f, "B"
    else:
        # Mode C
        f = (
            (sigma12 / (2 * (1 + p22m) * S21s)) ** 2
            + (sigma22 / S22c) ** 2
        ) * (S22c / -sigma22)
        return f, "C"


def ply_failure(sigma11, sigma22, sigma12, strength: PuckStrength):
    """Full per-ply check: fiber failure + inter-fiber failure, combined the
    way Equation (8)/(9) combine them (reserve factor / margin of safety on
    whichever of the two governs).

    Returns a dict:
      f_ff, f_iff, iff_mode, f_max, governing ('FF' or 'IFF'),
      reserve_factor, margin_of_safety, failed (f_max >= 1)
    """
    f_ff = fiber_failure(sigma11, strength)
    f_iff, mode = inter_fiber_failure(sigma22, sigma12, strength)
    f_max = max(f_ff, f_iff)
    governing = "FF" if f_ff >= f_iff else "IFF"

    # Equation (8)/(9). f_max == 0 means a zero stress state -- infinite
    # reserve, guard the division instead of propagating a ZeroDivisionError.
    if f_max <= 0:
        reserve_factor = math.inf
        mos = math.inf
    else:
        reserve_factor = 1.0 / f_max
        mos = reserve_factor - 1.0

    return {
        "f_ff": f_ff,
        "f_iff": f_iff,
        "iff_mode": mode,
        "f_max": f_max,
        "governing": governing,
        "reserve_factor": reserve_factor,
        "margin_of_safety": mos,
        "failed": f_max >= 1.0,
    }


def evaluate_laminate(dat_path, elset_name, strength):
    """Read every (element, integration point, ply layer) stress row from a
    solved .inp's .dat file and run puck.ply_failure on each.

    strength: either one PuckStrength (same material/strength for every
    ply -- matches how this project currently only ever writes one
    *Material for the whole layup) or a {layer_index: PuckStrength} dict
    for a future mixed-material layup.

    Returns:
      {
        "rows": [ {elem, ip, layer, sigma11, sigma22, sigma12, **ply_failure(...)}, ... ],
        "worst": the single row with the highest f_max (the laminate's
                 governing point -- this answers "does it crack, and where"),
        "by_layer": {layer: worst-row-in-that-layer, ...},
        "failed": worst["f_max"] >= 1.0,
      }

    Raises DatParseError if the .dat has no matching stress block (e.g. the
    .inp wasn't written with the *El print/S block this module adds in
    inp_writer.full_inp, or ccx didn't converge).
    """
    raw_rows = read_ply_stresses(dat_path, elset_name)
    if not raw_rows:
        raise DatParseError(f"No stress rows parsed for set {elset_name!r} in {dat_path}.")

    def strength_for(layer):
        if isinstance(strength, PuckStrength):
            return strength
        s = strength.get(layer)
        if s is None:
            raise PuckInputError(f"No Puck strength given for ply layer {layer}.")
        return s

    rows = []
    for r in raw_rows:
        s = strength_for(r["layer"])
        result = ply_failure(r["S11"], r["S22"], r["S12"], s)
        rows.append({
            "elem": r["elem"], "ip": r["ip"], "layer": r["layer"],
            "sigma11": r["S11"], "sigma22": r["S22"], "sigma12": r["S12"],
            **result,
        })

    worst = max(rows, key=lambda r: r["f_max"])

    by_layer = {}
    for r in rows:
        cur = by_layer.get(r["layer"])
        if cur is None or r["f_max"] > cur["f_max"]:
            by_layer[r["layer"]] = r

    return {
        "rows": rows,
        "worst": worst,
        "by_layer": by_layer,
        "failed": worst["f_max"] >= 1.0,
    }
