"""Fixed-size, stress-constrained minimum-compliance orientation searches.

FSD adaptation: min C(theta), subject to max ply utilization <= limit.
Equal-stress adaptation: same problem, with CV(peak ply utilizations) <= tol.
For prescribed displacements replace min C by max 2U.

Classical fully-stressed sizing is a minimum-weight optimality criterion;
with thickness/count frozen, utilization=1 cannot in general be attained.
Neither adaptation is represented as the classical sizing algorithm or as
a proof of a global optimum. Search uses deterministic multi-seeding and a
stress-ordered coordinate pattern search with decreasing angular moves.
Every candidate is solved; no surrogate score substitutes for FEM results.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

from structural_response import DesignError, DesignCancelled


@dataclass(frozen=True)
class SearchSettings:
    method: str = "fully_stressed"
    max_evaluations: int = 160
    angle_step: float = 1.0
    stress_limit: float = 1.0
    equal_tolerance: float = 0.10

    def validate(self):
        if self.method not in ("fully_stressed", "equal_stress"):
            raise DesignError("Choose fully_stressed or equal_stress.")
        if not 5 <= self.max_evaluations <= 2000:
            raise DesignError("The real-solve budget must be between 5 and 2000.")
        if not math.isfinite(self.angle_step) or not 0.25 <= self.angle_step <= 45:
            raise DesignError("Angle resolution must be between 0.25 and 45 degrees.")
        if not math.isclose(180/self.angle_step, round(180/self.angle_step), abs_tol=1e-8):
            raise DesignError("Angle resolution must divide 180 degrees exactly (for example 1, 2, 5, 15 or 45).")
        if not math.isfinite(self.stress_limit) or not 0 < self.stress_limit <= 1:
            raise DesignError("The utilization limit must be greater than zero and at most 1.")
        if not math.isfinite(self.equal_tolerance) or not 0 <= self.equal_tolerance <= 2:
            raise DesignError("Equal-stress CV tolerance must be between 0 and 2.")
        return self


def canonical(angles):
    return [round((float(a)+90) % 180 - 90, 8) for a in angles]


def constraint_violation(response, settings):
    strength = max(0.0, response["max_utilization"]/settings.stress_limit-1)
    equal = max(0.0, response["utilization_cv"]-settings.equal_tolerance) if settings.method == "equal_stress" else 0.0
    return strength + equal


def optimize_orientations(evaluator, initial_angles, symmetric=False, settings=None,
                          cancel_event=None, on_evaluation=None):
    settings = (settings or SearchSettings()).validate()
    initial = canonical(initial_angles)
    if not initial or len(initial) > 64 or not all(math.isfinite(a) for a in initial):
        raise DesignError("Supply 1 to 64 finite starting ply angles.")
    if any(not math.isclose(a/settings.angle_step, round(a/settings.angle_step), abs_tol=1e-7) for a in initial):
        raise DesignError("Starting angles must lie on the selected angle grid. Choose a finer resolution.")
    n_free = len(initial)
    def expand(free):
        return list(free) + list(reversed(free)) if symmetric else list(free)
    history, cache = [], {}
    best = None
    stopped, local_convergence = False, False

    class BudgetReached(Exception):
        pass

    def rank(record):
        violation = constraint_violation(record, settings)
        return (violation > 1e-8, violation if violation > 1e-8 else 0.0, record["objective_value"])

    def evaluate(free, source):
        nonlocal best
        if cancel_event is not None and cancel_event.is_set():
            raise DesignCancelled("Stopped by user.")
        free = canonical(np.round(np.asarray(free)/settings.angle_step)*settings.angle_step)
        key = tuple(free)
        if key in cache:
            return cache[key]
        if len(history) >= settings.max_evaluations:
            raise BudgetReached
        response = evaluator(expand(free))
        if not math.isfinite(response["objective_value"]) or response["objective_value"] <= 0:
            raise DesignError("The FEM objective must be positive and finite.")
        response = dict(response, free_angles=free, source=source, evaluation=len(history)+1)
        response["constraint_violation"] = constraint_violation(response, settings)
        response["constraints_met"] = response["constraint_violation"] <= 1e-8
        if best is None or rank(response) < rank(best):
            best = response
        response["best_objective"] = best["objective_value"]
        history.append(response)
        cache[key] = response
        if on_evaluation:
            on_evaluation(dict(response))
        return response

    baseline = None
    try:
        baseline = evaluate(initial, "baseline")
        # Diverse deterministic starts, including axial and shear-dominated laminates.
        for angle in (0, 90, 45, -45):
            evaluate([angle]*n_free, "uniform seed")
        current = best
        move = max(settings.angle_step, 45.0)
        while True:
            center = current
            ply_values = np.array([p["utilization"] for p in current["plies"]])
            priorities = ply_values[:n_free].copy()
            if symmetric:
                priorities = np.maximum(priorities, ply_values[::-1][:n_free])
            if settings.method == "equal_stress":
                priorities = np.abs(priorities - float(ply_values.mean()))
            for index in np.argsort(-priorities, kind="stable"):
                choices = [current]
                for sign in (-1, 1):
                    trial = list(current["free_angles"])
                    trial[index] += sign*move
                    choices.append(evaluate(trial, f"ply {int(index)+1}, move {move:g} deg"))
                current = min(choices, key=rank)
            if rank(current) >= rank(center):
                if move <= settings.angle_step:
                    local_convergence = True
                    break
                move = max(settings.angle_step, move/2)
    except BudgetReached:
        pass
    except DesignCancelled:
        stopped = True
    if best is None:
        return {"stopped": stopped, "best": None, "baseline": None, "evaluations": 0,
                "termination": "stopped", "settings": asdict(settings), "history": history}
    improvement = None if baseline is None else baseline["objective_value"]/best["objective_value"]
    return {"method": settings.method, "settings": asdict(settings), "best": best,
            "baseline": baseline, "history": history, "evaluations": len(history),
            "stopped": stopped, "constraints_met": best["constraints_met"],
            "local_convergence": local_convergence,
            "termination": "stopped" if stopped else ("angle resolution reached" if local_convergence else "evaluation budget reached"),
            "stiffness_ratio": improvement,
            "stiffness_gain_percent": None if improvement is None else 100*(improvement-1),
            "fully_stressed_achieved": all(abs(p["utilization"]/settings.stress_limit-1) <= 0.05 for p in best["plies"]),
            "ply_count": len(best["angles"]), "symmetry_enforced": symmetric}
