"""Linear-static stiffness and ply strength measured with real CalculiX solves.

Compliance = 2 * integrated elastic strain energy (ELSE), equal to F.T @ u
for fixed applied forces and homogeneous supports. Under prescribed motion,
2U is instead a stiffness measure to MAXIMIZE. These objectives cannot be
interchanged. Unsupported analysis conditions are rejected before searching.
See the CalculiX 2.22 manual, *EL PRINT and *NODE PRINT.
"""
from pathlib import Path
import math
import re

import numpy as np

from ccx_runner import run_ccx_blocking
from dat_parser import read_internal_energy, read_displacements, read_ply_stresses
from inp_import import parse_cards, render_cards, _keyword_params, splice_layup, shell_reference_axes

ALL_NODES = "FEMOPT_ALL_NODES"
ALL_ELEMENTS = "FEMOPT_ALL_ELEMENTS"


class DesignError(ValueError):
    pass


class DesignCancelled(DesignError):
    pass


def data_lines(card):
    return [line.strip() for line in card["body"] if line.strip() and not line.lstrip().startswith("**")]


def analyse_loading(text):
    """Classify a single linear-elastic static load case without changing it."""
    cards = parse_cards(text)
    if sum(c["keyword"] == "STEP" for c in cards) != 1:
        raise DesignError("Orientation optimization requires exactly one static load step.")
    keywords = {c["keyword"] for c in cards}
    unsupported = keywords & {"DYNAMIC", "FREQUENCY", "BUCKLE", "PLASTIC", "CREEP",
                             "HYPERELASTIC", "VISCOELASTIC", "CONTACT PAIR", "TEMPERATURE",
                             "INITIAL CONDITIONS", "DSLOAD", "INCLUDE", "SUBMODEL", "PRESTRESS"}
    if unsupported:
        raise DesignError("Linear stiffness optimization does not support: " + ", ".join(sorted(unsupported)))
    if "STATIC" not in keywords:
        raise DesignError("A *STATIC step is required for stiffness optimization.")
    step = next(c for c in cards if c["keyword"] == "STEP")
    parameters = _keyword_params(step["header"])
    if parameters.get("NLGEOM", "NO").upper() not in ("NO", "OFF", "0") or re.search(r",\s*NLGEOM\s*(?:,|$)", step["header"], re.I):
        raise DesignError("Use a linear static step (NLGEOM=NO). A nonlinear response is not a linear stiffness objective.")
    imposed = False
    for card in cards:
        if card["keyword"] == "BOUNDARY":
            for line in data_lines(card):
                values = [x.strip() for x in line.split(",")]
                if len(values) >= 4:
                    try:
                        value = float(values[3])
                    except ValueError as exc:
                        raise DesignError("Boundary values must be evaluated numeric values.") from exc
                    imposed |= value != 0
        if card["keyword"] in ("CLOAD", "DLOAD") and "AMPLITUDE" in _keyword_params(card["header"]):
            raise DesignError("Use constant end loads without an AMPLITUDE for this single-load-case optimizer.")
    forces = any(c["keyword"] in ("CLOAD", "DLOAD") and data_lines(c) for c in cards)
    if imposed and forces:
        raise DesignError("Mixed nonzero imposed displacement and force loading needs a separate objective; use one control mode.")
    if not imposed and not forces:
        raise DesignError("The input has no applied force or nonzero imposed displacement.")
    return {"mode": "displacement" if imposed else "force",
            "objective": "maximize strain energy at fixed displacement" if imposed else "minimize compliance at fixed force"}


def add_response_output(text, elset):
    """Add independent whole-model energy/deflection outputs to a spliced deck."""
    cards = parse_cards(text)
    nodes, elements = [], []
    for card in cards:
        if card["keyword"] == "NODE":
            nodes.extend(int(line.split(",")[0]) for line in data_lines(card))
        elif card["keyword"] == "ELEMENT":
            element_type = _keyword_params(card["header"]).get("TYPE", "").upper()
            special = {"B31": 2, "B31R": 2, "B32": 3, "B32R": 3,
                       "SPRING1": 1, "SPRING2": 2, "SPRINGA": 2, "MASS": 1, "ROTARYI": 1}
            match = re.match(r"^(?:C3D|CPS|CPE|CAX|T3D|S|M3D)(\d+)", element_type)
            count = special.get(element_type, int(match[1]) if match else 0)
            if not count:
                raise DesignError(f"Cannot collect whole-model energy for element layout {element_type!r}.")
            remaining = 0
            for line in data_lines(card):
                fields = [field.strip() for field in line.split(",") if field.strip()]
                if not remaining:
                    elements.append(int(fields[0]))
                    remaining = count-(len(fields)-1)
                else:
                    remaining -= len(fields)
                if remaining < 0:
                    raise DesignError(f"Unexpected connectivity length for {element_type}.")
            if remaining:
                raise DesignError(f"Incomplete element connectivity for {element_type}.")
    if not nodes or not elements:
        raise DesignError("The exported model must contain its nodes and elements directly.")
    names = {v.upper() for c in cards for v in _keyword_params(c["header"]).values()}
    if ALL_NODES in names or ALL_ELEMENTS in names:
        raise DesignError("The import uses a reserved FEMOPT output set name.")
    def set_card(kind, name, ids):
        return {"keyword": kind, "header": f"*{kind}, {kind}={name}",
                "body": [", ".join(map(str, ids[i:i+16])) for i in range(0, len(ids), 16)]}
    index = next(i for i, c in enumerate(cards) if c["keyword"] == "STEP")
    cards[index:index] = [set_card("NSET", ALL_NODES, sorted(set(nodes))),
                          set_card("ELSET", ALL_ELEMENTS, sorted(set(elements)))]
    index = next(i for i, c in enumerate(cards) if c["keyword"] == "END STEP")
    cards[index:index] = [
        {"keyword": "NODE PRINT", "header": f"*NODE PRINT, NSET={ALL_NODES}, GLOBAL=YES", "body": ["U, RF"]},
        {"keyword": "EL PRINT", "header": f"*EL PRINT, ELSET={ALL_ELEMENTS}, TOTALS=ONLY", "body": ["ELSE"]},
        {"keyword": "EL PRINT", "header": f"*EL PRINT, ELSET={elset}, GLOBAL=NO", "body": ["S"]},
        {"keyword": "NODE FILE", "header": "*NODE FILE", "body": ["U"]},
        {"keyword": "EL FILE", "header": "*EL FILE", "body": ["S"]},
    ]
    return render_cards(cards) + "\n"


def validate_strengths(raw):
    try:
        strengths = {key: float(raw[key]) for key in ("S11t", "S11c", "S22t", "S22c", "S21s")}
    except (KeyError, TypeError, ValueError) as exc:
        raise DesignError("Enter all five ply strength allowables in Material to use stress-based optimization.") from exc
    if any(not math.isfinite(x) or x <= 0 for x in strengths.values()):
        raise DesignError("All five strength allowables must be finite and positive.")
    return strengths


def stress_metrics(rows, n_plies, strengths):
    """Maximum-stress criterion, in the local material axes, at EVERY point.

    Equal stress means equal peak normalized utilization per ply, not equality
    between anisotropic stress components or their dimensional magnitudes.
    """
    plies = [{"ply": i+1, "utilization": 0.0, "max_abs_S11": 0.0,
              "max_abs_S22": 0.0, "max_abs_S12": 0.0, "points": 0} for i in range(n_plies)]
    for row in rows:
        if not 1 <= row["layer"] <= n_plies:
            raise DesignError("Solver ply indexing differs from the fixed layup.")
        ply = plies[row["layer"]-1]
        s1, s2, shear = (float(row[k]) for k in ("S11", "S22", "S12"))
        if not all(math.isfinite(x) for x in (s1, s2, shear)):
            raise DesignError("Non-finite ply stresses in solver output.")
        ratios = [abs(s1)/strengths["S11t" if s1 >= 0 else "S11c"],
                  abs(s2)/strengths["S22t" if s2 >= 0 else "S22c"], abs(shear)/strengths["S21s"]]
        ply["utilization"] = max(ply["utilization"], *ratios)
        for key, value in zip(("S11", "S22", "S12"), (s1, s2, shear)):
            ply["max_abs_"+key] = max(ply["max_abs_"+key], abs(value))
        ply["points"] += 1
    if any(not p["points"] for p in plies):
        raise DesignError("Missing integration-point stress output for one or more plies.")
    values = np.array([p["utilization"] for p in plies])
    mean = float(values.mean())
    return {"plies": plies, "max_utilization": float(values.max()),
            "utilization_cv": float(values.std()/mean) if mean > 1e-15 else 0.0}


class CalculixDesignEvaluator:
    def __init__(self, model, ccx_exe, output_folder, strengths, cancel_event=None):
        self.model, self.ccx_exe = model, ccx_exe
        self.folder = Path(output_folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.strengths = validate_strengths(strengths)
        self.cancel_event = cancel_event
        self.loading = analyse_loading(model["import_inp_text"])
        self.reference_axes = shell_reference_axes(parse_cards(model["import_inp_text"]), model["import_elset"])
        self.count = 0

    def __call__(self, angles):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise DesignCancelled("Stopped by user.")
        self.count += 1
        path = self.folder / f"design_{self.count:04d}.inp"
        text = splice_layup(self.model["import_inp_text"], self.model["import_elset"],
                            self.model["material_name"], self.model["constants"], angles,
                            self.model["thickness"], symmetric=False,
                            rf_nset_name=self.model.get("import_rf_nset"), nlgeom=False,
                            reference_axes=self.reference_axes)
        path.write_text(add_response_output(text, self.model["import_elset"]), encoding="utf-8")
        solve = run_ccx_blocking(str(path), self.ccx_exe, cancel_event=self.cancel_event)
        if solve["stopped"]:
            raise DesignCancelled("Stopped by user.")
        if not solve["converged"] or not solve["frd_exists"]:
            raise DesignError(f"CalculiX failed for {path.name}. See {solve['log_path']}\n{solve['output_tail']}")
        dat = str(path.with_suffix(".dat"))
        energy = read_internal_energy(dat, ALL_ELEMENTS)["energy"]
        motion = read_displacements(dat, ALL_NODES)
        metrics = stress_metrics(read_ply_stresses(dat, self.model["import_elset"]), len(angles), self.strengths)
        return {"angles": list(angles), "strain_energy": energy, "compliance": 2*energy,
                "objective_value": 2*energy if self.loading["mode"] == "force" else 1/(2*energy),
                "max_displacement": motion["maximum"], "mode": self.loading["mode"],
                "inp_path": str(path), "frd_path": str(path.with_suffix('.frd')),
                "dat_path": dat, "log_path": solve["log_path"], **metrics}
