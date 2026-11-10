"""Local Flask workflow for stress-based orientation optimization and FRD viewing."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import os
import shutil
import threading
import uuid

import numpy as np
from flask import Blueprint, jsonify, request, send_file

from inp_import import inspect_inp, splice_layup, InpImportError, shell_reference_axes, parse_cards
from inp_writer import material_block, parse_layup, InpWriteError
from orientation_optimization import SearchSettings, optimize_orientations
from structural_response import CalculixDesignEvaluator, DesignError, analyse_loading, validate_strengths

DEFAULT_OUTPUT_ROOT = os.environ.get("FEM_OUTPUT_ROOT", r"D:\DataBin_Composite_Automation")
DEFAULT_PREPOMAX = os.environ.get("FEM_PREPOMAX", r"D:\PrePoMax v2.5.0\PrePoMax.exe")
DEFAULT_CCX = r"D:\PrePoMax v2.5.0\Solver\ccx_dynamic.exe"
orientation_api = Blueprint("orientation", __name__)
_jobs = {}
_lock = threading.Lock()


def _validated_design(data):
    path = Path(data.get("import_inp_path") or "")
    if not path.is_file():
        raise DesignError("Import a PrePoMax .inp before starting optimization.")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    info = inspect_inp(text)
    selected = str(data.get("import_elset") or "")
    elset = next((s for s in info["elsets"] if s.casefold() == selected.casefold()), None)
    if elset is None:
        raise DesignError("Select an existing element set for the composite layup.")
    if not info["elset_types"].get(elset):
        raise DesignError("The selected element set has no resolvable shell elements.")
    initial = parse_layup(data.get("layup"))
    if not all(math.isfinite(x) for x in initial):
        raise DesignError("Ply angles must be finite numbers.")
    thickness = float(data.get("thickness") or 0)
    if not math.isfinite(thickness) or thickness <= 0:
        raise DesignError("Ply thickness must be finite and positive.")
    if not isinstance(data.get("constants"), dict):
        raise DesignError("Engineering constants must be a named set of numeric values.")
    constants = {k: float(v) for k, v in data["constants"].items()}
    material_block(data.get("material_name"), constants)
    if not all(math.isfinite(x) for x in constants.values()):
        raise DesignError("Engineering constants must be finite.")
    e1, e2, e3 = [constants[k] for k in ("E1", "E2", "E3")]
    n12, n13, n23 = [constants[k] for k in ("nu12", "nu13", "nu23")]
    try:
        np.linalg.cholesky([[1/e1, -n12/e1, -n13/e1], [-n12/e1, 1/e2, -n23/e2], [-n13/e1, -n23/e2, 1/e3]])
    except np.linalg.LinAlgError as exc:
        raise DesignError("These elastic constants do not define a positive-definite orthotropic material.") from exc
    symmetric = data.get("symmetric", False)
    if not isinstance(symmetric, bool):
        raise DesignError("Symmetry must be a true/false value.")
    model = {"import_inp_text": text, "import_elset": elset,
             "import_rf_nset": str(data.get("import_rf_nset") or "").strip() or None,
             "material_name": data.get("material_name"), "constants": constants, "thickness": thickness}
    model["reference_axes"] = shell_reference_axes(parse_cards(text), elset)
    analyse_loading(text)
    full = initial + initial[::-1] if symmetric else initial
    splice_layup(text, elset, model["material_name"], constants, full, thickness, nlgeom=False)
    strengths = validate_strengths(data.get("strengths") or data.get("puck_strength") or {})
    return model, initial, symmetric, strengths


def _launch_result(path):
    if os.name != "nt" or not Path(DEFAULT_PREPOMAX).is_file():
        raise DesignError(f"PrePoMax was not found at {DEFAULT_PREPOMAX}. The FRD remains available for download.")
    if not Path(path).is_file():
        raise DesignError("The optimized .frd file is missing.")
    from prepomax_viewer import open_results
    return open_results(DEFAULT_PREPOMAX, path)


def _archive_result(folder, result, model, strengths):
    for label, response in (("baseline", result.get("baseline")), ("optimized", result.get("best"))):
        if response:
            original = Path(response["inp_path"])
            for extension in (".inp", ".frd", ".dat", ".sta", ".cvg"):
                artifact = original.with_suffix(extension)
                if artifact.exists():
                    shutil.copy2(artifact, folder/(label+extension))
    with (folder/"history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["evaluation", "source", "angles_deg", "2U", "max_displacement", "max_utilization", "utilization_cv", "constraints_met"])
        for row in result["history"]:
            writer.writerow([row["evaluation"], row["source"], "/".join(map(str, row["angles"])),
                             row["compliance"], row["max_displacement"], row["max_utilization"],
                             row["utilization_cv"], row["constraints_met"]])
    report = {**result, "material": model["constants"], "strength_allowables": strengths,
              "orientation_reference_axes": model["reference_axes"],
              "ply_thickness": model["thickness"], "total_thickness": result.get("ply_count", 0)*model["thickness"],
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "definition": "Orientation-only stress-constrained stiffness search; fixed sizing. Equal-stress means CV of per-ply peak maximum-stress utilization.",
              "optimality": "Local pattern-search result at the reported angular resolution or evaluation budget. No global-optimum guarantee."}
    (folder/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def _run(job_id, model, initial, symmetric, strengths, settings, ccx_exe, folder, auto_open):
    with _lock:
        job = _jobs[job_id]
    def on_eval(entry):
        with _lock:
            job["history"].append(entry)
        with (folder/"evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, allow_nan=False)+"\n")
    try:
        folder.mkdir(parents=True, exist_ok=False)
        (folder/"source_prepo.inp").write_text(model["import_inp_text"], encoding="utf-8")
        evaluator = CalculixDesignEvaluator(model, ccx_exe, folder/"evaluations", strengths, job["cancel_event"])
        result = optimize_orientations(evaluator, initial, symmetric=symmetric, settings=settings,
                                      cancel_event=job["cancel_event"], on_evaluation=on_eval)
        result["ply_thickness"] = model["thickness"]
        result["total_thickness"] = result.get("ply_count", 0)*model["thickness"]
        _archive_result(folder, result, model, strengths)
        result["output_folder"] = str(folder)
        result["frd_path"] = str(folder/"optimized.frd") if result.get("best") else None
        with _lock:
            job["result"] = result
        if auto_open and result.get("best") and result.get("constraints_met") and not result["stopped"]:
            with _lock:
                job["phase"] = "opening_results"
            try:
                viewer_info = _launch_result(result["frd_path"])
                with _lock:
                    job["viewer"] = "Optimized displacement results verified in PrePoMax"
                    job["viewer_details"] = viewer_info
            except Exception as exc:
                with _lock:
                    job["viewer_error"] = str(exc)
    except Exception as exc:
        with _lock:
            job["error"] = str(exc)
    finally:
        with _lock:
            job["done"] = True
            job["phase"] = "complete"


@orientation_api.post("/orientation/start")
def start_orientation():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(ok=False, error="Send the optimization settings as a JSON object."), 400
    try:
        model, initial, symmetric, strengths = _validated_design(data)
        raw_budget = float(data.get("max_evaluations", 160))
        if not math.isfinite(raw_budget) or not raw_budget.is_integer():
            raise DesignError("The solve budget must be a whole number.")
        settings = SearchSettings(method=data.get("method", "fully_stressed"), max_evaluations=int(raw_budget),
                                  angle_step=float(data.get("angle_step", 1)),
                                  stress_limit=float(data.get("stress_limit", 1)),
                                  equal_tolerance=float(data.get("equal_tolerance", 0.10))).validate()
        # Validate the initial grid/length before launching a background job.
        if not 1 <= len(initial) <= 64:
            raise DesignError("Use between 1 and 64 independent ply angles.")
        if any(not math.isclose(a/settings.angle_step, round(a/settings.angle_step), abs_tol=1e-7) for a in initial):
            raise DesignError("Starting ply angles must lie on the selected angle grid.")
        output_root = Path(str(data.get("output_folder") or DEFAULT_OUTPUT_ROOT)).resolve()
        ccx_exe = str(data.get("ccx_exe") or DEFAULT_CCX)
        if not Path(ccx_exe).is_file():
            raise DesignError("The CalculiX executable does not exist.")
        auto_open = data.get("auto_open", True)
        if not isinstance(auto_open, bool):
            raise DesignError("Automatic result opening must be true or false.")
    except (ValueError, TypeError, KeyError, OSError, InpWriteError, InpImportError) as exc:
        return jsonify(ok=False, error=str(exc)), 400
    job_id = uuid.uuid4().hex
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = output_root/f"{settings.method}_{stamp}_{job_id[:6]}"
    with _lock:
        _jobs[job_id] = {"done": False, "cancel_event": threading.Event(), "history": [],
                         "result": None, "error": None, "viewer": None, "viewer_error": None,
                         "folder": folder, "settings": asdict(settings), "phase": "solving", "viewer_details": None}
    threading.Thread(target=_run, args=(job_id, model, initial, symmetric, strengths, settings, ccx_exe, folder, auto_open), daemon=True).start()
    return jsonify(ok=True, job_id=job_id, output_folder=str(folder), settings=asdict(settings))


@orientation_api.get("/orientation/status/<job_id>")
def orientation_status(job_id):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify(ok=False, error="Unknown optimization job."), 404
        history = [{k: row[k] for k in ("evaluation", "source", "angles", "compliance", "max_displacement",
                                        "max_utilization", "utilization_cv", "constraints_met", "best_objective", "mode", "objective_value")}
                   for row in job["history"]]
        return jsonify(ok=True, done=job["done"], history=history, result=job["result"],
                       error=job["error"], viewer=job["viewer"], viewer_error=job["viewer_error"],
                       output_folder=str(job["folder"]), settings=job["settings"],
                       phase=job["phase"], viewer_details=job["viewer_details"])


@orientation_api.post("/orientation/stop/<job_id>")
def stop_orientation(job_id):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify(ok=False, error="Unknown optimization job."), 404
        job["cancel_event"].set()
    return jsonify(ok=True)


@orientation_api.post("/orientation/open/<job_id>")
def open_orientation(job_id):
    with _lock:
        job = _jobs.get(job_id)
        result = job["result"] if job else None
    if not result or not result.get("frd_path"):
        return jsonify(ok=False, error="No completed result file is available."), 404
    try:
        details = _launch_result(result["frd_path"])
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 400
    return jsonify(ok=True, viewer_details=details)


@orientation_api.get("/orientation/artifact/<job_id>/<name>")
def orientation_artifact(job_id, name):
    allowed = {"optimized.inp", "optimized.frd", "optimized.dat", "baseline.inp", "baseline.frd",
               "report.json", "history.csv", "source_prepo.inp"}
    with _lock:
        job = _jobs.get(job_id)
        folder = job["folder"] if job else None
    if name not in allowed or folder is None or not (folder/name).is_file():
        return jsonify(ok=False, error="Artifact not found."), 404
    return send_file(folder/name, as_attachment=True)


@orientation_api.get("/examples/cantilever")
def cantilever_example():
    path = Path(DEFAULT_OUTPUT_ROOT)/"cantilever_beam"/"cantilever_prepo.inp"
    if not path.is_file():
        return jsonify(ok=False, error="The PrePoMax cantilever export has not been created yet."), 404
    try:
        info = inspect_inp(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 400
    if "BEAM" in info["elsets"]:
        info["suggested_elset"] = "BEAM"
    return jsonify(ok=True, import_inp_path=str(path), **info)
