import os
import uuid
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify

from ccx_runner import run_ccx_blocking
from dat_parser import read_reaction_force, DatParseError
from inp_writer import material_block, section_block, ELSET_NAME, InpWriteError
from inp_import import inspect_inp, splice_layup, InpImportError
from puck import PuckStrength, PuckInputError, evaluate_laminate
from surrogate import (
    make_ccx_evaluator, bayesian_optimise, genetic_search_then_verify, SurrogateError,
)

DEFAULT_CCX_EXE = r"D:\PrePoMax v2.5.0\Solver\ccx_dynamic.exe"

app = Flask(__name__)

JOBS = {}
JOBS_LOCK = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html", default_ccx_exe=DEFAULT_CCX_EXE)


INP_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")


@app.route("/import_inp", methods=["POST"])
def import_inp():
    """Accept a PrePoMax-exported .inp (run once with no material, so ccx
    errors out on the undefined material -- expected). Saves it and returns
    what the UI needs to let the user pick the composite Elset and the
    reaction-force Nset: every Nset/Elset found, plus whichever Elset the
    (now-stripped) incomplete Shell/Solid section card already pointed at,
    as a default guess.
    """
    uploaded = request.files.get("inp_file")
    if uploaded is None:
        return jsonify({"ok": False, "error": "Drop the .inp file PrePoMax produced."})
    os.makedirs(INP_UPLOAD_DIR, exist_ok=True)
    path = os.path.join(INP_UPLOAD_DIR, os.path.basename(uploaded.filename or "import.inp"))
    uploaded.save(path)
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read()
        info = inspect_inp(text)
    except InpImportError as e:
        return jsonify({"ok": False, "error": str(e)})
    except Exception as e:
        return jsonify({"ok": False, "error": "Could not read the .inp: " + str(e)})
    return jsonify({"ok": True, "import_inp_path": path, **info})


@app.route("/material_block", methods=["POST"])
def material_block_route():
    """Format the material inputs into *Material / *Elastic keyword text (preview)."""
    data = request.get_json(silent=True) or {}
    try:
        block = material_block(data.get("name"), data.get("constants"))
    except InpWriteError as e:
        return jsonify({"ok": False, "error": str(e)})
    return jsonify({"ok": True, "block": block})


@app.route("/section_block", methods=["POST"])
def section_block_route():
    """Format the layup inputs into *Orientation + *Shell section text (preview)."""
    data = request.get_json(silent=True) or {}
    symmetric = bool(data.get("symmetric"))
    try:
        angles = _parse_angles(data.get("layup"))
        block = section_block(
            data.get("material"), angles, data.get("thickness"), symmetric,
        )
    except InpWriteError as e:
        return jsonify({"ok": False, "error": str(e)})
    return jsonify({"ok": True, "block": block, "num_plies": len(angles) * (2 if symmetric else 1)})


@app.route("/inp_preview", methods=["POST"])
def inp_preview():
    """Assemble the complete .inp from the on-page model (mesh + material +
    layup) and return it as text."""
    d = request.get_json(silent=True) or {}
    import_inp_path = (d.get("import_inp_path") or "").strip()

    try:
        angles = _parse_angles(d.get("layup"))
    except InpWriteError as e:
        return jsonify({"ok": False, "error": str(e)})

    if not import_inp_path:
        return jsonify({"ok": False, "error": "Import a PrePoMax .inp first."})
    if not os.path.isfile(import_inp_path):
        return jsonify({"ok": False, "error": "Imported .inp file not found -- upload it again."})
    import_elset = (d.get("import_elset") or "").strip()
    if not import_elset:
        return jsonify({"ok": False, "error": "Pick the element set the composite layup applies to."})
    with open(import_inp_path, "r", errors="ignore") as f:
        import_inp_text = f.read()
    try:
        text = splice_layup(
            import_inp_text, import_elset, d.get("material_name"), d.get("constants"),
            angles, d.get("thickness"), bool(d.get("symmetric")),
            rf_nset_name=(d.get("import_rf_nset") or "").strip() or None,
        )
    except (InpWriteError, InpImportError) as e:
        return jsonify({"ok": False, "error": str(e)})
    return jsonify({"ok": True, "inp": text, "num_nodes": "--", "num_elements": "--"})


def _parse_angles(text):
    """'0/45/-45/90' or '0, 45' -> [0.0, 45.0, -45.0, 90.0]."""
    if not str(text or "").strip():
        raise InpWriteError("Enter at least one ply angle.")
    parts = [p.strip() for chunk in str(text).split("/") for p in chunk.split(",")]
    try:
        vals = [float(p) for p in parts if p]
    except ValueError:
        raise InpWriteError("Layup must be numbers separated by '/' or ','.")
    if not vals:
        raise InpWriteError("Enter at least one ply angle.")
    return vals


def _puck_strength_from_dict(d):
    """{"S11t":.., "S11c":.., "S22t":.., "S22c":.., "S21s":.., "p21_plus":..,
    "p21_minus": (optional)} -> PuckStrength. Returns None if d is empty/not
    given, so a Puck check is simply skipped when no strength data is
    provided (the simulate/optimization pipeline works exactly as before)."""
    if not d:
        return None
    try:
        return PuckStrength(
            S11t=float(d["S11t"]), S11c=float(d["S11c"]),
            S22t=float(d["S22t"]), S22c=float(d["S22c"]),
            S21s=float(d["S21s"]), p21_plus=float(d["p21_plus"]),
            p21_minus=float(d["p21_minus"]) if d.get("p21_minus") not in (None, "") else None,
        )
    except (KeyError, TypeError, ValueError) as e:
        raise PuckInputError(
            "Puck strength needs numeric S11t, S11c, S22t, S22c, S21s, p21_plus "
            "(p21_minus optional, defaults to p21_plus)."
        ) from e


def run_bo(job_id, model, output_folder, ccx_exe, n_plies, symmetric, method, params):
    """Background worker: builds the real ccx evaluator for this part's
    model (whatever its geometry, Lager placement and loads are) and runs
    either sequential Bayesian Optimization or the genetic
    search-then-verify pattern against it, streaming every real evaluation
    into job["history"] as it happens so /bo_status can show live progress
    -- one real CalculiX solve per row.
    """
    with JOBS_LOCK:
        job = JOBS[job_id]

    def on_eval(entry):
        job["history"].append(entry)
        job["current_index"] = entry.get("n_evaluations", len(job["history"])) - 1

    try:
        os.makedirs(output_folder, exist_ok=True)
        evaluator = make_ccx_evaluator(
            model, ccx_exe, output_folder,
            reduce=params.get("reduce", "sum"), cancel_event=job["cancel_event"],
        )
        rng = np.random.default_rng(params["seed"]) if params.get("seed") is not None else None

        if method == "bayesian":
            result = bayesian_optimise(
                evaluator, n_plies, symmetric=symmetric,
                n_init=params["n_init"], n_iter=params["n_iter"],
                xi=params["xi"], rng=rng, on_eval=on_eval,
                angle_step=params["angle_step"],
            )
        else:
            result = genetic_search_then_verify(
                evaluator, n_plies, symmetric=symmetric,
                n_init=params["n_init"], pop_size=params["pop_size"],
                n_generations=params["n_generations"], top_k=params["top_k"],
                rng=rng, on_eval=on_eval,
                angle_step=params["angle_step"],
            )
        job["result"] = result
    except SurrogateError as e:
        job["error"] = str(e)
    except Exception as e:
        job["error"] = str(e)
    finally:
        job["done"] = True


@app.route("/bo_start", methods=["POST"])
def bo_start():
    """Start a stacking-sequence search (Bayesian Optimization, or genetic
    search-then-verify) against the real CalculiX pipeline, using whatever
    geometry/mesh/supports/loads the imported PrePoMax .inp defines -- only
    the ply angles change between evaluations. Each candidate stacking
    sequence costs one real ccx solve, so this runs as a background job
    polled via /bo_status.
    """
    data = request.get_json() or {}
    output_folder = (data.get("output_folder") or "").strip()
    ccx_exe = (data.get("ccx_exe") or "").strip() or DEFAULT_CCX_EXE
    import_inp_path = (data.get("import_inp_path") or "").strip()
    method = (data.get("method") or "bayesian").strip().lower()
    if method not in ("bayesian", "genetic"):
        return jsonify({"ok": False, "error": "method must be 'bayesian' or 'genetic'."})

    try:
        n_plies = int(data.get("num_plies") or 1)
        if n_plies < 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Number of plies must be a whole number of at least 1."})

    try:
        params = {
            "reduce": (data.get("reduce") or "sum").strip().lower(),
            "seed": int(data["seed"]) if str(data.get("seed") or "").strip() else None,
            "n_init": int(data.get("n_init") or (8 if method == "bayesian" else 20)),
            "n_iter": int(data.get("n_iter") or 15),
            "xi": float(data.get("xi") or 0.01),
            "pop_size": int(data.get("pop_size") or 200),
            "n_generations": int(data.get("n_generations") or 60),
            "top_k": int(data.get("top_k") or 3),
            "angle_step": float(data["angle_step"]) if str(data.get("angle_step") or "").strip() else None,
        }
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Bad numeric input in the optimization settings."})
    if params["reduce"] not in ("sum", "max"):
        return jsonify({"ok": False, "error": "reduce must be 'sum' or 'max'."})
    if params["angle_step"] is not None and not (0 < params["angle_step"] <= 180):
        return jsonify({"ok": False, "error": "Angle step must be a positive number, at most 180."})

    if not output_folder:
        return jsonify({"ok": False, "error": "Please provide an output folder for the search."})

    if not import_inp_path:
        return jsonify({"ok": False, "error": "Import a PrePoMax .inp first."})
    if not os.path.isfile(import_inp_path):
        return jsonify({"ok": False, "error": "Imported .inp file not found -- upload it again."})
    import_elset = (data.get("import_elset") or "").strip()
    if not import_elset:
        return jsonify({"ok": False, "error": "Pick the element set the composite layup applies to."})
    import_rf_nset = (data.get("import_rf_nset") or "").strip() or None
    with open(import_inp_path, "r", errors="ignore") as f:
        import_inp_text = f.read()
    try:
        splice_layup(
            import_inp_text, import_elset, data.get("material_name"), data.get("constants"),
            [0.0] * n_plies, data.get("thickness"), bool(data.get("symmetric")),
            rf_nset_name=import_rf_nset,
        )
    except (InpWriteError, InpImportError) as e:
        return jsonify({"ok": False, "error": str(e)})
    model = {
        "import_inp_text": import_inp_text, "import_elset": import_elset,
        "import_rf_nset": import_rf_nset,
        "material_name": data.get("material_name"), "constants": data.get("constants"),
        "thickness": data.get("thickness"), "symmetric": bool(data.get("symmetric")),
    }

    # planned real-solve count, known up front: bayesian is init + every EI
    # step; genetic is init (real solves used to fit the GP) + the top_k
    # winners confirmed for real at the end (the GA generations in between
    # only ever touch the free surrogate, not ccx).
    total = params["n_init"] + (params["n_iter"] if method == "bayesian" else params["top_k"])

    job_id = uuid.uuid4().hex
    job = {
        "history": [],
        "result": None,
        "current_index": -1,
        "total": total,
        "done": False,
        "cancel_event": threading.Event(),
        "log_tail": "",
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(
        target=run_bo,
        args=(job_id, model, output_folder, ccx_exe, n_plies, bool(data.get("symmetric")), method, params),
        daemon=True,
    ).start()

    return jsonify({"ok": True, "job_id": job_id, "method": method})


@app.route("/bo_status/<job_id>")
def bo_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown search job id."})
    return jsonify({
        "ok": True,
        "done": job["done"],
        "history": job["history"],
        "current_index": job["current_index"],
        "total": job["total"],
        "result": job["result"],
        "fatal_error": job["error"],
    })


@app.route("/stop/<job_id>", methods=["POST"])
def stop(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown job id."})
    job["cancel_event"].set()
    return jsonify({"ok": True})


# ---------- single simulate ----------

def run_simulate(job_id, model, angles, output_folder, ccx_exe, puck_strength=None):
    """Splice one full .inp (imported PrePoMax geometry/mesh/BCs/loads + this
    app's material/layup) with the given ply angles, solve it, write the
    result files into output_folder, read the reaction force for the chosen
    Nset, and -- if puck_strength was given -- also run the Puck failure
    check on every ply's stress from the same solve (one ccx run covers
    both; the .inp already requests per-ply stress output unconditionally).
    """
    with JOBS_LOCK:
        job = JOBS[job_id]
    elset_for_puck = model.get("import_elset") or ELSET_NAME
    rf_nset = model.get("import_rf_nset")
    try:
        os.makedirs(output_folder, exist_ok=True)
        inp_path = os.path.join(output_folder, "model.inp")
        with open(inp_path, "w") as f:
            f.write(splice_layup(
                model["import_inp_text"], model.get("import_elset"),
                model["material_name"], model["constants"], angles, model["thickness"],
                model["symmetric"], rf_nset_name=rf_nset, nlgeom="auto", loads=None,
            ))
        job["inp_path"] = inp_path

        run_result = run_ccx_blocking(inp_path, ccx_exe, cancel_event=job["cancel_event"])
        job["log_tail"] = run_result["output_tail"]
        job["converged"] = run_result["converged"]

        if run_result["stopped"]:
            job["error"] = "Stopped by user"
            return
        if run_result["has_error"] or not run_result["converged"]:
            job["error"] = "CalculiX did not converge cleanly (see log)"
            return

        dat_path = os.path.splitext(inp_path)[0] + ".dat"
        if rf_nset:
            try:
                rf = read_reaction_force(dat_path, rf_nset)
                job["rf_total"] = {
                    "nset": rf_nset, "fx": rf["sum_fx"], "fy": rf["sum_fy"],
                    "fz": rf["sum_fz"], "magnitude": rf["magnitude"],
                }
            except DatParseError as e:
                job["error"] = "Could not read reaction force: " + str(e)
                return

        if puck_strength is not None:
            try:
                job["puck"] = evaluate_laminate(dat_path, elset_for_puck, puck_strength)
            except DatParseError as e:
                job["error"] = "Could not run Puck check: " + str(e)
    except Exception as e:
        job["error"] = str(e)
    finally:
        job["done"] = True


@app.route("/simulate_start", methods=["POST"])
def simulate_start():
    data = request.get_json() or {}
    output_folder = (data.get("output_folder") or "").strip()
    ccx_exe = (data.get("ccx_exe") or "").strip() or DEFAULT_CCX_EXE
    import_inp_path = (data.get("import_inp_path") or "").strip()

    if not output_folder:
        return jsonify({"ok": False, "error": "Enter an output folder."})

    try:
        angles = _parse_angles(data.get("layup"))
        puck_strength = _puck_strength_from_dict(data.get("puck_strength"))
    except (InpWriteError, PuckInputError) as e:
        return jsonify({"ok": False, "error": str(e)})

    if not import_inp_path:
        return jsonify({"ok": False, "error": "Import a PrePoMax .inp first."})
    if not os.path.isfile(import_inp_path):
        return jsonify({"ok": False, "error": "Imported .inp file not found -- upload it again."})
    import_elset = (data.get("import_elset") or "").strip()
    if not import_elset:
        return jsonify({"ok": False, "error": "Pick the element set the composite layup applies to."})
    import_rf_nset = (data.get("import_rf_nset") or "").strip() or None
    with open(import_inp_path, "r", errors="ignore") as f:
        import_inp_text = f.read()
    try:
        splice_layup(
            import_inp_text, import_elset, data.get("material_name"), data.get("constants"),
            angles, data.get("thickness"), bool(data.get("symmetric")),
            rf_nset_name=import_rf_nset,
        )
    except (InpWriteError, InpImportError) as e:
        return jsonify({"ok": False, "error": str(e)})
    model = {
        "import_inp_text": import_inp_text, "import_elset": import_elset,
        "import_rf_nset": import_rf_nset,
        "material_name": data.get("material_name"), "constants": data.get("constants"),
        "thickness": data.get("thickness"), "symmetric": bool(data.get("symmetric")),
    }

    job_id = uuid.uuid4().hex
    job = {
        "done": False, "cancel_event": threading.Event(), "log_tail": "",
        "error": None, "converged": False, "rf_total": None, "inp_path": None,
        "output_folder": output_folder, "puck": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(
        target=run_simulate,
        args=(job_id, model, angles, output_folder, ccx_exe, puck_strength),
        daemon=True,
    ).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/simulate_status/<job_id>")
def simulate_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown job id."})
    return jsonify({
        "ok": True,
        "done": job["done"],
        "converged": job.get("converged", False),
        "error": job["error"],
        "log_tail": job["log_tail"],
        "rf_total": job.get("rf_total"),
        "output_folder": job.get("output_folder", ""),
        "puck": job.get("puck"),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050, threaded=True)
