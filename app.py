import os
import uuid
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify

from ccx_runner import run_ccx_blocking
from dat_parser import read_reaction_force, DatParseError
from step_mesher import structured_grid, StepMeshError
from inp_writer import material_block, section_block, full_inp, LAGER_ALL, InpWriteError
from surrogate import (
    make_ccx_evaluator, bayesian_optimise, genetic_search_then_verify, SurrogateError,
)

DEFAULT_CCX_EXE = r"D:\PrePoMax v2.5.0\Solver\ccx_dynamic.exe"
STEP_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")

app = Flask(__name__)

JOBS = {}
JOBS_LOCK = threading.Lock()
GMSH_LOCK = threading.Lock()  # gmsh keeps global state -> serialise meshing calls


@app.route("/")
def index():
    return render_template("index.html", default_ccx_exe=DEFAULT_CCX_EXE)


def _step_source():
    """Return (step_path, spacing) from a multipart upload or JSON body.
    An uploaded file is saved under uploads/ and its path returned, so the
    same file can be re-meshed later without re-uploading.
    """
    uploaded = request.files.get("step_file")
    if uploaded is not None:
        os.makedirs(STEP_UPLOAD_DIR, exist_ok=True)
        path = os.path.join(STEP_UPLOAD_DIR, os.path.basename(uploaded.filename or "upload.step"))
        uploaded.save(path)
        return path, request.form.get("spacing", "1")
    d = request.get_json(silent=True) or {}
    return (d.get("path") or d.get("source_path") or "").strip().strip('"'), d.get("spacing", "1")


@app.route("/process_step", methods=["POST"])
def process_step():
    """Build a structured S8R grid over the STEP's bounding box and return it
    (geometry for the 3D viewport + the full node table, needed to drop
    supports onto exact nodes).
    """
    step_path, spacing = _step_source()
    if not step_path:
        return jsonify({"ok": False, "error": "Provide a STEP file path or drop a file."})
    try:
        with GMSH_LOCK:
            result = structured_grid(step_path, spacing=spacing)
    except StepMeshError as e:
        return jsonify({"ok": False, "error": str(e)})
    except Exception as e:
        return jsonify({"ok": False, "error": "Meshing failed: " + str(e)})

    return jsonify({"ok": True, "source_path": step_path, **result})


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
    layup + Festlager supports) and return it as text."""
    step_path, spacing = _step_source()
    if not step_path:
        return jsonify({"ok": False, "error": "Process a STEP file first."})
    d = request.get_json(silent=True) or {}
    try:
        angles = _parse_angles(d.get("layup"))
        with GMSH_LOCK:
            m = structured_grid(step_path, spacing=spacing)
        text = full_inp(
            m["nodes"], m["elements"], m["element_type"],
            d.get("material_name"), d.get("constants"),
            angles, d.get("thickness"),
            d.get("lagers") or [], bool(d.get("symmetric")), "auto",
            d.get("loads") or [],
        )
    except (StepMeshError, InpWriteError) as e:
        return jsonify({"ok": False, "error": str(e)})
    except Exception as e:
        return jsonify({"ok": False, "error": "Could not assemble .inp: " + str(e)})
    return jsonify({"ok": True, "inp": text, "num_nodes": m["num_nodes"], "num_elements": m["num_elements"]})


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
            )
        else:
            result = genetic_search_then_verify(
                evaluator, n_plies, symmetric=symmetric,
                n_init=params["n_init"], pop_size=params["pop_size"],
                n_generations=params["n_generations"], top_k=params["top_k"],
                rng=rng, on_eval=on_eval,
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
    search-then-verify) against the real CalculiX pipeline for whatever
    part is currently loaded -- its mesh, Lager placement and loads, not a
    fixed example. Each candidate stacking sequence costs one real ccx
    solve, so this runs as a background job polled via /bo_status.
    """
    data = request.get_json() or {}
    output_folder = (data.get("output_folder") or "").strip()
    ccx_exe = (data.get("ccx_exe") or "").strip() or DEFAULT_CCX_EXE
    source_path = (data.get("source_path") or "").strip().strip('"')
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
        }
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Bad numeric input in the optimization settings."})
    if params["reduce"] not in ("sum", "max"):
        return jsonify({"ok": False, "error": "reduce must be 'sum' or 'max'."})

    if not output_folder:
        return jsonify({"ok": False, "error": "Please provide an output folder for the search."})
    if not source_path or not os.path.isfile(source_path):
        return jsonify({"ok": False, "error": "Process a STEP file first (no meshed geometry)."})
    lagers = data.get("lagers") or []
    loads = data.get("loads") or []
    if not lagers:
        return jsonify({"ok": False, "error": "Place at least one Festlager on the model first."})

    try:
        with GMSH_LOCK:
            grid = structured_grid(source_path, spacing=data.get("spacing", "1"))
        full_inp(
            grid["nodes"], grid["elements"], grid["element_type"],
            data.get("material_name"), data.get("constants"),
            [0.0] * n_plies, data.get("thickness"),
            lagers, bool(data.get("symmetric")), "auto", loads,
        )
    except (StepMeshError, InpWriteError) as e:
        return jsonify({"ok": False, "error": str(e)})

    model = {
        "nodes": grid["nodes"],
        "elements": grid["elements"],
        "element_type": grid["element_type"],
        "material_name": data.get("material_name"),
        "constants": data.get("constants"),
        "thickness": data.get("thickness"),
        "symmetric": bool(data.get("symmetric")),
        "lagers": lagers,
        "loads": loads,
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

def run_simulate(job_id, model, angles, output_folder, ccx_exe):
    """Assemble one full .inp with the given layup, solve it, write the result
    files into output_folder, and read the reaction force at each Lager."""
    with JOBS_LOCK:
        job = JOBS[job_id]
    lager_node = {int(l["node"]): int(l["index"]) for l in model["lagers"]}
    try:
        os.makedirs(output_folder, exist_ok=True)
        inp_path = os.path.join(output_folder, "model.inp")
        with open(inp_path, "w") as f:
            f.write(full_inp(
                model["nodes"], model["elements"], model["element_type"],
                model["material_name"], model["constants"], angles, model["thickness"],
                model["lagers"], model["symmetric"], "auto", model["loads"],
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
        try:
            rf = read_reaction_force(dat_path, LAGER_ALL)
            job["rf_by_lager"] = {
                str(lager_node[int(nid)]): [fx, fy, fz]   # signed reaction force X, Y, Z
                for nid, (fx, fy, fz) in rf["per_node"].items() if int(nid) in lager_node
            }
        except DatParseError as e:
            job["error"] = "Could not read reaction force: " + str(e)
    except Exception as e:
        job["error"] = str(e)
    finally:
        job["done"] = True


@app.route("/simulate_start", methods=["POST"])
def simulate_start():
    data = request.get_json() or {}
    output_folder = (data.get("output_folder") or "").strip()
    ccx_exe = (data.get("ccx_exe") or "").strip() or DEFAULT_CCX_EXE
    source_path = (data.get("source_path") or "").strip().strip('"')
    lagers = data.get("lagers") or []
    loads = data.get("loads") or []

    if not output_folder:
        return jsonify({"ok": False, "error": "Enter an output folder."})
    if not source_path or not os.path.isfile(source_path):
        return jsonify({"ok": False, "error": "Process a STEP file first (no meshed geometry)."})
    if not lagers:
        return jsonify({"ok": False, "error": "Place at least one Festlager on the model first."})

    try:
        angles = _parse_angles(data.get("layup"))
        with GMSH_LOCK:
            grid = structured_grid(source_path, spacing=data.get("spacing", "1"))
        full_inp(
            grid["nodes"], grid["elements"], grid["element_type"],
            data.get("material_name"), data.get("constants"), angles, data.get("thickness"),
            lagers, bool(data.get("symmetric")), "auto", loads,
        )
    except (StepMeshError, InpWriteError) as e:
        return jsonify({"ok": False, "error": str(e)})

    model = {
        "nodes": grid["nodes"], "elements": grid["elements"], "element_type": grid["element_type"],
        "material_name": data.get("material_name"), "constants": data.get("constants"),
        "thickness": data.get("thickness"), "symmetric": bool(data.get("symmetric")),
        "lagers": lagers, "loads": loads,
    }
    job_id = uuid.uuid4().hex
    job = {
        "done": False, "cancel_event": threading.Event(), "log_tail": "",
        "error": None, "converged": False, "rf_by_lager": {}, "inp_path": None,
        "output_folder": output_folder,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(
        target=run_simulate, args=(job_id, model, angles, output_folder, ccx_exe), daemon=True
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
        "rf_by_lager": job.get("rf_by_lager", {}),
        "output_folder": job.get("output_folder", ""),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050, threaded=True)
