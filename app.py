import os
import json
import math
import uuid
import itertools
import threading
from flask import Flask, render_template, request, jsonify

from ply_orientation import angle_range, PlyEditError
from ccx_runner import run_ccx_blocking
from dat_parser import read_reaction_force, DatParseError
from step_mesher import structured_grid, StepMeshError
from inp_writer import material_block, section_block, full_inp, LAGER_ALL, InpWriteError

DEFAULT_CCX_EXE = r"D:\PrePoMax v2.5.0\Solver\ccx_dynamic.exe"
MAX_COMBOS = 4028
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
    same file can be re-meshed for the sweep without re-uploading.
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
            d.get("lagers") or [], bool(d.get("symmetric")), bool(d.get("nlgeom", True)),
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


@app.route("/stop/<job_id>", methods=["POST"])
def stop(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown job id."})
    job["cancel_event"].set()
    return jsonify({"ok": True})


# ---------- angle sweep ----------

def run_sweep(job_id, model, output_folder, combos, ccx_exe):
    """model: dict with nodes, elements, element_type, material_name,
    constants, thickness, symmetric, nlgeom, lagers (list of
    {index, node, type}). One full .inp is assembled and solved per angle
    combination; reaction force is read at each Lager and reported per
    Lager index (not per node id).
    """
    with JOBS_LOCK:
        job = JOBS[job_id]

    lager_node = {int(lg["node"]): int(lg["index"]) for lg in model["lagers"]}

    try:
        os.makedirs(output_folder, exist_ok=True)

        for i, combo in enumerate(combos):
            if job["cancel_event"].is_set():
                break

            job["current_index"] = i
            job["current_combo"] = list(combo)
            combo_num = i + 1

            suffix = "_".join(str(a).replace("-", "m").replace(".", "p") for a in combo)
            output_path = os.path.join(output_folder, "sweep_" + suffix + "deg.inp")

            def add_result(rf_by_lager=None, error=None):
                job["results"].append({
                    "combo": combo_num,
                    "angles": list(combo),
                    "rf_by_lager": rf_by_lager or {},
                    "error": error,
                })

            try:
                text = full_inp(
                    model["nodes"], model["elements"], model["element_type"],
                    model["material_name"], model["constants"],
                    list(combo), model["thickness"], model["lagers"],
                    model["symmetric"], model["nlgeom"], model["loads"],
                )
                with open(output_path, "w") as f:
                    f.write(text)
            except InpWriteError as e:
                add_result(error=str(e))
                continue

            run_result = run_ccx_blocking(output_path, ccx_exe, cancel_event=job["cancel_event"])
            job["log_tail"] = run_result["output_tail"]

            if run_result["stopped"]:
                add_result(error="Stopped by user")
                break

            if run_result["has_error"] or not run_result["converged"]:
                add_result(error="CalculiX did not converge cleanly")
                continue

            dat_path = os.path.splitext(output_path)[0] + ".dat"
            try:
                rf = read_reaction_force(dat_path, LAGER_ALL)
                rf_by_lager = {}
                for node_id, (fx, fy, fz) in rf["per_node"].items():
                    idx = lager_node.get(int(node_id))
                    if idx is not None:
                        rf_by_lager[str(idx)] = math.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
                add_result(rf_by_lager=rf_by_lager)
            except DatParseError as e:
                add_result(error="Could not read reaction force: " + str(e))

        try:
            with open(os.path.join(output_folder, "sweep_results.json"), "w") as f:
                json.dump(job["results"], f, indent=2)
        except OSError:
            pass

    except Exception as e:
        job["error"] = str(e)
    finally:
        job["done"] = True


@app.route("/sweep_start", methods=["POST"])
def sweep_start():
    data = request.get_json() or {}
    output_folder = (data.get("output_folder") or "").strip()
    ccx_exe = (data.get("ccx_exe") or "").strip() or DEFAULT_CCX_EXE
    source_path = (data.get("source_path") or "").strip().strip('"')

    try:
        angle_min = float(data.get("angle_min"))
        angle_max = float(data.get("angle_max"))
        angle_step = float(data.get("angle_step"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Angle min/max/step must be numbers."})

    try:
        num_plies = int(data.get("num_plies") or 1)
        if num_plies < 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Number of plies must be a whole number of at least 1."})

    if not output_folder:
        return jsonify({"ok": False, "error": "Please provide an output folder for the sweep."})
    if not source_path or not os.path.isfile(source_path):
        return jsonify({"ok": False, "error": "Process a STEP file first (no meshed geometry)."})
    lagers = data.get("lagers") or []
    loads = data.get("loads") or []
    if not lagers:
        return jsonify({"ok": False, "error": "Place at least one Festlager on the model first."})

    try:
        angles = angle_range(angle_min, angle_max, angle_step)
    except PlyEditError as e:
        return jsonify({"ok": False, "error": str(e)})

    combo_count = len(angles) ** num_plies
    if combo_count > MAX_COMBOS:
        return jsonify({"ok": False, "error": (
            f"{combo_count} combinations ({len(angles)} angles ^ {num_plies} plies) "
            f"exceeds the {MAX_COMBOS} limit. Reduce the angle range/step or ply count."
        )})

    # build the fixed part of the model once (the grid doesn't change per combo)
    try:
        with GMSH_LOCK:
            grid = structured_grid(source_path, spacing=data.get("spacing", "1"))
        # validate material + supports up front by assembling one .inp
        full_inp(
            grid["nodes"], grid["elements"], grid["element_type"],
            data.get("material_name"), data.get("constants"),
            [0.0] * num_plies, data.get("thickness"),
            lagers, bool(data.get("symmetric")), bool(data.get("nlgeom", True)), loads,
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
        "nlgeom": bool(data.get("nlgeom", True)),
        "lagers": lagers,
        "loads": loads,
    }
    combos = list(itertools.product(angles, repeat=num_plies))

    job_id = uuid.uuid4().hex
    job = {
        "results": [],
        "total": len(combos),
        "current_index": -1,
        "current_combo": None,
        "done": False,
        "cancel_event": threading.Event(),
        "log_tail": "",
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(
        target=run_sweep, args=(job_id, model, output_folder, combos, ccx_exe), daemon=True
    ).start()

    return jsonify({"ok": True, "job_id": job_id, "total": len(combos)})


@app.route("/sweep_status/<job_id>")
def sweep_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown sweep job id."})
    return jsonify(
        {
            "ok": True,
            "done": job["done"],
            "total": job["total"],
            "current_index": job["current_index"],
            "current_combo": job["current_combo"],
            "results": job["results"],
            "log_tail": job["log_tail"],
            "fatal_error": job["error"],
        }
    )


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
                model["lagers"], model["symmetric"], model["nlgeom"], model["loads"],
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
            lagers, bool(data.get("symmetric")), bool(data.get("nlgeom", True)), loads,
        )
    except (StepMeshError, InpWriteError) as e:
        return jsonify({"ok": False, "error": str(e)})

    model = {
        "nodes": grid["nodes"], "elements": grid["elements"], "element_type": grid["element_type"],
        "material_name": data.get("material_name"), "constants": data.get("constants"),
        "thickness": data.get("thickness"), "symmetric": bool(data.get("symmetric")),
        "nlgeom": bool(data.get("nlgeom", True)), "lagers": lagers, "loads": loads,
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
