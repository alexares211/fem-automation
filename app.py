import os
import json
import math
import uuid
import itertools
import threading
from flask import Flask, render_template, request, jsonify

from ply_orientation import generate_inp, angle_range, PlyEditError
from ccx_runner import run_ccx_blocking
from dat_parser import read_reaction_force, DatParseError

DEFAULT_CCX_EXE = r"D:\PrePoMax v2.5.0\Solver\ccx_dynamic.exe"
MAX_COMBOS = 4028

app = Flask(__name__)

JOBS = {}
JOBS_LOCK = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html", default_ccx_exe=DEFAULT_CCX_EXE)


@app.route("/stop/<job_id>", methods=["POST"])
def stop(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown job id."})
    job["cancel_event"].set()
    return jsonify({"ok": True})


# ---------- angle sweep ----------

def run_sweep(job_id, input_path, output_folder, combos, ccx_exe):
    with JOBS_LOCK:
        job = JOBS[job_id]

    try:
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        for i, combo in enumerate(combos):
            if job["cancel_event"].is_set():
                break

            job["current_index"] = i
            job["current_combo"] = list(combo)
            combo_num = i + 1

            suffix = "_".join(
                str(a).replace("-", "m").replace(".", "p") for a in combo
            )
            file_name = base_name + "_" + suffix + "deg.inp"
            output_path = os.path.join(output_folder, file_name)

            # One record per combination. `angles` is the per-ply layup
            # (one entry per ply, stacking order); `rf_by_node` maps node id
            # -> reaction-force magnitude at that constrained node. Reaction
            # force is a property of the whole solved model, not of a ply,
            # so it is not broken out per ply.
            def add_result(rf_by_node=None, error=None):
                job["results"].append(
                    {
                        "combo": combo_num,
                        "angles": list(combo),
                        "rf_by_node": rf_by_node or {},
                        "error": error,
                    }
                )

            try:
                gen_result = generate_inp(input_path, output_path, list(combo))
            except PlyEditError as e:
                add_result(error=str(e))
                continue

            rf_nset = gen_result["rf_nset"]
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
                rf = read_reaction_force(dat_path, rf_nset)
                rf_by_node = {
                    str(node_id): math.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
                    for node_id, (fx, fy, fz) in rf["per_node"].items()
                }
                add_result(rf_by_node=rf_by_node)
            except DatParseError as e:
                add_result(error="Could not read reaction force: " + str(e))

        try:
            summary_path = os.path.join(output_folder, "sweep_results.json")
            with open(summary_path, "w") as f:
                json.dump(job["results"], f, indent=2)
        except OSError:
            pass

    except Exception as e:
        job["error"] = str(e)
    finally:
        job["done"] = True


@app.route("/sweep_start", methods=["POST"])
def sweep_start():
    data = request.get_json()
    input_path = (data.get("input_path") or "").strip()
    output_folder = (data.get("output_folder") or "").strip()
    ccx_exe = (data.get("ccx_exe") or "").strip() or DEFAULT_CCX_EXE

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

    if not input_path or not os.path.isfile(input_path):
        return jsonify({"ok": False, "error": "No file found at: " + input_path})
    if not output_folder:
        return jsonify({"ok": False, "error": "Please provide an output folder for the sweep."})

    try:
        angles = angle_range(angle_min, angle_max, angle_step)
    except PlyEditError as e:
        return jsonify({"ok": False, "error": str(e)})

    combo_count = len(angles) ** num_plies
    if combo_count > MAX_COMBOS:
        return jsonify(
            {
                "ok": False,
                "error": (
                    f"{combo_count} combinations would be generated "
                    f"({len(angles)} angles ^ {num_plies} plies), which exceeds the "
                    f"{MAX_COMBOS}-combination limit. Reduce the angle range/step or ply count."
                ),
            }
        )

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

    thread = threading.Thread(
        target=run_sweep,
        args=(job_id, input_path, output_folder, combos, ccx_exe),
        daemon=True,
    )
    thread.start()

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


if __name__ == "__main__":
    app.run(debug=True, port=5050, threaded=True)
