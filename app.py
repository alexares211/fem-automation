import os
import json
import math
import uuid
import threading
from flask import Flask, render_template, request, jsonify

from ply_orientation import generate_inp, angle_range, PlyEditError
from ccx_runner import start_ccx, run_ccx_blocking, read_log_tail, evaluate_result, CCXRunError
from dat_parser import read_reaction_force, DatParseError

DEFAULT_CCX_EXE = r"D:\PrePoMax v2.5.0\Solver\ccx_dynamic.exe"

app = Flask(__name__)

JOBS = {}
JOBS_LOCK = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html", default_ccx_exe=DEFAULT_CCX_EXE)


# ---------- single-run ----------

@app.route("/start", methods=["POST"])
def start():
    data = request.get_json()
    input_path = (data.get("input_path") or "").strip()
    output_folder = (data.get("output_folder") or "").strip()
    angle_raw = (data.get("angle") or "").strip()
    ccx_exe = (data.get("ccx_exe") or "").strip() or DEFAULT_CCX_EXE

    try:
        angle = float(angle_raw)
        if not input_path:
            raise PlyEditError("Please provide the path to the input .inp file.")
        if not os.path.isfile(input_path):
            raise PlyEditError("No file found at: " + input_path)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        suffix = angle_raw.replace("-", "m").replace(".", "p")
        file_name = base_name + "_" + suffix + "deg.inp"

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, file_name)
        else:
            output_path = os.path.join(os.path.dirname(input_path), file_name)

        gen_result = generate_inp(input_path, output_path, angle)
        ccx_state = start_ccx(output_path, ccx_exe)

        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {
                "type": "single",
                "proc": ccx_state["proc"],
                "log_file": ccx_state["log_file"],
                "log_path": ccx_state["log_path"],
                "frd_path": ccx_state["frd_path"],
                "jobname": ccx_state["jobname"],
                "stopped": False,
            }

        return jsonify({"ok": True, "job_id": job_id, "gen_result": gen_result})

    except PlyEditError as e:
        return jsonify({"ok": False, "error": str(e)})
    except CCXRunError as e:
        return jsonify({"ok": False, "error": str(e)})
    except ValueError:
        return jsonify({"ok": False, "error": "Angle must be a number."})
    except OSError as e:
        return jsonify({"ok": False, "error": "File error: " + str(e)})


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or job.get("type") != "single":
        return jsonify({"ok": False, "error": "Unknown job id."})

    proc = job["proc"]
    returncode = proc.poll()
    log_tail = read_log_tail(job["log_path"])

    if returncode is None:
        return jsonify({"ok": True, "done": False, "log_tail": log_tail})

    if not job["log_file"].closed:
        job["log_file"].close()

    result = evaluate_result(job["log_path"], job["frd_path"], returncode)
    result["stopped"] = job["stopped"]
    return jsonify({"ok": True, "done": True, "result": result, "log_tail": log_tail})


@app.route("/stop/<job_id>", methods=["POST"])
def stop(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Unknown job id."})
    job["stopped"] = True
    if job.get("type") == "single":
        job["proc"].terminate()
    elif job.get("type") == "sweep":
        job["cancel_event"].set()
    return jsonify({"ok": True})


# ---------- angle sweep ----------

def run_sweep(job_id, input_path, output_folder, angles, ccx_exe):
    with JOBS_LOCK:
        job = JOBS[job_id]

    try:
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        for i, angle in enumerate(angles):
            if job["cancel_event"].is_set():
                break

            job["current_index"] = i
            job["current_angle"] = angle

            suffix = str(angle).replace("-", "m").replace(".", "p")
            file_name = base_name + "_" + suffix + "deg.inp"
            output_path = os.path.join(output_folder, file_name)

            try:
                gen_result = generate_inp(input_path, output_path, angle)
            except PlyEditError as e:
                job["results"].append(
                    {"angle": angle, "node_id": None, "rf": None, "error": str(e)}
                )
                continue

            rf_nset = gen_result["rf_nset"]
            run_result = run_ccx_blocking(output_path, ccx_exe, cancel_event=job["cancel_event"])
            job["log_tail"] = run_result["output_tail"]

            if run_result["stopped"]:
                job["results"].append(
                    {"angle": angle, "node_id": None, "rf": None, "error": "Stopped by user"}
                )
                break

            if run_result["has_error"] or not run_result["converged"]:
                job["results"].append(
                    {
                        "angle": angle,
                        "node_id": None,
                        "rf": None,
                        "error": "CalculiX did not converge cleanly",
                    }
                )
                continue

            dat_path = os.path.splitext(output_path)[0] + ".dat"
            try:
                rf = read_reaction_force(dat_path, rf_nset)
                for node_id, (fx, fy, fz) in sorted(rf["per_node"].items()):
                    magnitude = math.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
                    job["results"].append(
                        {"angle": angle, "node_id": node_id, "rf": magnitude, "error": None}
                    )
            except DatParseError as e:
                job["results"].append(
                    {
                        "angle": angle,
                        "node_id": None,
                        "rf": None,
                        "error": "Could not read reaction force: " + str(e),
                    }
                )

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

    if not input_path or not os.path.isfile(input_path):
        return jsonify({"ok": False, "error": "No file found at: " + input_path})
    if not output_folder:
        return jsonify({"ok": False, "error": "Please provide an output folder for the sweep."})

    try:
        angles = angle_range(angle_min, angle_max, angle_step)
    except PlyEditError as e:
        return jsonify({"ok": False, "error": str(e)})

    job_id = uuid.uuid4().hex
    job = {
        "type": "sweep",
        "results": [],
        "total": len(angles),
        "current_index": -1,
        "current_angle": None,
        "done": False,
        "cancel_event": threading.Event(),
        "log_tail": "",
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    thread = threading.Thread(
        target=run_sweep,
        args=(job_id, input_path, output_folder, angles, ccx_exe),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "job_id": job_id, "total": len(angles)})


@app.route("/sweep_status/<job_id>")
def sweep_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or job.get("type") != "sweep":
        return jsonify({"ok": False, "error": "Unknown sweep job id."})
    return jsonify(
        {
            "ok": True,
            "done": job["done"],
            "total": job["total"],
            "current_index": job["current_index"],
            "current_angle": job["current_angle"],
            "results": job["results"],
            "log_tail": job["log_tail"],
            "fatal_error": job["error"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050, threaded=True)