import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify

from ply_orientation import generate_inp, PlyEditError
from ccx_runner import start_ccx, read_log_tail, evaluate_result, CCXRunError

DEFAULT_CCX_EXE = r"D:\PrePoMax v2.5.0\Solver\ccx_dynamic.exe"

app = Flask(__name__)

JOBS = {}
JOBS_LOCK = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html", default_ccx_exe=DEFAULT_CCX_EXE)


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
    if job is None:
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
    job["proc"].terminate()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5050, threaded=True)