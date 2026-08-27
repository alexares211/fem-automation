# Run CalculiX on a generated .inp and report progress.
#
# ccx always writes its output files (.frd, .dat, .log, .sta, .cvg, ...) into
# the same folder as the .inp it is given -- so pointing this at a file in
# your chosen output folder keeps everything together there.

import os
import time
import subprocess


class CCXRunError(Exception):
    pass


def start_ccx(inp_path, ccx_exe):
    if not os.path.isfile(ccx_exe):
        raise CCXRunError("CalculiX executable not found at: " + ccx_exe)
    if not os.path.isfile(inp_path):
        raise CCXRunError("Input file not found at: " + inp_path)

    folder = os.path.dirname(inp_path)
    jobname = os.path.splitext(os.path.basename(inp_path))[0]
    log_path = os.path.join(folder, jobname + "_run.log")
    frd_path = os.path.join(folder, jobname + ".frd")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [ccx_exe, jobname],
        cwd=folder,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    return {
        "proc": proc,
        "log_file": log_file,
        "log_path": log_path,
        "frd_path": frd_path,
        "jobname": jobname,
    }


def run_ccx_blocking(inp_path, ccx_exe, cancel_event=None, poll_interval=0.3):
    """Run ccx and wait for it to finish, checking cancel_event periodically.

    Used by the angle sweep, which runs many solves back to back in a
    background thread -- cancel_event lets the user's Stop button kill the
    currently running solve (and skip the rest of the sweep) without having
    to wait for it to finish on its own.
    """
    state = start_ccx(inp_path, ccx_exe)
    proc = state["proc"]

    stopped = False
    while True:
        returncode = proc.poll()
        if returncode is not None:
            break
        if cancel_event is not None and cancel_event.is_set():
            proc.terminate()
            proc.wait()
            stopped = True
            break
        time.sleep(poll_interval)

    if not state["log_file"].closed:
        state["log_file"].close()

    result = evaluate_result(state["log_path"], state["frd_path"], proc.returncode)
    result["stopped"] = stopped
    result["log_path"] = state["log_path"]
    result["jobname"] = state["jobname"]
    return result


def read_log_tail(log_path, n=40):
    if not os.path.isfile(log_path):
        return ""
    with open(log_path, "r", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def evaluate_result(log_path, frd_path, returncode):
    text = ""
    if os.path.isfile(log_path):
        with open(log_path, "r", errors="ignore") as f:
            text = f.read()
    return {
        "returncode": returncode,
        "has_error": "*ERROR" in text,
        "converged": "Job finished" in text,
        "frd_path": frd_path,
        "frd_exists": os.path.isfile(frd_path),
        "output_tail": "\n".join(text.strip().splitlines()[-30:]),
    }