
# Run CalculiX on a generated .inp in the background and report progress.
#
# ccx always writes its output files (.frd, .dat, .log, .sta, .cvg, ...) into
# the same folder as the .inp it is given -- so pointing this at a file in
# your chosen output folder keeps everything together there.
#
# start_ccx launches the process without blocking (Popen), redirecting its
# output to a log file, so a web page can poll progress and offer a working
# Stop button (terminates the real OS process, not just the browser request).

import os
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