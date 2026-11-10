"""Open and verify this run's FRD in the active Windows PrePoMax desktop.

PrePoMax automatically tries to parse a same-name DAT file. Its history
reader cannot consume all composite/local-axis stress rows used here, so
the viewer receives a byte-for-byte FRD copy in a separate subdirectory.
The original solver DAT and all integration-point data remain in the run.
Only the recovery prompt of the newly created viewer process is declined;
no recovery files or existing model windows are modified by this helper.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import threading
import time

_lock = threading.Lock()
_viewers = {}


def open_results(executable, frd_path, timeout=45):
    if os.name != "nt":
        raise RuntimeError("Automatic PrePoMax viewing requires an interactive Windows desktop.")
    try:
        from pywinauto import Desktop
    except ImportError as exc:
        raise RuntimeError("Install requirements-desktop.txt to open and verify PrePoMax results automatically.") from exc
    executable, source = Path(executable).resolve(), Path(frd_path).resolve()
    if not executable.is_file() or not source.is_file():
        raise RuntimeError("The PrePoMax executable or optimized FRD file is missing.")
    with _lock:
        desktop = Desktop(backend="win32")
        viewer_folder = source.parent/"prepomax_viewer"
        viewer_folder.mkdir(exist_ok=True)
        target = viewer_folder/source.name
        shutil.copy2(source, target)
        key = str(target).casefold()
        proc = _viewers.get(key)
        if proc is None or proc.poll() is not None:
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 1  # SW_SHOWNORMAL, an explicitly requested interactive viewer
            proc = subprocess.Popen([str(executable), "-f", str(target), "-u", "MM_TON_S_C"],
                                    cwd=str(viewer_folder), startupinfo=startup,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _viewers[key] = proc
        skipped_recovery = False
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"PrePoMax exited before loading the result (code {proc.returncode}).")
            for window in desktop.windows(process=proc.pid, visible_only=False):
                title = window.window_text()
                if title in ("Warning", "Error") and window.is_visible():
                    message = " ".join(child.window_text() for child in window.children() if child.class_name() == "Static")
                    if title == "Warning" and "recovery file from a previous PrePoMax session" in message:
                        window.set_focus()
                        next(child for child in window.children()
                             if child.window_text() == "Cancel" and child.class_name() == "Button").click_input()
                        skipped_recovery = True
                        continue
                    raise RuntimeError("PrePoMax needs attention: " + message)
                if not title.startswith("PrePoMax"):
                    continue
                controls = [child for child in window.descendants()
                            if "COMBOBOX" in child.class_name().upper()]
                combos = [child.window_text() for child in controls]
                # The result selector, unlike the main window title, contains
                # the actual loaded FRD path. A launched PID alone is not success.
                if not any(str(target).casefold() == value.casefold() for value in combos):
                    continue
                if "Displacements" not in combos:
                    for control in controls:
                        if "Displacements" in control.texts():
                            control.select("Displacements")
                            break
                window.restore()
                window.maximize()
                window.set_focus()
                info = dict(pid=proc.pid, result_path=str(target), display_verified=True,
                            recovery_prompt_declined=skipped_recovery,
                            variable="Displacements")
                (viewer_folder/"viewer.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
                return info
            time.sleep(0.25)
        raise RuntimeError("PrePoMax did not expose the loaded FRD result within 45 seconds. Open the FRD manually; solver results are saved.")
