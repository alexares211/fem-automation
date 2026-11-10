"""Capture this project's PrePoMax result windows on the active desktop."""
import argparse
from pathlib import Path
import time

from pywinauto import Desktop


def capture(output):
    output.mkdir(parents=True, exist_ok=True)
    desktop = Desktop(backend="win32")
    windows = []
    for window in desktop.windows():
        if not window.window_text().startswith("PrePoMax"):
            continue
        combo_values = [child.window_text() for child in window.descendants()
                        if "COMBOBOX" in child.class_name().upper()]
        result_path = next((v for v in combo_values if "DataBin_Composite_Automation" in v and v.endswith(".frd")), "")
        if result_path or "DataBin_Composite_Automation" in window.window_text():
            windows.append((window, result_path or window.window_text()))
    reports = []
    for window, result_path in sorted(windows, key=lambda item: ("equal_stress" in item[1], item[1])):
        title = window.window_text()
        print("PrePoMax:", title, flush=True)
        window.restore()
        window.maximize()
        window.set_focus()
        time.sleep(0.5)
        method = "equal_stress" if "equal_stress" in result_path else ("fully_stressed" if "fully_stressed" in result_path else "beam_model")
        path = output/(method+"_prepomax.png")
        window.capture_as_image().save(str(path))
        trees = [child.texts() for child in window.descendants() if "SysTreeView32" in child.class_name() and child.is_visible()]
        combos = [(child.handle, child.window_text()) for child in window.descendants()
                  if "COMBOBOX" in child.class_name().upper() and child.is_visible()]
        print("Screenshot:", path, "Tree:", trees, "Fields:", combos, flush=True)
        reports.append(dict(title=title, result_path=result_path, screenshot=str(path), tree=trees, fields=combos))
    if len(windows) < 2:
        raise RuntimeError("Expected the project model and at least one PrePoMax results window.")
    import json
    (output/"prepomax_windows.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=r"D:\DataBin_Composite_Automation\prepomax_verification")
    args = parser.parse_args()
    capture(Path(args.output))
