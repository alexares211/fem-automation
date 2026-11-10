r"""Run both cantilever optimizations through the actual Chrome interface.

Requires the local Flask app and requirements-desktop.txt. All screenshots
and numerical results are written under D:\DataBin_Composite_Automation.
Pass --keep-open to leave the tested Chrome page available for review.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inp_import import parse_cards
from playwright.sync_api import sync_playwright

ROOT = Path(r"D:\DataBin_Composite_Automation")


def conditions(text):
    protected = {"NODE", "ELEMENT", "BOUNDARY", "CLOAD", "DLOAD", "STEP", "STATIC"}
    return [c for c in parse_cards(text) if c["keyword"] in protected]


def run_workflow(output, keep_open=False):
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=False,
                                              args=["--start-maximized", "--remote-debugging-port=9222"])
        context = browser.new_context(no_viewport=True, accept_downloads=True)
        page = context.new_page()
        errors, summaries = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:5050/", wait_until="networkidle")
        page.locator("#loadBeamBtn").click()
        page.wait_for_function("window.femImportState && window.femImportState.path")
        page.locator("#inpElsetPick").select_option("BEAM")
        assert page.locator("#fixedPlyCount").input_value() == "8"
        assert page.locator("#fixedTotalThickness").input_value() == "2"
        assert page.locator("#run_output_folder").input_value() == str(ROOT)
        for method in ("fully_stressed", "equal_stress"):
            page.locator("#orientationMethod").select_option(method)
            assert page.locator("#equalStressSettings").is_visible() == (method == "equal_stress")
            page.locator("#optBudget").fill("160")
            page.locator("#optAutoOpen").check()
            page.locator("#optAngleStep").fill("1")
            with page.expect_response(lambda r: r.url.endswith("/orientation/start")) as started:
                page.locator("#orientationRun").click()
            start = started.value.json()
            assert start["ok"], start
            job = start["job_id"]
            print(method, "started", job, start["output_folder"], flush=True)
            deadline, reported = time.monotonic()+600, -1
            while True:
                status = context.request.get(f"http://127.0.0.1:5050/orientation/status/{job}").json()
                count = len(status["history"])
                if count//10 != reported:
                    print(method, count, "completed real solves", flush=True)
                    reported = count//10
                if status["done"]:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("The browser workflow exceeded ten minutes.")
                page.wait_for_timeout(400)
            assert not status["error"], status["error"]
            result = status["result"]
            assert result["constraints_met"], result
            assert result["ply_count"] == 8 and result["symmetry_enforced"]
            assert result["best"]["angles"] == result["best"]["angles"][::-1]
            assert result["best"]["compliance"] < result["baseline"]["compliance"]
            assert result["best"]["compliance"] <= 1.291, "Search must at least recover the checked all-axial design."
            assert result["best"]["max_utilization"] <= 1
            if method == "equal_stress":
                assert result["best"]["utilization_cv"] <= 0.10+1e-8
            folder = Path(result["output_folder"])
            original = (folder/"source_prepo.inp").read_text()
            optimized = (folder/"optimized.inp").read_text()
            assert conditions(original) == conditions(optimized), "Geometry, supports, loads, and step cards changed."
            section = next(c for c in parse_cards(optimized) if c["keyword"] == "SHELL SECTION")
            layers = [line for line in section["body"] if line.strip() and not line.lstrip().startswith("**")]
            assert len(layers) == 8 and all(float(line.split(",")[0]) == 0.25 for line in layers)
            page.locator("#orientationFiles a", has_text="report.json").wait_for(timeout=10000)
            assert "feasible design was found" in page.locator("#orientationSummary").inner_text()
            assert not page.locator("#orientationRun").is_disabled()
            assert page.locator("#orientationStop").is_disabled()
            assert page.locator("#orientationPlyResults tbody tr").count() == 8
            with page.expect_download() as download:
                page.locator("#orientationFiles a", has_text="report.json").click()
            download.value.save_as(str(output/f"{method}_downloaded_report.json"))
            page.locator("#optSection").screenshot(path=str(output/f"{method}_web_results.png"))
            summary = dict(method=method, job_id=job, output_folder=str(folder),
                           evaluations=result["evaluations"], termination=result["termination"],
                           baseline_compliance=result["baseline"]["compliance"],
                           optimized_compliance=result["best"]["compliance"],
                           stiffness_gain_percent=result["stiffness_gain_percent"],
                           angles=result["best"]["angles"], peak_utilization=result["best"]["max_utilization"],
                           ply_utilization_cv=result["best"]["utilization_cv"],
                           max_displacement_mm=result["best"]["max_displacement"],
                           viewer=status["viewer"], viewer_error=status["viewer_error"])
            summaries.append(summary)
            print(json.dumps(summary, indent=2), flush=True)
            assert status["viewer"] and not status["viewer_error"], "Automatic PrePoMax launch failed."
            assert status["viewer_details"]["display_verified"], "The actual FRD result selector was not verified."
        assert not errors, errors
        (output/"browser_validation.json").write_text(json.dumps(dict(runs=summaries, browser_errors=errors), indent=2), encoding="utf-8")
        page.locator("#optSection").scroll_into_view_if_needed()
        page.bring_to_front()
        print("BOTH BROWSER WORKFLOWS PASSED", str(output), flush=True)
        if keep_open:
            print("Chrome is being kept open for review.", flush=True)
            while browser.is_connected():
                page.wait_for_timeout(1000)
        else:
            browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--output", default=str(ROOT/("browser_validation_"+datetime.now().strftime("%Y%m%d_%H%M%S"))))
    args = parser.parse_args()
    run_workflow(Path(args.output), args.keep_open)
