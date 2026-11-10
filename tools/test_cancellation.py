"""Verify native solver cancellation keeps completed designs and their files."""
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.beam_benchmark import beam_payload, DATA_ROOT


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request("http://127.0.0.1:5050"+path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def verify():
    payload = beam_payload(auto_open=False)
    payload["output_folder"] = str(DATA_ROOT/"cancellation_validation")
    started = request("/orientation/start", payload)
    assert started["ok"], started
    job = started["job_id"]
    deadline = time.monotonic()+30
    while True:
        status = request("/orientation/status/"+job)
        if status["history"]:
            break
        assert time.monotonic() < deadline, status
        time.sleep(0.05)
    assert request("/orientation/stop/"+job, {})["ok"]
    while True:
        status = request("/orientation/status/"+job)
        if status["done"]:
            break
        assert time.monotonic() < deadline, status
        time.sleep(0.05)
    assert not status["error"], status["error"]
    result = status["result"]
    assert result["stopped"] and result["evaluations"] >= 1 and result["best"]
    folder = Path(result["output_folder"])
    assert (folder/"optimized.frd").is_file() and (folder/"report.json").is_file()
    assert not status["viewer"], "A cancelled run must not automatically open a result viewer."
    print(json.dumps(dict(passed=True, completed_evaluations=result["evaluations"],
                          best_compliance=result["best"]["compliance"], report=str(folder/"report.json")), indent=2))


if __name__ == "__main__":
    verify()
