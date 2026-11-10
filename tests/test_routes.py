from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import app
from optimization_routes import _jobs
from tools.beam_mesh import mesh_text
from tools.beam_benchmark import beam_payload
from inp_import import parse_cards


class OrientationApiTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/"runs"/"test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.inp = Path(self.temp.name)/"source.inp"
        self.inp.write_text(mesh_text(2, 1)+"*STEP\n*STATIC\n*BOUNDARY\nFIXED_EDGE,1,6,0\n*CLOAD\nTIP_EDGE,2,-1\n*END STEP\n")
        self.payload = beam_payload()
        self.payload.update(import_inp_path=str(self.inp), output_folder=self.temp.name,
                            ccx_exe=str(Path(__file__).resolve()))
        self.client = app.test_client()
        self.jobs_patch = patch.dict(_jobs, {}, clear=True)
        self.jobs_patch.start()
        self.addCleanup(self.jobs_patch.stop)

    def test_fixed_size_snapshot_is_passed_to_worker(self):
        with patch("optimization_routes.threading.Thread") as thread:
            response = self.client.post("/orientation/start", json=self.payload)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json["ok"])
            worker_args = thread.call_args.kwargs["args"]
            model, initial, symmetric = worker_args[1:4]
            self.assertEqual(model["thickness"], 0.25)
            self.assertEqual(len(initial), 4)
            self.assertTrue(symmetric)
            self.assertEqual(model["import_inp_text"], self.inp.read_text())
            self.payload["thickness"] = 9
            self.assertEqual(model["thickness"], 0.25)
            status = self.client.get("/orientation/status/"+response.json["job_id"]).json
            self.assertEqual(status["phase"], "solving")
            stopped = self.client.post("/orientation/stop/"+response.json["job_id"])
            self.assertTrue(stopped.json["ok"])
            self.assertTrue(_jobs[response.json["job_id"]]["cancel_event"].is_set())

    def test_numeric_and_strength_validation_returns_actionable_errors(self):
        mutations = [dict(max_evaluations=8.5), dict(angle_step=7), dict(thickness=float("nan")),
                     dict(strengths={}), dict(symmetric="false"), dict(method="unknown")]
        for mutation in mutations:
            with self.subTest(mutation=mutation), patch("optimization_routes.threading.Thread") as thread:
                response = self.client.post("/orientation/start", json={**self.payload, **mutation})
                self.assertEqual(response.status_code, 400, response.json)
                self.assertFalse(response.json["ok"])
                thread.assert_not_called()

    def test_invalid_elastic_constants_do_not_launch_a_solver(self):
        data = deepcopy(self.payload)
        data["constants"]["nu23"] = 2
        response = self.client.post("/orientation/start", json=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("positive-definite", response.json["error"])

    def test_missing_job_and_unlisted_artifact_are_not_served(self):
        self.assertEqual(self.client.get("/orientation/status/missing").status_code, 404)
        self.assertEqual(self.client.post("/orientation/open/missing").status_code, 404)
        self.assertEqual(self.client.get("/orientation/artifact/missing/source.py").status_code, 404)

    def test_layup_and_complete_deck_previews_use_the_same_physical_45_degree_axis(self):
        payload = {**self.payload, "layup": "45", "symmetric": False, "material": self.payload["material_name"]}
        section = self.client.post("/section_block", json=payload).json
        deck = self.client.post("/inp_preview", json=payload).json
        self.assertTrue(section["ok"], section)
        self.assertTrue(deck["ok"], deck)
        section_axis = next(c for c in parse_cards(section["block"]) if c["keyword"] == "ORIENTATION")["body"][0]
        deck_axis = next(c for c in parse_cards(deck["inp"]) if c["keyword"] == "ORIENTATION")["body"][0]
        self.assertEqual(section_axis, deck_axis)
        self.assertEqual([float(v) for v in deck_axis.split(",")[:3]], [0.707107, 0.707107, 0.0])


if __name__ == "__main__":
    unittest.main()
