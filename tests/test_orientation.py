import math
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from dat_parser import read_internal_energy, read_displacements, DatParseError
from orientation_optimization import SearchSettings, optimize_orientations, canonical
from structural_response import analyse_loading, add_response_output, stress_metrics, DesignError, ALL_ELEMENTS
from inp_import import parse_cards, shell_reference_axes
from inp_writer import InpWriteError
from ply_orientation import orientation_vectors
from tools.beam_mesh import mesh_text


STRENGTHS = dict(S11t=1000, S11c=800, S22t=40, S22c=120, S21s=60)


def response(angles, objective, utilizations):
    return dict(angles=list(angles), objective_value=objective, compliance=objective,
                strain_energy=objective/2, max_displacement=objective,
                mode="force", max_utilization=max(utilizations),
                utilization_cv=float(np.std(utilizations)/np.mean(utilizations)),
                plies=[dict(ply=i+1, utilization=v) for i, v in enumerate(utilizations)])


class SearchTests(unittest.TestCase):
    def test_strength_constraint_excludes_stiffest_but_overstressed_layup(self):
        # An independent one-variable constrained problem: optimum is the
        # first grid angle above acos(sqrt(1/3)) = 54.7356 degrees.
        def evaluate(angles):
            c2 = math.cos(math.radians(angles[0]))**2
            return response(angles, 4-3*c2, [0.5+1.5*c2])
        result = optimize_orientations(evaluate, [45], settings=SearchSettings(angle_step=5, max_evaluations=100))
        self.assertTrue(result["constraints_met"])
        self.assertAlmostEqual(abs(result["best"]["angles"][0]), 55)
        self.assertTrue(result["fully_stressed_achieved"])
        self.assertLessEqual(result["best"]["max_utilization"], 1)

    def test_equal_stress_adds_an_independent_balance_constraint(self):
        def evaluate(angles):
            s = [math.sin(math.radians(a))**2 for a in angles]
            return response(angles, 1+0.5*sum(s), [0.2+0.6*s[0], 0.8-0.6*s[1]])
        fsd = optimize_orientations(evaluate, [45, 45], settings=SearchSettings(angle_step=5))
        equal = optimize_orientations(evaluate, [45, 45], settings=SearchSettings(method="equal_stress", angle_step=5, equal_tolerance=0.05))
        self.assertAlmostEqual(fsd["best"]["objective_value"], 1)
        self.assertGreater(fsd["best"]["utilization_cv"], 0.05)
        self.assertTrue(equal["constraints_met"])
        self.assertLessEqual(equal["best"]["utilization_cv"], 0.05+1e-8)
        self.assertGreater(equal["best"]["objective_value"], fsd["best"]["objective_value"])

    def test_symmetry_periodicity_and_count_are_preserved(self):
        seen = []
        def evaluate(angles):
            seen.append(list(angles))
            return response(angles, 1+sum(math.sin(math.radians(a))**2 for a in angles), [0.2]*len(angles))
        result = optimize_orientations(evaluate, [30, -30], symmetric=True, settings=SearchSettings(angle_step=5))
        self.assertTrue(all(len(a) == 4 and a == a[::-1] for a in seen))
        self.assertEqual(len({tuple(canonical(a)) for a in seen}), len(seen))
        self.assertEqual(result["ply_count"], 4)
        self.assertAlmostEqual(result["best"]["objective_value"], 1)
        self.assertFalse(result["fully_stressed_achieved"])

    def test_stop_keeps_completed_results(self):
        stop = threading.Event()
        def evaluate(angles):
            return response(angles, 1, [0.2]*len(angles))
        result = optimize_orientations(evaluate, [30], cancel_event=stop,
                                      on_evaluation=lambda entry: stop.set())
        self.assertTrue(result["stopped"])
        self.assertEqual(result["evaluations"], 1)
        self.assertIsNotNone(result["best"])

    def test_infeasibility_is_not_reported_as_success(self):
        result = optimize_orientations(lambda a: response(a, 1, [2]*len(a)), [0],
                                      settings=SearchSettings(max_evaluations=5))
        self.assertFalse(result["constraints_met"])
        self.assertEqual(result["evaluations"], 5)


class ResponseTests(unittest.TestCase):
    def deck(self, boundary="FIXED_EDGE,1,6,0", load="*CLOAD\nTIP_EDGE,2,-1"):
        return mesh_text(2, 1) + f"*STEP\n*STATIC\n*BOUNDARY\n{boundary}\n{load}\n*END STEP\n"

    def test_force_and_displacement_have_opposite_energy_objectives(self):
        self.assertEqual(analyse_loading(self.deck())["mode"], "force")
        self.assertEqual(analyse_loading(self.deck(boundary="TIP_EDGE,2,2,0.1", load=""))["mode"], "displacement")
        with self.assertRaises(DesignError):
            analyse_loading(self.deck(boundary="TIP_EDGE,2,2,0.1"))

    def test_rejects_nonlinear_and_multiple_steps(self):
        for text in (self.deck().replace("*STEP", "*STEP, NLGEOM"), self.deck()*2,
                     self.deck().replace("*STATIC", "*DYNAMIC"), "*INCLUDE,INPUT=hidden.inp\n"+self.deck()):
            with self.assertRaises(DesignError):
                analyse_loading(text)

    def test_output_requests_preserve_conditions(self):
        deck = self.deck()
        updated = add_response_output(deck, "BEAM")
        self.assertIn("*BOUNDARY\nFIXED_EDGE,1,6,0\n*CLOAD\nTIP_EDGE,2,-1", updated)
        self.assertIn("TOTALS=ONLY\nELSE", updated)
        self.assertNotIn("\nENER\n", updated)

    def test_off_axis_fibers_stay_in_the_actual_xy_shell_plane(self):
        axes = shell_reference_axes(parse_cards(self.deck()), "BEAM")
        a, b = orientation_vectors(45, reference_axes=axes)
        np.testing.assert_allclose(a, [2**-0.5, 2**-0.5, 0], atol=1e-12)
        np.testing.assert_allclose(b, [-2**-0.5, 2**-0.5, 0], atol=1e-12)
        self.assertAlmostEqual(float(np.dot(a, b)), 0)

    def test_rotated_planar_mesh_defines_its_own_tangent_frame(self):
        cards = parse_cards(self.deck())
        for card in cards:
            if card["keyword"] == "NODE":
                changed = []
                for line in card["body"]:
                    if line.startswith("**") or not line.strip():
                        changed.append(line)
                        continue
                    node, x, y, z = line.split(",")
                    y = float(y)
                    changed.append(f"{node},{x},{y/2**0.5},{y/2**0.5}")
                card["body"] = changed
        axes = shell_reference_axes(cards, "BEAM")
        a, _ = orientation_vectors(45, reference_axes=axes)
        np.testing.assert_allclose(a, [2**-0.5, 0.5, 0.5], atol=1e-12)

    def test_nonplanar_laminate_is_rejected_before_wrong_projection(self):
        cards = parse_cards(self.deck())
        node_card = next(c for c in cards if c["keyword"] == "NODE")
        node_card["body"][0] = "1, 0, -5, 1"
        with self.assertRaises(InpWriteError):
            shell_reference_axes(cards, "BEAM")

    def test_stress_sign_allowables_and_all_plies_are_checked(self):
        rows = [dict(layer=1, S11=-400, S22=0, S12=0), dict(layer=2, S11=0, S22=20, S12=12)]
        result = stress_metrics(rows, 2, STRENGTHS)
        self.assertEqual(result["max_utilization"], 0.5)
        self.assertEqual(result["utilization_cv"], 0)
        with self.assertRaises(DesignError):
            stress_metrics(rows[:1], 2, STRENGTHS)

    def test_other_parts_with_wrapped_connectivity_are_counted_once(self):
        deck = self.deck().replace("*STEP", "*ELEMENT,TYPE=C3D20R,ELSET=OTHER\n901,"+",".join(map(str, range(1,17)))+"\n17,18,19,20\n*STEP")
        updated = add_response_output(deck, "BEAM")
        all_elements = next(c for c in parse_cards(updated)
                            if c["keyword"] == "ELSET" and ALL_ELEMENTS in c["header"])
        ids = [int(x.strip()) for line in all_elements["body"] for x in line.split(",")]
        self.assertEqual(ids, [1, 2, 901])

    def test_dat_parser_uses_integrated_energy_and_final_increment(self):
        text = """total internal energy for set FEMOPT_ALL_ELEMENTS and time 0.5000000E+00

 1.000000E-03

total internal energy for set FEMOPT_ALL_ELEMENTS and time 1.0000000E+00

 4.000000E-03

displacements (vx,vy,vz) for set FEMOPT_ALL_NODES and time 1.0000000E+00

 1 0.000000E+00-1.000000E-02 0.000000E+00

"""
        scratch = Path(__file__).resolve().parents[1]/"runs"/"test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path = Path(folder)/"example.dat"
            path.write_text(text)
            self.assertEqual(read_internal_energy(path, "femopt_all_elements")["energy"], 0.004)
            self.assertEqual(read_displacements(path, "femopt_all_nodes")["maximum"], 0.01)
            with self.assertRaises(DatParseError):
                read_internal_energy(path, "OTHER_SET")


if __name__ == "__main__":
    unittest.main()
