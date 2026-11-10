"""Independent cantilever checks against the actual PrePoMax-exported input.

Run from the repository: python tools/beam_benchmark.py
All solver outputs and the validation JSON are kept in the chosen data folder.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dat_parser import read_displacements, read_reaction_force
from structural_response import CalculixDesignEvaluator, ALL_NODES


CONSTANTS = dict(E1=135100, E2=9600, E3=9600, nu12=0.32, nu13=0.32,
                 nu23=0.487, G12=5300, G13=5300, G23=3228.6)
STRENGTHS = dict(S11t=1996, S11c=1398, S22t=64, S22c=268, S21s=92)
DATA_ROOT = Path(r"D:\DataBin_Composite_Automation")


def laminate_moduli(angles, ply_thickness=0.25):
    """Independent classical-laminate extensional stiffness for the beam check."""
    e1, e2, nu, shear = (CONSTANTS[k] for k in ("E1", "E2", "nu12", "G12"))
    denominator = 1-nu*nu*e2/e1
    q11, q22, q12, q66 = e1/denominator, e2/denominator, nu*e2/denominator, shear
    matrix = np.zeros((3, 3))
    for angle in angles:
        m, n = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        a = q11*m**4 + 2*(q12+2*q66)*m*m*n*n + q22*n**4
        b = q11*n**4 + 2*(q12+2*q66)*m*m*n*n + q22*m**4
        c = (q11+q22-4*q66)*m*m*n*n + q12*(m**4+n**4)
        d = (q11-q12-2*q66)*m**3*n - (q22-q12-2*q66)*m*n**3
        e = (q11-q12-2*q66)*m*n**3 - (q22-q12-2*q66)*m**3*n
        f = (q11+q22-2*q12-2*q66)*m*m*n*n + q66*(m**4+n**4)
        matrix += ply_thickness*np.array([[a,c,d],[c,b,e],[d,e,f]])
    compliance = np.linalg.inv(matrix)
    thickness = ply_thickness*len(angles)
    return float(1/(thickness*compliance[0,0])), float(1/(thickness*compliance[2,2]))


def beam_payload(method="fully_stressed", budget=160, auto_open=False):
    return dict(import_inp_path=str(DATA_ROOT/"cantilever_beam"/"cantilever_prepo.inp"),
                import_elset="BEAM", import_rf_nset="FIXED_EDGE", material_name="AS4_8552",
                constants=CONSTANTS, strengths=STRENGTHS, thickness=0.25,
                layup="45/-45/90/0", symmetric=True, method=method,
                max_evaluations=budget, angle_step=1, stress_limit=1,
                equal_tolerance=0.10, output_folder=str(DATA_ROOT),
                ccx_exe=r"D:\PrePoMax v2.5.0\Solver\ccx_dynamic.exe", auto_open=auto_open)


def validate(output, inp):
    text = Path(inp).read_text(encoding="utf-8-sig")
    model = dict(import_inp_text=text, import_elset="BEAM", import_rf_nset="FIXED_EDGE",
                 material_name="AS4_8552", constants=CONSTANTS, thickness=0.25)
    evaluator = CalculixDesignEvaluator(model, beam_payload()["ccx_exe"], output/"solves", STRENGTHS)
    results = {}
    initial_angles = [45,-45,90,0,0,90,-45,45]
    for name, angles in (("initial", initial_angles), ("axial", [0]*8), ("transverse", [90]*8), ("off_axis_45", [45]*8)):
        result = evaluator(angles)
        print(name, {k: result[k] for k in ("compliance", "max_displacement", "max_utilization", "utilization_cv")}, flush=True)
        results[name] = result
    axial = results["axial"]
    # The following topology/load data is checked against the actual export,
    # rather than assuming what the PrePoMax concentrated-force dialog meant.
    from inp_import import parse_cards
    cards = parse_cards(text)
    tip = next(c for c in cards if c["keyword"] == "NSET" and "NSET=TIP_EDGE" in c["header"].upper().replace(" ", ""))
    tip_ids = [int(x.strip()) for line in tip["body"] if not line.lstrip().startswith("**") for x in line.split(",") if x.strip()]
    if not any("TIP_EDGE, 2, -1" in line for c in cards if c["keyword"] == "CLOAD" for line in c["body"]):
        raise AssertionError("This benchmark expects PrePoMax's 1 N per TIP_EDGE node load.")
    u = read_displacements(axial["dat_path"], ALL_NODES)["per_node"]
    force = float(len(tip_ids))
    tip_work = sum(-u[n][1] for n in tip_ids)
    equilibrium = read_reaction_force(axial["dat_path"], "FIXED_EDGE")
    mean_tip = tip_work/force
    width, thickness, length = 10.0, 2.0, 100.0
    area, inertia = width*thickness, thickness*width**3/12
    euler = force*length**3/(3*CONSTANTS["E1"]*inertia)
    timoshenko = euler + force*length/((5/6)*CONSTANTS["G12"]*area)
    initial_e, initial_g = laminate_moduli(initial_angles)
    initial_theory = force*length**3/(3*initial_e*inertia) + force*length/((5/6)*initial_g*area)
    initial_tip = results["initial"]["compliance"]/force
    checks = dict(
        energy_equals_force_dot_displacement=abs(axial["compliance"]/tip_work-1) < 0.005,
        support_equilibrium=abs(equilibrium["sum_fy"]/force-1) < 1e-5,
        beam_reference_within_5_percent=abs(mean_tip/timoshenko-1) < 0.05,
        axial_stiffer_than_initial=results["axial"]["compliance"] < results["initial"]["compliance"],
        initial_stiffer_than_transverse=results["initial"]["compliance"] < results["transverse"]["compliance"],
        off_axis_45_differs_physically_from_axial=results["off_axis_45"]["compliance"] > 5*results["axial"]["compliance"],
        initial_laminate_matches_classical_laminate_beam=abs(initial_tip/initial_theory-1) < 0.05,
        eight_ply_stress_output=all(len(r["plies"]) == 8 and all(p["points"] for p in r["plies"]) for r in results.values()),
    )
    report = dict(checks=checks, total_force_N=force, tip_displacement_mm=mean_tip,
                  euler_bernoulli_mm=euler, timoshenko_mm=timoshenko,
                  beam_reference_error_percent=100*(mean_tip/timoshenko-1),
                  energy_vs_external_work_error_percent=100*(axial["compliance"]/tip_work-1),
                  initial_laminate_reference_error_percent=100*(initial_tip/initial_theory-1),
                  initial_laminate_E_MPa=initial_e, initial_laminate_G_MPa=initial_g,
                  results=results)
    output.mkdir(parents=True, exist_ok=True)
    (output/"validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "results"}, indent=2), flush=True)
    if not all(checks.values()):
        raise AssertionError("A cantilever validation check failed; inspect validation.json.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inp", default=str(DATA_ROOT/"cantilever_beam"/"cantilever_prepo.inp"))
    parser.add_argument("--output", default=str(DATA_ROOT/("beam_validation_"+datetime.now().strftime("%Y%m%d_%H%M%S"))))
    args = parser.parse_args()
    validate(Path(args.output), args.inp)
