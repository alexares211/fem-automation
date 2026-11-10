# Composite orientation optimization

This local Flask application optimizes ply orientations using the mesh, supports, and loads exported by PrePoMax. The current stage holds the number of plies, individual ply thickness, and laminate symmetry fixed. Both stress methods evaluate candidates with CalculiX and display the best feasible result in PrePoMax.

Illustrated PDF guides: [User instructions](docs/FEM_Automation_User_Instructions.pdf) and [Methods explained with a worked example](docs/Composite_Optimization_Methods_Explained.pdf).

## Start the application

Python 3.12 and Windows were used for the complete desktop workflow. Install the desktop requirements when PrePoMax should open automatically.

```powershell
py -3.12 -m venv runs\.venv
.\runs\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\runs\.venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 5050
```

Open [FEM Automation](http://127.0.0.1:5050/). The default solver is `D:\PrePoMax v2.5.0\Solver\ccx_dynamic.exe`. The default output directory is `D:\DataBin_Composite_Automation`; each optimization creates its own timestamped subdirectory. `FEM_OUTPUT_ROOT` and `FEM_PREPOMAX` can override the data directory and viewer executable. `FEM_OMP_THREADS` controls solver threads and defaults to 1.

## Run an optimization

1. Export a CalculiX `.inp` from PrePoMax with the mesh, one linear static step, supports, and loads. A material or section can be supplied in the export; the application replaces the selected part's section with the composite definition.
2. Upload that input or choose **Load cantilever beam example**. Select the element set receiving the laminate.
3. Set the engineering constants and five in-plane strength allowables. Enter the starting ply angles and ply thickness in **Layup**; enable symmetry if required.
4. Choose the stress method, angular resolution, strength limit, and FEM evaluation budget. Equal stress also has a tolerance on variation between ply utilizations.
5. Run **Optimize orientations**. The page reports compliance, displacement, stress utilization, and the best feasible result. **Stop and keep best** retains completed evaluations. **Use optimized layup** restores the optimized angles together with their fixed sizing and symmetry.

The example uses mm, N, and MPa. Keep imported units consistent with material properties and thickness. Material directions follow the selected planar shell mesh: 0° is projected global X, or global Y if X is normal to the part. The mesh normal defines positive rotation. Rotated planar parts are supported; curved parts and inconsistent shell normals are rejected because they need a local material-direction field.

## Methods and objective

The [research and method definitions](docs/optimization_methods.md) give the equations and scope. In this stage, **Fully stressed design — orientation only** means minimum compliance subject to a maximum ply utilization constraint. **Equal stress** adds a constraint on the coefficient of variation of each ply's peak normalized utilization. These are explicit fixed-sizing adaptations: classical fully stressed sizing changes material sizing, and equal utilization can trade stiffness for a more balanced stress distribution.

For force control, the objective is to minimize `C = 2U = Fᵀu`. For imposed displacement, the application maximizes `2U`, which increases stiffness at the prescribed motion. Mixed force and nonzero displacement loading, multiple steps, geometric/material nonlinearities, and thermal/contact analyses are rejected for this linear stiffness formulation. The maximum-stress check covers local `σ11`, `σ22`, and `τ12`; the separate simulation section retains its optional Puck check.

## Cantilever example and validation

The test model is a 100 × 10 mm shell beam, using 80 quadratic S8R elements and 289 reference-surface nodes. `tools/beam_mesh.py` generates the mesh for import. In PrePoMax, a linear static step, a fixed constraint on `FIXED_EDGE` for DOFs 1–6, and a concentrated y force of −1 N on each of the nine `TIP_EDGE` nodes were created. The native model and input were saved with PrePoMax's own save/export commands:

```text
D:\DataBin_Composite_Automation\cantilever_beam\cantilever_prepo.pmx
D:\DataBin_Composite_Automation\cantilever_beam\cantilever_prepo.inp
```

The baseline is `[45/−45/90/0]s`, with eight plies of 0.25 mm each. Both methods recovered `[0/0/0/0]s` at 1° resolution after 53 real FEM evaluations.

| Quantity | Initial layup | Optimized layup |
| --- | ---: | ---: |
| Compliance in N mm | 3.2131000 | 1.2903496 |
| Maximum reference-node displacement in mm | 0.3580192 | 0.1438428 |
| Stiffness relative to baseline | 1 | 2.4901004 |
| Maximum in-plane stress utilization | 0.0709631 | 0.0177817 |
| Variation of peak ply utilizations | 0.270200 | 0.00037124 |

The independent checks balance the 9 N load against support reactions, compare integrated energy against `Fᵀu` (0.0279% difference), and compare axial-layup tip displacement against a shear-flexible beam reference (0.0635% difference). The starting laminate also agrees with an independent classical-laminate beam reference within 3.46%, and the uniform 45° case is checked against the axial response to catch material-axis projection errors. The optimized stiffness increases by 149.01% relative to the starting laminate. These measurements validate this beam and load case. The low achieved utilization is expected with fixed sizing; the laminate is not fully stressed.

```powershell
.\runs\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\runs\.venv\Scripts\python.exe tools\beam_benchmark.py
.\runs\.venv\Scripts\python.exe tools\test_browser_workflow.py --keep-open
```

The browser test runs both methods through Chrome, checks sizing and symmetry, compares the imported mesh/support/load/step cards with the optimized input, downloads reports, checks for browser script errors, and verifies that the requested FRD is loaded in PrePoMax.

## Saved results

Each run retains the original input, every candidate's solver files, `baseline.*`, `optimized.*`, `report.json`, `history.csv`, and a JSON-lines evaluation log. The report records the exact material, strength allowables, sizing, starting design, final design, constraints, and termination condition. All integration-point stresses used for optimization remain in the raw DAT files.

The PrePoMax button opens an identical FRD copy in `prepomax_viewer`. This avoids the viewer's automatic import of a same-name DAT containing composite local-axis stress output. The launcher handles only the new viewer's recovery prompt and verifies its result selector before reporting display success. Existing model windows and recovery files are not edited by the launcher. The deformation scale shown in PrePoMax affects the illustration; its displacement legend reports the actual result.
