# Fixed sizing composite optimization methods

The stage-one design variables are one uniform fiber angle per independent ply on a planar shell patch. Ply count, ply thickness, geometry, supports, and load magnitudes stay fixed. Symmetric laminates use mirrored angle pairs. A design is accepted as feasible only when its reported stress constraints are satisfied.

The material reference axes are computed from the selected shell mesh. The first is global X projected into the tangent plane, with global Y as fallback when X is normal. The second follows the first element's oriented normal. All ply directions rotate within that plane. Nonplanar patches and inconsistent normals are rejected before a search; they require an element-wise material-direction field. This prevents a global X–Z rotation from collapsing off-axis angles onto X on an X–Y shell.

## Research basis and interpretation

Classical fully stressed design is a sizing criterion for a lightweight structure whose members reach their strength limit under at least one load case. Extending it to laminates involves both orientation and thickness. The review by [Ghiasi, Fayazbakhsh, Pasini and Lessard](https://mdog.mcgill.ca/paper%20pdf%20to%20upload/Optimum%20stacking%20sequence%20design%20part2.pdf), sections 4.2–4.3, distinguishes principal-direction alignment, energy criteria, and fully stressed sizing. With sizing frozen, reaching full utilization generally cannot be required while also maximizing stiffness.

[Fine and Springer](https://journals.sagepub.com/doi/10.1177/002199839703102301) describe a composite layup procedure targeting strength, weight, and manufacturability. Its HALO procedure includes sizing and a Tsai–Wu strength criterion. The present fixed-sizing implementation uses the explicit in-plane maximum-stress constraints below; it does not reproduce HALO or substitute its failure criterion.

The stiffness formulation is also consistent with the distinction between minimum compliance and stress constraints in [Guo, Meng and Wang](https://www.sciencedirect.com/science/article/pii/S0965997824000954). Their topology and fiber-field algorithm is a different method. Here, every candidate is verified by FEM and a direct search changes the uniform ply angles.

In this project, **equal stress** is defined as balancing each ply's peak normalized utilization, using its own directional allowables. Equality of raw longitudinal and transverse stresses would ignore their different strengths. The coefficient-of-variation constraint below makes this interpretation measurable and editable in the interface.

## Stiffness objective

For a linear elastic model with homogeneous supports and fixed force vector `F`, equilibrium is `K(θ)u = F`. Minimize

```text
C(θ) = Fᵀu = uᵀK(θ)u = 2U(θ).
```

Under fixed nonzero prescribed displacement with no applied force loading, maximize `2U` instead. For a fixed reference load pattern, the reported stiffness ratio is `C_initial/C_best` in force control and `U_best/U_initial` in displacement control. Maximizing support reactions under force control would instead measure a quantity fixed by equilibrium.

The [CalculiX 2.22 manual](https://www.dhondt.de/ccx_2.22.pdf), `*EL PRINT`, defines `ELSE` as whole-element internal energy and `ENER` as energy density. The solver deck requests `ELSE` with `TOTALS=ONLY` over all structural elements, including other parts outside the laminate. This gives integrated strain energy for the supported linear elastic case. Energy density is never summed without integration weights.

## In-plane stress utilization

At every integration point in each ply's local material coordinates, compute

```text
r11 = |σ11| / Xt when σ11 ≥ 0, otherwise |σ11| / Xc
r22 = |σ22| / Yt when σ22 ≥ 0, otherwise |σ22| / Yc
r12 = |τ12| / S
r_p = maximum of r11, r22 and r12 over every integration point in ply p
```

`Xt`, `Xc`, `Yt`, `Yc`, and `S` are the five positive strength inputs. The constraint is `max(r_p) ≤ r_limit`; a limit of 0.5 corresponds to a factor of two against this maximum-stress measure. The calculation addresses in-plane lamina failure and does not evaluate buckling, interlaminar failure, or progressive damage. Peak stresses near a clamp remain sensitive to the mesh and local boundary idealization.

The DAT output retains the per-ply orientation label, which identifies the layer, and local stress components. FRD stress contours are interpolated visualization data; the optimizer uses integration-point DAT values for its constraint checks.

## Fully stressed orientation adaptation

Solve the following fixed-sizing problem:

```text
minimize C(θ)             under force control
subject to max(r_p) ≤ r_limit
           θ lies on the selected manufacturing angle grid
           fixed count, fixed thickness, prescribed symmetry
```

The displacement-controlled variant maximizes `2U` under the same constraints. A separate flag records whether all plies reach within 5% of the chosen utilization limit. A low-utilization feasible stiffness design remains a valid result; the interface explicitly states that full utilization was not achieved.

## Equal stress orientation adaptation

Add the constraint

```text
CV(r) = population standard deviation of r_p / mean(r_p) ≤ tolerance.
```

The objective remains maximum stiffness under this additional constraint. For example, tolerance 0.10 limits the spread to 10% of the mean. A laminate in bending through its thickness may need to sacrifice stiffness to balance ply peaks, or may have no feasible equal-utilization solution with fixed sizing. The application reports unmet constraints rather than presenting the lowest-compliance infeasible design as a successful optimization.

## Search and termination

Both formulations use a deterministic direct search. The initial layup and uniform 0°, 90°, +45°, and −45° seeds receive real solves. The search then tests positive and negative coordinate moves, prioritizing high-utilization plies for the fully stressed adaptation and utilization outliers for equal stress. The angular move decreases until it reaches the selected grid resolution. Mirrored pairs move together when symmetry is enabled.

Candidates are ranked by constraint feasibility first, total violation second when infeasible, and the stiffness objective thereafter. Physically equivalent angles are wrapped with a 180° period and cached so identical designs are not solved twice. The evaluation budget is a hard limit on real completed solves. Cancellation terminates the active solver and retains completed evaluations.

Termination at the angular resolution means that the checked coordinate moves did not improve the current design. It is a local-search condition, not a global optimality certificate. The response and saved report distinguish resolution termination, budget exhaustion, user cancellation, and solver failure.

## PrePoMax workflow

The beam mesh is imported into PrePoMax, where the clamp and transverse tip load are defined and the `.pmx` and CalculiX `.inp` are saved. The app replaces the chosen section with its material and composite stack, retains the imported conditions, solves each candidate, and archives the best verified result. [PrePoMax 2.5 documentation](https://prepomax.fs.um.si/wp-content/uploads/2026/03/PrePoMax-v2.5.0-manual.pdf) describes CalculiX input export and opening FRD results.

The viewer receives an identical FRD copy in a dedicated subdirectory because its automatic DAT history reader cannot parse all of this workflow's local-axis composite stress rows. Raw DAT values remain available for auditing. Display verification checks the actual FRD pathname in the new viewer's result selector, rather than treating process launch as proof of a loaded result.
