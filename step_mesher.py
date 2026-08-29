"""Mesh a STEP file with gmsh for a CalculiX composite shell model.

Scope for now: STEP in -> 2D surface mesh out, returned as plain data for the
browser 3D viewport. Generating the actual CalculiX .inp from this mesh
(instead of editing an existing template) is a later step.

Only the intuitive knobs are exposed here (element family + target element
size). Everything else -- second-order node placement, meshing algorithm,
optimisation passes -- stays on gmsh defaults; the deeper solver settings
still come from the template .inp for now.
"""
import os
import gmsh

# CalculiX shell element  ->  how gmsh should build it.
# NOTE: *SHELL SECTION, COMPOSITE in CalculiX (ccx 2.22) only accepts S6 or
# S8R. S3/S4 are offered for a quick first-order preview / non-composite use.
ELEMENT_TYPES = {
    "S6":  {"order": 2, "recombine": False, "desc": "6-node triangle, 2nd order (composite)"},
    "S8R": {"order": 2, "recombine": True,  "desc": "8-node quad, 2nd order, reduced (composite)"},
    "S3":  {"order": 1, "recombine": False, "desc": "3-node triangle, 1st order (preview)"},
    "S4":  {"order": 1, "recombine": True,  "desc": "4-node quad, 1st order (preview)"},
}


class StepMeshError(Exception):
    pass


def mesh_step(step_path, element_type="S6", target_size=None, out_dir=None):
    """Import `step_path`, mesh its surfaces, and return a dict:

        element_type, element_desc, target_size,
        num_nodes, num_elements, bbox [xmin,ymin,zmin,xmax,ymax,zmax],
        nodes {id: [x,y,z]}, elements [[node ids], ...],
        render {positions [x,y,z,...], tris [i,i,i,...], edges [i,i,...]}

    `render` uses a dense 0-based index space (corner nodes only) so the
    browser can drop it straight into a BufferGeometry.
    """
    if not os.path.isfile(step_path):
        raise StepMeshError("No STEP file at: " + step_path)
    if element_type not in ELEMENT_TYPES:
        raise StepMeshError(
            "Unknown element type %r (expected one of: %s)"
            % (element_type, ", ".join(ELEMENT_TYPES))
        )
    cfg = ELEMENT_TYPES[element_type]

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("step_model")

        try:
            gmsh.model.occ.importShapes(step_path)
            gmsh.model.occ.synchronize()
        except Exception as e:
            raise StepMeshError("gmsh could not read the STEP file: " + str(e))

        surfaces = gmsh.model.getEntities(dim=2)
        if not surfaces:
            raise StepMeshError(
                "The STEP file has no surfaces to mesh. For a shell model it "
                "should contain a sheet / mid-surface body, not only a solid."
            )

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5

        try:
            size = float(target_size) if target_size not in (None, "") else 0.0
        except (TypeError, ValueError):
            raise StepMeshError("Target element size must be a number.")
        if size <= 0:
            size = diag / 25.0 if diag > 0 else 1.0

        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

        if cfg["recombine"]:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # blossom
        else:
            gmsh.option.setNumber("Mesh.RecombineAll", 0)

        gmsh.model.mesh.generate(2)
        if cfg["order"] == 2:
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)  # 8-node quad, not 9
            gmsh.model.mesh.setOrder(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = {
            int(tag): [node_coords[3 * i], node_coords[3 * i + 1], node_coords[3 * i + 2]]
            for i, tag in enumerate(node_tags)
        }

        elements = []
        etypes, etags, enodes = gmsh.model.mesh.getElements(dim=2)
        for et, tags, conn in zip(etypes, etags, enodes):
            npe = gmsh.model.mesh.getElementProperties(et)[3]
            for k in range(len(tags)):
                elements.append([int(x) for x in conn[k * npe:(k + 1) * npe]])

        if not elements:
            raise StepMeshError("Meshing produced no surface elements.")

        result = {
            "element_type": element_type,
            "element_desc": cfg["desc"],
            "target_size": size,
            "num_nodes": len(nodes),
            "num_elements": len(elements),
            "bbox": [xmin, ymin, zmin, xmax, ymax, zmax],
            "nodes": nodes,
            "elements": elements,
            "render": _render_geometry(nodes, elements, element_type),
        }

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            msh_path = os.path.join(out_dir, "step_mesh.msh")
            gmsh.write(msh_path)
            result["msh_path"] = msh_path

        return result
    finally:
        gmsh.finalize()


def _render_geometry(nodes, elements, element_type):
    is_quad = element_type in ("S4", "S8R")
    corners = 4 if is_quad else 3

    # corner nodes only (drop mid-side nodes of 2nd-order elements), in first-
    # seen order, so `positions` stays compact
    used = []
    seen = set()
    for el in elements:
        for n in el[:corners]:
            if n not in seen:
                seen.add(n)
                used.append(n)
    index_of = {nid: i for i, nid in enumerate(used)}

    positions = []
    for nid in used:
        positions.extend(nodes[nid])

    tris = []
    edges = set()
    for el in elements:
        c = [index_of[n] for n in el[:corners]]
        if is_quad:
            tris += [c[0], c[1], c[2], c[0], c[2], c[3]]
            loop = ((c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0]))
        else:
            tris += [c[0], c[1], c[2]]
            loop = ((c[0], c[1]), (c[1], c[2]), (c[2], c[0]))
        for a, b in loop:
            edges.add((a, b) if a < b else (b, a))

    edge_flat = []
    for a, b in edges:
        edge_flat += [a, b]

    return {"positions": positions, "tris": tris, "edges": edge_flat}


def _bounding_box(step_path):
    if not os.path.isfile(step_path):
        raise StepMeshError("No STEP file at: " + step_path)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("bbox")
        try:
            gmsh.model.occ.importShapes(step_path)
            gmsh.model.occ.synchronize()
        except Exception as e:
            raise StepMeshError("gmsh could not read the STEP file: " + str(e))
        return gmsh.model.getBoundingBox(-1, -1)
    finally:
        gmsh.finalize()


def structured_grid(step_path, spacing=1.0):
    """A regular S8R (8-node serendipity quad) mesh over the STEP's bounding
    box, treated as a flat rectangle. `spacing` is the node spacing (mm) in
    both in-plane directions; the thin axis (smallest bbox extent) is dropped
    to a single plane. Returns the same dict shape as mesh_step, with
    element_type 'S8R' and the full `nodes` table (needed to drop supports
    onto exact nodes).
    """
    try:
        spacing = float(spacing)
    except (TypeError, ValueError):
        raise StepMeshError("Node spacing must be a number.")
    if spacing <= 0:
        raise StepMeshError("Node spacing must be positive.")

    xmin, ymin, zmin, xmax, ymax, zmax = _bounding_box(step_path)
    lo = [xmin, ymin, zmin]
    ext = [xmax - xmin, ymax - ymin, zmax - zmin]
    flat = ext.index(min(ext))            # the through-thickness axis
    ip = [a for a in (0, 1, 2) if a != flat]   # the two in-plane axes
    w0 = lo[flat] + ext[flat] / 2.0

    Lu, Lv = ext[ip[0]], ext[ip[1]]
    if Lu <= 0 or Lv <= 0:
        raise StepMeshError("The STEP bounding box is degenerate in-plane.")
    ncu = max(1, round(Lu / spacing))
    ncv = max(1, round(Lv / spacing))
    # every corner + edge-midside node is sent to the browser as JSON, so keep
    # the total node count sane
    if (ncu + 1) * (ncv + 1) > 120000:
        raise StepMeshError(
            f"{ncu}x{ncv} cells (~{3 * ncu * ncv // 1000}k nodes) is too fine here "
            f"-- increase the node spacing."
        )
    du, dv = Lu / ncu, Lv / ncv
    ou, ov = lo[ip[0]], lo[ip[1]]

    def xyz(u, v):
        p = [0.0, 0.0, 0.0]
        p[ip[0]] = ou + u
        p[ip[1]] = ov + v
        p[flat] = w0
        return p

    nodes = {}
    nid = [0]

    def add(u, v):
        nid[0] += 1
        nodes[nid[0]] = xyz(u, v)
        return nid[0]

    corner = {}
    for j in range(ncv + 1):
        for i in range(ncu + 1):
            corner[(i, j)] = add(i * du, j * dv)
    umid = {}
    for j in range(ncv + 1):
        for i in range(ncu):
            umid[(i, j)] = add((i + 0.5) * du, j * dv)
    vmid = {}
    for j in range(ncv):
        for i in range(ncu + 1):
            vmid[(i, j)] = add(i * du, (j + 0.5) * dv)

    # CalculiX S8 order: 4 corners CCW, then the 4 edge mid-side nodes
    elements = []
    for j in range(ncv):
        for i in range(ncu):
            elements.append([
                corner[(i, j)], corner[(i + 1, j)], corner[(i + 1, j + 1)], corner[(i, j + 1)],
                umid[(i, j)], vmid[(i + 1, j)], umid[(i, j + 1)], vmid[(i, j)],
            ])

    return {
        "element_type": "S8R",
        "element_desc": "8-node quad, 2nd order, reduced (structured grid)",
        "spacing": spacing,
        "cells": [ncu, ncv],
        "num_nodes": len(nodes),
        "num_elements": len(elements),
        "bbox": [xmin, ymin, zmin, xmax, ymax, zmax],
        "nodes": nodes,
        "elements": elements,
        "render": _render_geometry(nodes, elements, "S8R"),
    }
