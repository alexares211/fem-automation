"""Assemble CalculiX .inp keyword blocks from the browser's inputs.

Has the material block and the composite section (layup) block. The mesh
block, supports (*Nset / *Boundary) and the full-file writer come next.
"""
from ply_orientation import orientation_vectors, format_vector_line

# CalculiX *ELASTIC, TYPE=ENGINEERING CONSTANTS wants exactly 8 values on the
# first data line and the 9th (G23) alone on a continuation line. All 9 on one
# line makes ccx 2.22 crash silently (access violation, no *ERROR text).
ENGINEERING_CONSTANT_KEYS = ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23")


class InpWriteError(Exception):
    pass


# Single element set for the whole part -- not user-configurable. A one-part
# composite shell only ever needs "all elements" in one section.
ELSET_NAME = "EALL"


def _fmt(v):
    """Match PrePoMax's number style: '135100.' for integers, plain decimal otherwise."""
    f = float(v)
    return f"{int(f)}." if f == int(f) else repr(f)


def _fmt_coord(v):
    return f"{float(v):.10g}"


SHELL_TYPES = ("S6", "S8R", "S3", "S4")


def mesh_block(nodes, elements, element_type, elset_name=ELSET_NAME):
    """Return the '*Node' + '*Element' blocks for the whole mesh.

    nodes         : {id: [x, y, z]}
    elements      : list of connectivity lists (CalculiX/gmsh node order)
    element_type  : one of SHELL_TYPES
    elset_name    : the set every element is put in, created right on the
                    *Element card (default "EALL" = all elements)
    """
    elset_name = _check_name("Element set", elset_name)
    if element_type not in SHELL_TYPES:
        raise InpWriteError(f"Unsupported element type {element_type!r}.")
    if not nodes or not elements:
        raise InpWriteError("Mesh has no nodes / elements.")

    lines = ["*Node"]
    for nid in sorted(nodes, key=int):
        x, y, z = nodes[nid]
        lines.append(f"{int(nid)}, {_fmt_coord(x)}, {_fmt_coord(y)}, {_fmt_coord(z)}")

    lines.append(f"*Element, Type={element_type}, Elset={elset_name}")
    for i, conn in enumerate(elements, start=1):
        lines.append(f"{i}, " + ", ".join(str(int(n)) for n in conn))

    return "\n".join(lines)


def material_block(name, constants):
    """Return the '*Material' + '*Elastic, Type=Engineering constants' text
    (no trailing newline).

    name       : material name (no spaces or commas)
    constants  : dict keyed by ENGINEERING_CONSTANT_KEYS, or a 9-item sequence
                 in that order (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)
    """
    name = (name or "").strip()
    if not name:
        raise InpWriteError("Material name is required.")
    if any(c in name for c in ", \t\n"):
        raise InpWriteError("Material name must not contain spaces or commas.")

    if isinstance(constants, dict):
        missing = [k for k in ENGINEERING_CONSTANT_KEYS if k not in constants]
        if missing:
            raise InpWriteError("Missing engineering constant(s): " + ", ".join(missing))
        raw = [constants[k] for k in ENGINEERING_CONSTANT_KEYS]
    else:
        raw = list(constants)
        if len(raw) != 9:
            raise InpWriteError(
                "Expected 9 engineering constants (E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)."
            )

    try:
        nums = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise InpWriteError("All engineering constants must be numbers.")

    for key, val in zip(ENGINEERING_CONSTANT_KEYS, nums):
        if key[0] in ("E", "G") and val <= 0:
            raise InpWriteError(f"{key} must be positive (got {val}).")

    line1 = ", ".join(_fmt(v) for v in nums[:8]) + ","
    line2 = _fmt(nums[8])
    return (
        f"*Material, Name={name}\n"
        f"*Elastic, Type=Engineering constants\n"
        f"{line1}\n"
        f"{line2}"
    )


def parse_layup(text):
    """'0/45/-45/90' or '0, 45, -45, 90' -> [0.0, 45.0, -45.0, 90.0]."""
    if not str(text or "").strip():
        raise InpWriteError("Enter at least one ply angle.")
    parts = [p.strip() for chunk in str(text).split("/") for p in chunk.split(",")]
    parts = [p for p in parts if p]
    try:
        angles = [float(p) for p in parts]
    except ValueError:
        raise InpWriteError("Layup must be numbers separated by '/' or ','.")
    if not angles:
        raise InpWriteError("Enter at least one ply angle.")
    return angles


def _check_name(label, name):
    name = (name or "").strip()
    if not name:
        raise InpWriteError(f"{label} name is required.")
    if any(c in name for c in ", \t\n"):
        raise InpWriteError(f"{label} name must not contain spaces or commas.")
    return name


def section_block(material, angles, thickness, symmetric=False, elset=ELSET_NAME):
    """Return one uniquely-named '*Orientation' per ply plus the
    '*Shell section, Elset=..., Composite' card and its ply lines
    (no trailing newline). Ply names are AutoPly1, AutoPly2, ... to match
    ply_orientation.generate_inp.
    """
    elset = _check_name("Element set", elset)
    material = _check_name("Material", material)

    try:
        t = float(thickness)
    except (TypeError, ValueError):
        raise InpWriteError("Enter a ply thickness in the Layup section.")
    if t <= 0:
        raise InpWriteError("Ply thickness must be positive.")

    stack = list(angles)
    if symmetric:
        stack = stack + stack[::-1]
    if not stack:
        raise InpWriteError("Enter at least one ply angle.")

    orient_blocks, ply_lines = [], []
    for i, angle in enumerate(stack, start=1):
        name = f"AutoPly{i}"
        a, b = orientation_vectors(float(angle))
        orient_blocks.append(
            f"*Orientation, Name={name}, System=Rectangular\n{format_vector_line(a, b)}"
        )
        ply_lines.append(f"{_fmt(t)},,{material},{name}")

    return (
        "\n".join(orient_blocks)
        + f"\n*Shell section, Elset={elset}, Composite\n"
        + "\n".join(ply_lines)
    )


# ---------- supports (Lager) ----------

LAGER_ALL = "LAGER_ALL"

# Lager type -> the DOF range fixed to zero in *Boundary.
# Festlager = pinned: Ux = Uy = Uz = 0.  More types get added here later.
LAGER_DOF = {
    "festlager": (1, 3),
}


def _lager_name(index):
    return f"Lager_{int(index)}"


def _clean_lagers(lagers):
    if not lagers:
        raise InpWriteError("Place at least one support (Festlager) on the model.")
    out = []
    for lg in lagers:
        typ = (lg.get("type") or "festlager").strip().lower()
        if typ not in LAGER_DOF:
            raise InpWriteError(f"Unknown support type {typ!r}.")
        try:
            node = int(lg["node"])
            index = int(lg["index"])
        except (KeyError, TypeError, ValueError):
            raise InpWriteError("Each support needs an integer node and index.")
        out.append({"index": index, "node": node, "type": typ})
    out.sort(key=lambda l: l["index"])
    return out


def lager_nsets(lagers):
    """One *Nset per support plus a combined LAGER_ALL set (for RF output)."""
    lagers = _clean_lagers(lagers)
    lines = []
    for lg in lagers:
        lines.append(f"*Nset, Nset={_lager_name(lg['index'])}")
        lines.append(str(lg["node"]))
    lines.append(f"*Nset, Nset={LAGER_ALL}")
    lines.append(", ".join(str(lg["node"]) for lg in lagers))
    return "\n".join(lines)


def lager_boundaries(lagers):
    """The *Boundary block (goes inside the step)."""
    lagers = _clean_lagers(lagers)
    lines = ["*Boundary"]
    for lg in lagers:
        d0, d1 = LAGER_DOF[lg["type"]]
        lines.append(f"{_lager_name(lg['index'])}, {d0}, {d1}, 0")
    return "\n".join(lines)


# ---------- loads (force / prescribed displacement) ----------

LOAD_TYPES = ("force", "displacement")


def _load_name(index):
    return f"Load_{int(index)}"


def _clean_loads(loads):
    """Each load: {index, node, type in LOAD_TYPES, dir:[dx,dy,dz], mag}.
    Returns them with an extra 'comp' = magnitude * unit(dir), and drops
    loads whose components are all ~0.
    """
    out = []
    for ld in loads or []:
        typ = (ld.get("type") or "").strip().lower()
        if typ not in LOAD_TYPES:
            raise InpWriteError(f"Unknown load type {typ!r}.")
        try:
            node = int(ld["node"])
            index = int(ld["index"])
            d = [float(v) for v in (ld.get("dir") or [0, 0, 0])]
            mag = float(ld.get("mag"))
        except (KeyError, TypeError, ValueError):
            raise InpWriteError("Each load needs a node, index, direction and magnitude.")
        if len(d) != 3:
            raise InpWriteError("Load direction must be three numbers (X, Y, Z).")
        norm = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5
        if norm == 0 or mag == 0:
            continue  # nothing applied
        comp = [mag * d[i] / norm for i in range(3)]
        out.append({"index": index, "node": node, "type": typ, "comp": comp})
    out.sort(key=lambda l: l["index"])
    return out


def load_nsets(loads):
    loads = _clean_loads(loads)
    lines = []
    for ld in loads:
        lines.append(f"*Nset, Nset={_load_name(ld['index'])}")
        lines.append(str(ld["node"]))
    return "\n".join(lines)


def load_cards(loads):
    """*Cload lines for 'force' loads and *Boundary lines for 'displacement'
    loads (both go inside the step). Only non-zero components are written.
    """
    loads = _clean_loads(loads)
    cload, disp = [], []
    for ld in loads:
        name = _load_name(ld["index"])
        for dof, val in enumerate(ld["comp"], start=1):
            if abs(val) < 1e-12:
                continue
            if ld["type"] == "force":
                cload.append(f"{name}, {dof}, {_fmt(val)}")
            else:
                disp.append(f"{name}, {dof}, {dof}, {_fmt(val)}")
    out = []
    if disp:
        out.append("*Boundary\n" + "\n".join(disp))
    if cload:
        out.append("*Cload\n" + "\n".join(cload))
    return "\n".join(out)


# A plate goes geometrically nonlinear once its out-of-plane deflection is a
# meaningful fraction of its thickness; 0.5 is the usual rule of thumb.
NLGEOM_DEFLECTION_RATIO = 0.5


def auto_nlgeom(loads, ply_thickness, ply_angles, symmetric=False):
    """Decide whether the step needs *Nlgeom, without solving first.

    - any non-zero point force  -> on (the deflection it causes is unknown, so
      assume it matters)
    - any prescribed displacement >= 0.5 x total laminate thickness -> on
    - otherwise -> off (small-displacement, linear is fine and faster)
    """
    try:
        t = float(ply_thickness)
    except (TypeError, ValueError):
        return False
    n_plies = len(list(ply_angles)) * (2 if symmetric else 1)
    total_t = t * max(n_plies, 1)

    for ld in loads or []:
        typ = (ld.get("type") or "").strip().lower()
        try:
            mag = abs(float(ld.get("mag") or 0))
        except (TypeError, ValueError):
            return True  # unparseable -> be safe
        if mag == 0:
            continue
        if typ == "force":
            return True
        if typ == "displacement" and mag >= NLGEOM_DEFLECTION_RATIO * total_t:
            return True
    return False


def full_inp(nodes, elements, element_type, material_name, constants,
             ply_angles, ply_thickness, lagers, symmetric=False, nlgeom="auto",
             loads=None):
    """Assemble a complete, solvable CalculiX .inp from the browser's model:
    structured mesh + orthotropic material + composite layup + Festlager
    supports + point loads (force / prescribed displacement). Reaction force
    is written for LAGER_ALL via *Node print.

    nlgeom: "auto" (decide from the imposed loads vs. laminate thickness),
    or an explicit bool to force it.
    """
    if nlgeom == "auto":
        nlgeom = auto_nlgeom(loads, ply_thickness, ply_angles, symmetric)

    load_ns = load_nsets(loads)
    load_c = load_cards(loads)
    parts = [
        "*Heading",
        "Generated by fem-automation (STEP -> structured grid)",
        mesh_block(nodes, elements, element_type, ELSET_NAME),
        lager_nsets(lagers),
    ]
    if load_ns:
        parts.append(load_ns)
    parts += [
        material_block(material_name, constants),
        section_block(material_name, ply_angles, ply_thickness, symmetric),
        "*Step, Nlgeom" if nlgeom else "*Step",
        # nonlinear: ramp the load in 10 increments so CalculiX can iterate
        # (and auto-cut back) instead of applying everything at once
        "*Static\n0.1, 1.0" if nlgeom else "*Static",
        lager_boundaries(lagers),
    ]
    if load_c:
        parts.append(load_c)
    parts += [
        f"*Node print, Nset={LAGER_ALL}, Global=Yes",
        "RF, U",
        "*Node file",
        "RF, U",
        "*El file",
        "S, E",
        "*End step",
    ]
    return "\n".join(parts) + "\n"
