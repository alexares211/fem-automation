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


def _fmt(v):
    """Match PrePoMax's number style: '135100.' for integers, plain decimal otherwise."""
    f = float(v)
    return f"{int(f)}." if f == int(f) else repr(f)


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


def section_block(elset, material, angles, thickness, symmetric=False):
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
        raise InpWriteError("Ply thickness must be a number.")
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
