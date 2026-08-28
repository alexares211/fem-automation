"""Generate a CalculiX .inp with a laminate's fiber orientations swapped in.

Assumes the plate lies in the global X-Z plane (Y normal) -- the same
convention as TorsionPlatte.inp. Builds a fresh N-ply *Shell section,
Composite card with a uniquely-named *Orientation per ply (rather than
reusing whatever orientation names the template happened to share across
plies), so each ply's angle can be set independently -- this is what lets
generate_inp accept any number of plies, each with its own angle, instead
of requiring the template be reduced to exactly one ply first.
"""
import math
import re


class PlyEditError(Exception):
    pass


def angle_range(start, end, step):
    if step <= 0:
        raise PlyEditError("Step must be a positive number.")
    if end < start:
        start, end = end, start
    n_steps = int(round((end - start) / step))
    values = [round(start + i * step, 6) for i in range(n_steps + 1)]
    if abs(values[-1] - end) > 1e-6:
        values.append(round(end, 6))
    return values


def orientation_vectors(angle_deg):
    """a = fiber direction, b = a rotated +90 deg, both in the X-Z plane."""
    theta = math.radians(angle_deg)
    a = (math.cos(theta), 0.0, math.sin(theta))
    b = (-math.sin(theta), 0.0, math.cos(theta))
    return a, b


def format_vector_line(a, b):
    vals = list(a) + list(b)
    vals = [0.0 if abs(v) < 1e-9 else v for v in vals]
    return ", ".join(f"{v:.6f}" for v in vals)


SHELL_SECTION_RE = re.compile(
    r'^(\*Shell section,\s*Elset=(\S+?),\s*Composite\s*\n)'
    r'((?:[^\*\n][^\n]*\n?)*)',
    re.IGNORECASE | re.MULTILINE,
)


def find_reference_ply(text):
    """Locate the Shell section, Composite block and read its first ply's
    thickness/material -- used as the template for every ply in the
    regenerated layup, regardless of how many plies the file originally had.

    Returns (elset, thickness, material).
    """
    m = SHELL_SECTION_RE.search(text)
    if not m:
        raise PlyEditError(
"Could not find a '*Shell section, ..., Composite' block in this .inp file."
        )
    elset = m.group(2)
    ply_lines = [l for l in m.group(3).splitlines() if l.strip()]
    if not ply_lines:
        raise PlyEditError("The Composite section has no ply lines to use as a template.")
    fields = [f.strip() for f in ply_lines[0].split(",")]
    if len(fields) < 4:
        raise PlyEditError(f"Could not parse the ply line: {ply_lines[0]!r}")
    thickness, material = fields[0], fields[2]
    return elset, thickness, material


ORIENTATION_NAME_RE = re.compile(
    r'^\*Orientation,\s*Name=(\S+?),', re.IGNORECASE | re.MULTILINE
)


def unique_orientation_names(text, count):
    existing = {m.group(1).strip().lower() for m in ORIENTATION_NAME_RE.finditer(text)}
    names = []
    i = 1
    while len(names) < count:
        candidate = f"AutoPly{i}"
        if candidate.lower() not in existing:
            names.append(candidate)
            existing.add(candidate.lower())
        i += 1
    return names


NODE_PRINT_RE = re.compile(
    r'^\*Node print,\s*Nset=(\S+?)\s*(?:,[^\n]*)?\n([^\n]*)\n',
    re.IGNORECASE | re.MULTILINE,
)
END_STEP_RE = re.compile(r'^\*End step\s*$', re.IGNORECASE | re.MULTILINE)


def detect_rf_nset(text):
    """Find whichever node set the .inp already requests RF history output for.

    Lets the tool discover the right node set on its own (e.g. a name a
    PrePoMax GUI selection generated, or NS_Fixed) instead of asking the
    user to type it in.
    """
    for m in NODE_PRINT_RE.finditer(text):
        if "RF" in m.group(2).upper():
            return m.group(1).strip()
    return None


def ensure_rf_output(text, nset_name="NS_Fixed"):
    """Guarantee a '*Node print, Nset=<nset_name> / RF' block exists in the step.

    Reaction force is only reliably correct when read from this History
    Output / .dat path -- the Field Output / .frd FORC value is known to be
    wrong (always 0) for multi-ply composite shells in this ccx build.
    Building this ourselves rather than relying on whatever History Output
    happened to already exist (e.g. a GUI selection with an arbitrary
    internal name) keeps the automation self-contained.
    """
    for m in NODE_PRINT_RE.finditer(text):
        if m.group(1).strip().lower() == nset_name.strip().lower() and "RF" in m.group(2).upper():
            return text

    nset_pattern = re.compile(
        rf'^\*Nset,\s*Nset={re.escape(nset_name)}\s*\n', re.IGNORECASE | re.MULTILINE
    )
    if not nset_pattern.search(text):
        raise PlyEditError(
            f"Cannot add a reaction-force output request: node set '{nset_name}' "
"was not found in this .inp file."
        )

    end_step_m = END_STEP_RE.search(text)
    if not end_step_m:
        raise PlyEditError(
"Could not find '*End step' to insert the reaction-force output request before."
        )

    insertion = f"*Node print, Nset={nset_name}, Global=Yes\nRF\n"
    insert_at = end_step_m.start()
    return text[:insert_at] + insertion + text[insert_at:]


def generate_inp(input_path, output_path, angles, rf_nset=None):
    """angles: a list of per-ply angles (degrees), one entry per ply, in
    stacking order. len(angles) sets the number of plies in the regenerated
    layup -- it does not need to match however many plies the template had.
    """
    if not angles:
        raise PlyEditError("Need at least one ply angle.")

    with open(input_path, "r") as f:
        text = f.read()

    if rf_nset is None:
        rf_nset = detect_rf_nset(text) or "NS_Fixed"

    elset, thickness, material = find_reference_ply(text)

    names = unique_orientation_names(text, len(angles))
    orientation_blocks = []
    ply_lines = []
    for name, angle_deg in zip(names, angles):
        a, b = orientation_vectors(angle_deg)
        vector_line = format_vector_line(a, b)
        orientation_blocks.append(
            f"*Orientation, Name={name}, System=Rectangular\n{vector_line}\n"
        )
        ply_lines.append(f"{thickness},,{material},{name}")

    m = SHELL_SECTION_RE.search(text)
    new_ply_body = "\n".join(ply_lines) + "\n"
    new_text = text[: m.start()] + "".join(orientation_blocks) + m.group(1) + new_ply_body + text[m.end() :]

    new_text = ensure_rf_output(new_text, nset_name=rf_nset)

    with open(output_path, "w") as f:
        f.write(new_text)

    return {
        "elset": elset,
        "thickness": thickness,
        "material": material,
        "num_plies": len(angles),
        "angles": angles,
        "orientation_names": names,
        "output_path": output_path,
        "rf_nset": rf_nset,
    }
