"""Generate a CalculiX .inp with a single ply's fiber orientation swapped in.

Assumes the plate lies in the global X-Z plane (Y normal) -- the same
convention as TorsionPlatte.inp. Only touches the *Orientation vector line
for the one ply currently used by the *Shell section, Composite card;
everything else in the file is left untouched.
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
    r'^\*Shell section,\s*Elset=(\S+?),\s*Composite\s*\n'
    r'((?:[^\*\n][^\n]*\n?)*)',
    re.IGNORECASE | re.MULTILINE,
)


def find_single_ply(text):
    """Locate the Shell section, Composite block and confirm it has exactly one ply.

    Returns (elset, thickness, material, orientation_name).
    """
    m = SHELL_SECTION_RE.search(text)
    if not m:
        raise PlyEditError(
            "Could not find a '*Shell section, ..., Composite' block in this .inp file."
        )
    elset = m.group(1)
    ply_lines = [l for l in m.group(2).splitlines() if l.strip()]
    if len(ply_lines) != 1:
        raise PlyEditError(
            f"Expected exactly 1 ply in the Composite section, found {len(ply_lines)}. "
            "This milestone only supports single-ply .inp files -- reduce the section "
            "to one active ply line before using this tool."
        )
    fields = [f.strip() for f in ply_lines[0].split(",")]
    if len(fields) < 4:
        raise PlyEditError(f"Could not parse the ply line: {ply_lines[0]!r}")
    thickness, _blank, material, orientation_name = fields[0], fields[1], fields[2], fields[3]
    return elset, thickness, material, orientation_name


def orientation_block_re(orientation_name):
    return re.compile(
        rf'(^\*Orientation,\s*Name={re.escape(orientation_name)},\s*System=Rectangular\s*\n)'
        rf'([^\n]+)(\n)',
        re.IGNORECASE | re.MULTILINE,
    )


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


def generate_inp(input_path, output_path, angle_deg, rf_nset=None):
    with open(input_path, "r") as f:
        text = f.read()

    if rf_nset is None:
        rf_nset = detect_rf_nset(text) or "NS_Fixed"

    elset, thickness, material, orientation_name = find_single_ply(text)

    pattern = orientation_block_re(orientation_name)
    m = pattern.search(text)
    if not m:
        raise PlyEditError(
            f"Ply references orientation '{orientation_name}', but no matching "
            f"'*Orientation, Name={orientation_name}, System=Rectangular' block was found."
        )
    old_line = m.group(2)

    a, b = orientation_vectors(angle_deg)
    new_line = format_vector_line(a, b)

    new_text = pattern.sub(lambda mm: mm.group(1) + new_line + mm.group(3), text, count=1)
    new_text = ensure_rf_output(new_text, nset_name=rf_nset)

    with open(output_path, "w") as f:
        f.write(new_text)

    return {
        "elset": elset,
        "thickness": thickness,
        "material": material,
        "orientation_name": orientation_name,
        "old_vector_line": old_line,
        "new_vector_line": new_line,
        "angle_deg": angle_deg,
        "output_path": output_path,
        "rf_nset": rf_nset,
    }
