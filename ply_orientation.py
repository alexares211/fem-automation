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


def generate_inp(input_path, output_path, angle_deg):
    with open(input_path, "r") as f:
        text = f.read()

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
    }