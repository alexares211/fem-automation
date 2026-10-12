# Parse reaction force (RF) values out of a CalculiX .dat file.
#
# *NODE PRINT, NSET=..., writes blocks shaped like:
#
#     forces (fx,fy,fz) for set NS_FIXED and time  1.0000000e+00
#
#            201 0.000000E+00-9.545213E+02 0.000000E+00
#            481  ...
#
# Columns are fixed-width, so a negative value's '-' sign can sit directly
# against the previous number with no separating space (same convention as
# the .frd file) -- values are found by scanning for scientific-notation
# numbers directly rather than splitting on whitespace, so spacing doesn't
# matter either way.
#
# A nonlinear step prints one such block per converged increment (matching
# the sub-increment rows seen in PrePoMax's History Output table) -- the
# final, fully-loaded result is the block with the largest time value.
# CalculiX set names are case-insensitive internally, so matching here is
# case-insensitive too.

import re
import math


class DatParseError(Exception):
    pass


_HEADER_RE = re.compile(
    r'forces\s*\(fx,fy,fz\)\s*for\s*set\s*(\S+)\s*and\s*time\s*([0-9.eE+\-]+)',
    re.IGNORECASE,
)
_FLOAT_RE = re.compile(r'[-+]?\d+\.\d+[Ee][-+]?\d+')
_ID_PREFIX_RE = re.compile(r'(\d+)\s*$')


def _parse_body_rows(body_text):
    rows = {}
    started = False
    for line in body_text.splitlines():
        line = line.strip()
        if not line:
            if started:
                break
            continue
        floats = _FLOAT_RE.findall(line)
        if len(floats) < 3:
            continue
        first_pos = line.find(floats[0])
        prefix = line[:first_pos]
        id_match = _ID_PREFIX_RE.search(prefix)
        if not id_match:
            continue
        node_id = int(id_match.group(1))
        fx, fy, fz = (float(v) for v in floats[:3])
        rows[node_id] = (fx, fy, fz)
        started = True
    return rows


def read_reaction_force(dat_path, nset_name):
    with open(dat_path, "r", errors="ignore") as f:
        text = f.read()

    headers = list(_HEADER_RE.finditer(text))
    matches = []
    for idx, m in enumerate(headers):
        set_name, time_str = m.group(1), m.group(2)
        if set_name.strip().lower() != nset_name.strip().lower():
            continue
        body_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        body_text = text[m.end():body_end]
        rows = _parse_body_rows(body_text)
        if rows:
            matches.append((float(time_str), rows))

    if not matches:
        raise DatParseError(
            "No 'forces (fx,fy,fz) for set " + nset_name + "' block found in " + dat_path
        )

    matches.sort(key=lambda t: t[0])
    final_time, final_rows = matches[-1]

    sum_fx = sum(v[0] for v in final_rows.values())
    sum_fy = sum(v[1] for v in final_rows.values())
    sum_fz = sum(v[2] for v in final_rows.values())
    magnitude = math.sqrt(sum_fx ** 2 + sum_fy ** 2 + sum_fz ** 2)

    return {
        "time": final_time,
        "per_node": final_rows,
        "sum_fx": sum_fx,
        "sum_fy": sum_fy,
        "sum_fz": sum_fz,
        "magnitude": magnitude,
    }


# ---------- per-ply stress (for puck.ply_failure) ----------
#
# *El print, Elset=..., with plain "S" on a *Shell section, Composite part
# writes blocks shaped like this CONFIRMED-REAL example (verified against an
# actual ccx-produced .dat, 2026-10-12 -- see model.dat sent by the user;
# the original version of this comment described a guessed/undocumented
# format that turned out to be wrong, see git history):
#
#     stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EALL and
#     time  0.5000000E-01
#
#          1   1  5.316022E-02 -5.075319E-01 -2.318234E-02 -8.457374E-02 -8.058973E-02  6.602915E-02 AUTOPLY1_shell_00000
#          1   2  3.550983E-01 -7.247057E-01  7.767127E-02  3.405060E-01 -1.223229E-01  1.655349E-01 AUTOPLY1_shell_00000
#          ...
#          1   9 -2.355381E+00 -1.663048E-01 -1.571778E-02  2.524321E-01 -8.365164E-02  1.717588E-02 AUTOPLY2_shell_00000
#
# i.e. only TWO leading integer columns (element, integration point) -- there
# is no separate numeric "layer" column. The ply/layer number instead comes
# from the trailing orientation-name token CalculiX prints at the end of each
# row: inp_writer.section_block names each ply's *Orientation "AutoPly<N>"
# (1-based), and ccx echoes it back uppercased with a "_shell_<NNNNN>" suffix
# of its own (e.g. "AUTOPLY1_shell_00000") -- the suffix is a ccx-internal
# detail, not something inp_writer controls, so it's matched loosely here
# rather than pinned to "_shell_00000" exactly. Several integration points
# share each ply (8 per ply in the verified file); that's expected, not a
# parsing error -- evaluate_laminate() checks every one and keeps the worst.
# The six values are already in each ply's LOCAL material frame because of
# the per-ply *Orientation (sxx=S11 fiber direction, syy=S22 transverse,
# sxy=S12 in-plane shear -- exactly what puck.ply_failure wants), even
# though ccx's header always labels them generically as sxx/syy/.../syz
# regardless of which orientation is active.

_STRESS_HEADER_RE = re.compile(
    r'stresses\s*\(elem,\s*integ\.?\s*pnt\.?\s*,.*?\)\s*for\s*set\s*(\S+)\s*and\s*time\s*([0-9.eE+\-]+)',
    re.IGNORECASE | re.DOTALL,
)
_STRESS_PREFIX_RE = re.compile(r'(\d+)\s+(\d+)\s*$')
_STRESS_PLY_RE = re.compile(r'AUTOPLY(\d+)', re.IGNORECASE)
_STRESS_COMPONENTS = ("S11", "S22", "S33", "S12", "S13", "S23")


def _parse_stress_rows(body_text):
    rows = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line:
            if rows:
                break
            continue
        floats = _FLOAT_RE.findall(line)
        if len(floats) < len(_STRESS_COMPONENTS):
            continue
        floats = floats[: len(_STRESS_COMPONENTS)]
        first_pos = line.find(floats[0])
        prefix = line[:first_pos]
        m = _STRESS_PREFIX_RE.search(prefix)
        if not m:
            continue
        elem, ip = (int(v) for v in m.groups())
        ply_m = _STRESS_PLY_RE.search(line)
        if not ply_m:
            continue
        layer = int(ply_m.group(1))
        values = {k: float(v) for k, v in zip(_STRESS_COMPONENTS, floats)}
        rows.append({"elem": elem, "ip": ip, "layer": layer, **values})
    return rows


def read_ply_stresses(dat_path, elset_name):
    """Return every (element, integration point, ply layer) stress row from
    the LAST (highest-time) 'stresses (elem, integ.pnt.,sxx,syy,...)' block
    for elset_name -- same "take the final converged increment" convention
    as read_reaction_force, for the same reason (a nonlinear step prints one
    block per increment).

    Returns a list of dicts: {elem, ip, layer, S11, S22, S33, S12, S13, S23}.
    """
    with open(dat_path, "r", errors="ignore") as f:
        text = f.read()

    headers = list(_STRESS_HEADER_RE.finditer(text))
    matches = []
    for idx, m in enumerate(headers):
        set_name, time_str = m.group(1), m.group(2)
        if set_name.strip().lower() != elset_name.strip().lower():
            continue
        body_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        body_text = text[m.end():body_end]
        rows = _parse_stress_rows(body_text)
        if rows:
            matches.append((float(time_str), rows))

    if not matches:
        raise DatParseError(
            "No 'stresses (elem, integ.pnt.,sxx,syy,...) for set " + elset_name
            + "' block found in " + dat_path
            + " -- check that *El print, Elset=" + elset_name + " / S is in the .inp,"
            + " and if it is, the .dat header text may not match what this parser expects"
            + " (see the comment above read_ply_stresses)."
        )

    matches.sort(key=lambda t: t[0])
    return matches[-1][1]