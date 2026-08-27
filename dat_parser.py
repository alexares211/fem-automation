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

    matches = []
    for m in _HEADER_RE.finditer(text):
        set_name, time_str = m.group(1), m.group(2)
        if set_name.strip().lower() != nset_name.strip().lower():
            continue
        body_text = text[m.end():m.end() + 4000]
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