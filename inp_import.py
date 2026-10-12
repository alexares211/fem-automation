"""Import a CalculiX/Abaqus-syntax .inp written by an external tool (PrePoMax)
and splice this app's own Material + composite layup + Puck output requests
into it, instead of building the whole .inp from scratch (step_mesher.py +
inp_writer.mesh_block/lager_*/load_* -- that path stays for STEP-drop users).

Workflow this supports: the user builds geometry, mesh, supports and loads
in PrePoMax (more capable there than this app's own click-to-place UI), runs
it once WITHOUT a material assigned (so ccx errors out on the undefined
material -- expected, not a bug), and uploads the resulting .inp here. This
module replaces ONLY the dangling/incomplete *Shell/*Solid section card for
the one Elset the user picked for the composite layup (plus that section's
own *Material, if nothing else still needs it) -- every other section,
material, node, element, set, *Boundary/*Cload/*Dload and the *Step
structure passes through completely untouched. This matters because a real
PrePoMax export can define several parts/sections in one .inp (confirmed
2026-10-13 against a real user model with 4 Sections and 2 Materials);
stripping every section/material in the file instead of just the target one
orphans the other parts' thickness and makes ccx fail with '*ERROR in
gen3delem: first thickness ... is zero' on whichever element happens to be
first in the file.

Parsing is a flat, generic Abaqus/CalculiX keyword-card split (keyword line
starts with a single '*', comment lines start with '**', a card's body is
every line up to the next keyword line) -- this grammar is the documented
CalculiX convention (ccx 2.22 user's manual, "7 INPUT DECK FORMAT"), not a
PrePoMax-specific guess. Confirmed 2026-10-13 against real PrePoMax exports:
(a) a material-less run does write the .inp up to the point of failure,
including the incomplete *Shell section card (elset name recovered from it
works as the default), and (b) PrePoMax's default shell mesh used plain S8
(full integration), which CalculiX's Composite shell section rejects --
see upgrade_s8_to_s8r() below, which fixes that without remeshing.
"""

import re

from inp_writer import InpWriteError, material_block, section_block, auto_nlgeom


class InpImportError(Exception):
    pass


def _is_keyword_line(line):
    s = line.strip()
    return s.startswith("*") and not s.startswith("**")


def _keyword_name(line):
    s = line.strip().lstrip("*")
    return s.split(",", 1)[0].strip().upper()


def _keyword_params(line):
    s = line.strip().lstrip("*")
    parts = s.split(",")[1:]
    params = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip().upper()] = v.strip()
    return params


def parse_cards(text):
    """Split .inp text into a flat list of {keyword, header, body} cards.
    keyword is None for any content before the first keyword line (so the
    split is lossless and round-trips via render_cards)."""
    cards = []
    cur = None
    for line in text.splitlines():
        if _is_keyword_line(line):
            if cur is not None:
                cards.append(cur)
            cur = {"keyword": _keyword_name(line), "header": line, "body": []}
        elif cur is not None:
            cur["body"].append(line)
        else:
            cards.append({"keyword": None, "header": line, "body": []})
    if cur is not None:
        cards.append(cur)
    return cards


def render_cards(cards):
    out = []
    for c in cards:
        out.append(c["header"])
        out.extend(c["body"])
    return "\n".join(out)


def find_set_names(cards, kind):
    """kind: 'NSET' or 'ELSET'. Picks up both standalone *Nset/*Elset cards
    and sets created inline via Nset=/Elset= on *Node/*Element."""
    kind = kind.upper()
    inline_source = "NODE" if kind == "NSET" else "ELEMENT"
    names, seen = [], set()
    for c in cards:
        name = None
        if c["keyword"] == kind:
            name = _keyword_params(c["header"]).get(kind)
        elif c["keyword"] == inline_source:
            name = _keyword_params(c["header"]).get(kind)
        if name and name.upper() not in seen:
            seen.add(name.upper())
            names.append(name)
    return names


# CalculiX only allows the Composite option on *Shell section for these two
# element types (ccx 2.22 manual, *SHELL SECTION) -- anything else fails
# deep inside ccx with "*ERROR reading *SHELL SECTION: ... can only be used
# for S8R or S6 shell elements", which is confirmed against a real PrePoMax
# export (2026-10-13): PrePoMax's default mesh used a different shell type,
# ccx rejected it. Checking this at import time instead of after a wasted
# solve.
_COMPOSITE_SHELL_TYPES = {"S8R", "S6"}


def _parse_element_types(cards):
    """Map element id (int) -> its *Element card's Type= value, scanning
    every *Element card's body rows (first comma-separated field is the
    element id; the rest is connectivity, ignored here)."""
    types = {}
    for c in cards:
        if c["keyword"] != "ELEMENT":
            continue
        etype = (_keyword_params(c["header"]).get("TYPE") or "").strip().upper()
        for line in c["body"]:
            line = line.strip().rstrip(",")
            if not line:
                continue
            first = line.split(",", 1)[0].strip()
            if first.isdigit():
                types[int(first)] = etype
    return types


def _parse_id_list(body_lines):
    """Parse the body of a plain (non-generate) *Elset/*Nset card: comma-
    separated integer ids, possibly spread over several lines."""
    ids = []
    for line in body_lines:
        for tok in line.strip().rstrip(",").split(","):
            tok = tok.strip()
            if tok.isdigit():
                ids.append(int(tok))
    return ids


def _resolve_elset_members(cards, elset_name):
    """Element ids belonging to elset_name: from *Element cards with an
    inline Elset= matching, and/or standalone *Elset cards (GENERATE
    start/end/step triples, or plain id lists). Returns None if the elset
    isn't found at all (caller decides what that means), rather than an
    empty set, so "not found" and "found but empty" aren't confused.
    """
    elset_name_u = elset_name.strip().upper()
    found = False
    members = set()
    for c in cards:
        if c["keyword"] == "ELEMENT":
            params = _keyword_params(c["header"])
            if (params.get("ELSET") or "").strip().upper() == elset_name_u:
                found = True
                for line in c["body"]:
                    first = line.strip().rstrip(",").split(",", 1)[0].strip()
                    if first.isdigit():
                        members.add(int(first))
        elif c["keyword"] == "ELSET":
            params = _keyword_params(c["header"])
            if (params.get("ELSET") or "").strip().upper() == elset_name_u:
                found = True
                is_generate = any(
                    p.strip().upper() == "GENERATE" for p in c["header"].split(",")[1:]
                )
                if is_generate:
                    for line in c["body"]:
                        parts = [p.strip() for p in line.strip().rstrip(",").split(",") if p.strip()]
                        if len(parts) >= 2 and all(p.lstrip("-").isdigit() for p in parts[:3] if p):
                            start = int(parts[0])
                            end = int(parts[1])
                            step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
                            members.update(range(start, end + 1, step))
                else:
                    members.update(_parse_id_list(c["body"]))
    return members if found else None


def elset_element_types(cards, elset_name):
    """Sorted list of distinct element Type= values found among elset_name's
    members. Empty list if the elset has no resolvable members (e.g. it's
    defined only by nesting other elsets by name -- not handled here)."""
    members = _resolve_elset_members(cards, elset_name)
    if not members:
        return []
    id_to_type = _parse_element_types(cards)
    return sorted({id_to_type[i] for i in members if i in id_to_type})


def upgrade_s8_to_s8r(cards, elset_name):
    """Rewrite Type=S8 to Type=S8R on every *Element card that contributes a
    member to elset_name. S8 and S8R are the SAME element (ccx 2.22 manual
    section 6.2.14, "Eight-node shell element (S8 and S8R)": identical node
    numbering/connectivity, the only difference is the solver's internal
    integration scheme -- full for S8, reduced for S8R) -- CalculiX's
    Composite shell section just requires the reduced-integration label, so
    this relabelling is lossless and needs no remesh. Returns a new cards
    list; the input is left untouched.
    """
    members = _resolve_elset_members(cards, elset_name) or set()
    out = []
    for c in cards:
        if c["keyword"] == "ELEMENT":
            params = _keyword_params(c["header"])
            etype = (params.get("TYPE") or "").strip().upper()
            card_ids = {int(line.strip().rstrip(",").split(",", 1)[0].strip())
                        for line in c["body"]
                        if line.strip().rstrip(",").split(",", 1)[0].strip().isdigit()}
            if etype == "S8" and (card_ids & members):
                c = dict(c)
                c["header"] = re.sub(r"(?i)type\s*=\s*S8\b", "Type=S8R", c["header"])
        out.append(c)
    return out


def split_preamble_and_step(cards):
    step_idx = next((i for i, c in enumerate(cards) if c["keyword"] == "STEP"), None)
    if step_idx is None:
        raise InpImportError(
            "No *Step card found in the uploaded .inp -- is this a complete "
            "PrePoMax export (mesh + supports + loads + a *Step), just without "
            "a material assigned?"
        )
    return cards[:step_idx], cards[step_idx:]


# Sub-cards that belong to a *Material definition -- DENSITY/EXPANSION are
# material sub-cards PrePoMax sometimes writes even for an otherwise
# undefined material; harmless to drop alongside it if present.
_MATERIAL_SUBCARD_KEYWORDS = {"ELASTIC", "DENSITY", "EXPANSION"}


def find_section_elsets(preamble_cards):
    """Every Elset= value referenced by a *Shell section/*Solid section card
    in the preamble, in document order, without removing anything."""
    out = []
    for c in preamble_cards:
        if c["keyword"] in ("SHELL SECTION", "SOLID SECTION"):
            elset = _keyword_params(c["header"]).get("ELSET")
            if elset:
                out.append(elset)
    return out


def find_incomplete_section_elsets(preamble_cards):
    """Elset= values from *Shell/*Solid section cards whose Material= name
    has no matching *Material card anywhere in the preamble -- i.e. the
    genuinely material-less part(s) a "run once with no material" PrePoMax
    export leaves behind. A model with several parts (confirmed real case:
    4 Sections, 2 Materials) can have other, fully-defined sections too;
    those must NOT be suggested as the layup target.
    """
    defined = {
        (_keyword_params(c["header"]).get("NAME") or "").strip().upper()
        for c in preamble_cards if c["keyword"] == "MATERIAL"
    }
    out = []
    for c in preamble_cards:
        if c["keyword"] in ("SHELL SECTION", "SOLID SECTION"):
            params = _keyword_params(c["header"])
            elset = params.get("ELSET")
            material = (params.get("MATERIAL") or "").strip().upper()
            if elset and material not in defined:
                out.append(elset)
    return out


def remove_section_for_elset(preamble_cards, target_elset):
    """Remove ONLY the *Shell section/*Solid section card for target_elset
    (plus its *Material, if that material isn't also used by some OTHER
    section left in place). Every other card -- other parts' sections,
    other materials, nodes, elements, sets, everything -- passes through
    untouched.

    This is deliberately narrow: a real PrePoMax export can define several
    parts/sections in one .inp (confirmed 2026-10-13 against a real user
    model with 4 Sections and 2 Materials) -- stripping every section/
    material card in the file, not just the one the user is replacing,
    orphans the other parts' thickness definitions and makes ccx fail with
    '*ERROR in gen3delem: first thickness ... is zero' on whichever element
    happens to be first in the file.

    Returns (cleaned_cards, removed_material_name_or_None).
    """
    target_u = (target_elset or "").strip().upper()
    removed_material = None
    section_cards, other_cards = [], []
    for c in preamble_cards:
        if c["keyword"] in ("SHELL SECTION", "SOLID SECTION"):
            section_cards.append(c)
        else:
            other_cards.append(c)

    kept_sections = []
    for c in section_cards:
        params = _keyword_params(c["header"])
        if (params.get("ELSET") or "").strip().upper() == target_u:
            removed_material = params.get("MATERIAL")
            continue  # drop only this one
        kept_sections.append(c)

    # Only drop the Material/Elastic/... cards for removed_material if no
    # remaining section still needs them.
    still_needed = removed_material and any(
        (_keyword_params(c["header"]).get("MATERIAL") or "").strip().upper()
        == removed_material.strip().upper()
        for c in kept_sections
    )

    cleaned = []
    in_removed_material = False
    for c in other_cards:
        if c["keyword"] == "MATERIAL":
            name = (_keyword_params(c["header"]).get("NAME") or "").strip()
            in_removed_material = bool(
                removed_material and not still_needed
                and name.upper() == removed_material.strip().upper()
            )
            if in_removed_material:
                continue
        elif c["keyword"] in _MATERIAL_SUBCARD_KEYWORDS and in_removed_material:
            continue
        else:
            in_removed_material = False
        cleaned.append(c)

    return cleaned + kept_sections, removed_material


def inspect_inp(text):
    """What the UI needs right after upload: available Nsets/Elsets, and the
    Elset the (now-stripped) section card was already pointing at, if any --
    used as the pre-selected default for 'which Elset gets the layup'.
    Raises InpImportError if there's no *Step (so this clearly isn't a
    PrePoMax export we can complete) -- does NOT require anything else to be
    "wrong" the way a material-less run is expected to be.
    """
    cards = parse_cards(text)
    preamble, step = split_preamble_and_step(cards)
    # Prefer a section whose material is genuinely undefined (the actual
    # "ran once with no material" part) over just the first section card in
    # the file -- a multi-part model can have other, fully-defined sections
    # that must not be suggested as the layup target.
    incomplete = find_incomplete_section_elsets(preamble)
    section_elsets = incomplete or find_section_elsets(preamble)
    elset_from_section = section_elsets[0] if section_elsets else None
    elsets = find_set_names(preamble, "ELSET")
    # Report types post-upgrade (S8 -> S8R) so the UI doesn't flag something
    # splice_layup() is about to fix anyway -- see upgrade_s8_to_s8r().
    elset_types = {
        name: elset_element_types(upgrade_s8_to_s8r(preamble, name), name)
        for name in elsets
    }
    return {
        "nsets": find_set_names(preamble, "NSET"),
        "elsets": elsets,
        "elset_types": elset_types,
        "suggested_elset": elset_from_section,
        "num_cards": len(cards),
    }


def splice_layup(base_text, elset_name, material_name, constants,
                  ply_angles, ply_thickness, symmetric=False,
                  rf_nset_name=None, nlgeom="auto", loads=None):
    """Combine an imported (material-less) .inp with this app's own
    Material + per-ply Orientation + composite Shell section + Puck/reaction
    output requests. Everything else from the import (nodes, elements,
    Nsets/Elsets, *Boundary, *Cload/*Dload, *Static) passes through
    untouched -- this only ever ADDS cards, except for removing the
    incomplete material/section cards the material-less PrePoMax run left
    behind.

    rf_nset_name: an existing Nset from the import to report reaction force
    for (adds its own *Node print, RF/U card); pass None to skip reaction
    force reporting.
    """
    cards = parse_cards(base_text)
    preamble, step = split_preamble_and_step(cards)

    section_elsets = find_section_elsets(preamble)
    target_elset = (elset_name or (section_elsets[0] if section_elsets else "")).strip()
    if not target_elset:
        raise InpWriteError(
            "Could not tell which element set the composite layup applies to "
            "-- pick one of the imported .inp's element sets."
        )

    preamble, _ = remove_section_for_elset(preamble, target_elset)

    # S8 -> S8R is a lossless relabel (same element, same connectivity, see
    # upgrade_s8_to_s8r's docstring), so it's fixed here rather than
    # rejected -- do this before the compatibility check below.
    preamble = upgrade_s8_to_s8r(preamble, target_elset)

    found_types = elset_element_types(preamble, target_elset)
    bad_types = sorted(set(found_types) - _COMPOSITE_SHELL_TYPES)
    if bad_types:
        raise InpWriteError(
            f"Elset {target_elset!r} contains {', '.join(bad_types)} elements. "
            "CalculiX's Composite shell section only works with S8R or S6 shell "
            "elements -- remesh this part in PrePoMax using S8R (quad) or S6 "
            "(triangle) elements, re-export, and re-upload."
        )

    if nlgeom == "auto":
        nlgeom = auto_nlgeom(loads, ply_thickness, ply_angles, symmetric)

    mat_text = material_block(material_name, constants)
    sec_text = section_block(material_name, ply_angles, ply_thickness, symmetric, elset=target_elset)
    preamble_text = render_cards(preamble)
    combined_preamble = preamble_text + "\n" + mat_text + "\n" + sec_text

    step_cards = [dict(c) for c in step]  # shallow copy, header gets rewritten below
    if nlgeom and "NLGEOM" not in step_cards[0]["header"].upper():
        step_cards[0]["header"] = step_cards[0]["header"].rstrip() + ", Nlgeom"

    extra_cards = [{
        "keyword": "EL PRINT",
        "header": f"*El print, Elset={target_elset}",
        "body": ["S"],
    }]
    if rf_nset_name:
        extra_cards.append({
            "keyword": "NODE PRINT",
            "header": f"*Node print, Nset={rf_nset_name}, Global=Yes",
            "body": ["RF, U"],
        })

    end_idx = next((i for i, c in enumerate(step_cards) if c["keyword"] == "END STEP"), None)
    if end_idx is None:
        raise InpImportError("*Step is missing a matching *End Step card.")
    step_cards = step_cards[:end_idx] + extra_cards + step_cards[end_idx:]

    return combined_preamble + "\n" + render_cards(step_cards)
