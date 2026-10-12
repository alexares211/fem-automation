"""Import a CalculiX/Abaqus-syntax .inp written by an external tool (PrePoMax)
and splice this app's own Material + composite layup + Puck output requests
into it, instead of building the whole .inp from scratch (step_mesher.py +
inp_writer.mesh_block/lager_*/load_* -- that path stays for STEP-drop users).

Workflow this supports: the user builds geometry, mesh, supports and loads
in PrePoMax (more capable there than this app's own click-to-place UI), runs
it once WITHOUT a material assigned (so ccx errors out on the undefined
material -- expected, not a bug), and uploads the resulting .inp here. This
module strips the resulting dangling/incomplete *Material and *Shell/*Solid
section cards (the only thing missing) and leaves everything else -- nodes,
elements, *Nset/*Elset, *Boundary, *Cload/*Dload, the *Step structure --
untouched, so the app only ever adds composite-specific content on top of
PrePoMax's own geometry/BC/load work.

Parsing is a flat, generic Abaqus/CalculiX keyword-card split (keyword line
starts with a single '*', comment lines start with '**', a card's body is
every line up to the next keyword line) -- this grammar is the documented
CalculiX convention (ccx 2.22 user's manual, "7 INPUT DECK FORMAT"), not a
PrePoMax-specific guess. What IS still an assumption, not yet verified
against a real PrePoMax export, is: (a) that a material-less PrePoMax run
still writes the .inp up to the point of failure, including an incomplete
*Shell/*Solid section card referencing an ELSET, and (b) PrePoMax's exact
set-naming conventions. Both should be checked against a real file before
trusting this for anything safety-relevant.
"""

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


def split_preamble_and_step(cards):
    step_idx = next((i for i, c in enumerate(cards) if c["keyword"] == "STEP"), None)
    if step_idx is None:
        raise InpImportError(
            "No *Step card found in the uploaded .inp -- is this a complete "
            "PrePoMax export (mesh + supports + loads + a *Step), just without "
            "a material assigned?"
        )
    return cards[:step_idx], cards[step_idx:]


# Cards that belong to material/section definition -- dropped so this app's
# own material_block()/section_block() can replace them. DENSITY/EXPANSION
# are material sub-cards PrePoMax sometimes writes even for an otherwise
# undefined material; harmless to drop if present, no-op if not.
_MATERIAL_SECTION_KEYWORDS = {
    "MATERIAL", "ELASTIC", "DENSITY", "EXPANSION",
    "SHELL SECTION", "SOLID SECTION",
}


def strip_material_and_section(preamble_cards):
    """Remove any *Material/*Elastic/*Shell section/*Solid section cards.
    Returns (cleaned_cards, elset_name_from_removed_section_or_None) -- the
    Elset= value off a removed *Shell/*Solid section card is what the
    composite layup should target, so it's surfaced even though the card
    itself is dropped.
    """
    cleaned = []
    elset_found = None
    for c in preamble_cards:
        if c["keyword"] in ("SHELL SECTION", "SOLID SECTION"):
            elset_found = elset_found or _keyword_params(c["header"]).get("ELSET")
            continue
        if c["keyword"] in _MATERIAL_SECTION_KEYWORDS:
            continue
        cleaned.append(c)
    return cleaned, elset_found


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
    _, elset_from_section = strip_material_and_section(preamble)
    return {
        "nsets": find_set_names(preamble, "NSET"),
        "elsets": find_set_names(preamble, "ELSET"),
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
    preamble, found_elset = strip_material_and_section(preamble)

    target_elset = (elset_name or found_elset or "").strip()
    if not target_elset:
        raise InpWriteError(
            "Could not tell which element set the composite layup applies to "
            "-- pick one of the imported .inp's element sets."
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
