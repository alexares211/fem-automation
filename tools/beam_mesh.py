"""Create only a structured quadratic SHELL MESH to import into PrePoMax.

The clamp and tip force are subsequently defined in PrePoMax and exported
with its own CalculiX writer. Dimensions: mm; force: N; modulus: MPa.
Use: python tools/beam_mesh.py --output <mesh.inp> [--nx 20 --ny 4]
"""
import argparse
from pathlib import Path


def mesh_text(nx=20, ny=4, length=100.0, width=10.0):
    nodes, lookup = {}, {}
    for j in range(2*ny+1):
        for i in range(2*nx+1):
            if i % 2 and j % 2:
                continue  # eight-node serendipity quads have no center node
            node = len(nodes)+1
            lookup[i, j] = node
            nodes[node] = (length*i/(2*nx), width*j/(2*ny)-width/2, 0.0)
    lines = ["*HEADING", "100 x 10 mm cantilever shell mesh; configure BCs/loads in PrePoMax", "*NODE"]
    lines += [f"{node}, {x:g}, {y:g}, {z:g}" for node, (x, y, z) in nodes.items()]
    lines += ["*ELEMENT, TYPE=S8R, ELSET=BEAM"]
    for j in range(ny):
        for i in range(nx):
            a, b = 2*i, 2*j
            conn = [(a,b), (a+2,b), (a+2,b+2), (a,b+2),
                    (a+1,b), (a+2,b+1), (a+1,b+2), (a,b+1)]
            lines.append(f"{j*nx+i+1}, " + ", ".join(str(lookup[p]) for p in conn))
    for name, indices in (("FIXED_EDGE", [lookup[0,j] for j in range(2*ny+1)]),
                          ("TIP_EDGE", [lookup[2*nx,j] for j in range(2*ny+1)])):
        lines.extend([f"*NSET, NSET={name}", ", ".join(map(str, indices))])
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=4)
    args = parser.parse_args()
    if args.nx < 1 or args.ny < 1:
        parser.error("Mesh divisions must be positive.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(mesh_text(args.nx, args.ny), encoding="utf-8")
    print(output)
