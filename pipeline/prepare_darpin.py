#!/usr/bin/env python3
"""Prepare 2XEH NI3C Mut6 DARPin scaffold for partial diffusion with RFdiffusion.

Downloads the thermally-stabilized DARPin structure (Plückthun lab, 2010),
extracts chain B, renumbers residues, orients the concave binding face
toward the 15-PGDH C-terminal seam zone, and outputs the inpaint_str
for fixing helices while diffusing loops.
"""

import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.vectors import rotmat, Vector
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent

# 2XEH chain B helix ranges (from PDB HELIX records)
# Loops between helix pairs are the concave binding surface
HELIX_RANGES_ORIG = [
    (13, 25), (26, 36),    # N-cap helix pair
    (49, 58), (59, 69),    # Internal repeat 1 helix pair
    (82, 91), (92, 102),   # Internal repeat 2 helix pair
    (115, 124), (125, 135),# Internal repeat 3 helix pair
    (148, 157), (158, 169),# C-cap helix pair
]

LOOP_RANGES_ORIG = [
    (37, 48),   # Loop between N-cap and IR1
    (70, 81),   # Loop between IR1 and IR2
    (103, 114), # Loop between IR2 and IR3
    (136, 147), # Loop between IR3 and C-cap
]


class ChainAndProteinSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "


def download_pdb(pdb_id: str, output_path: Path):
    import urllib.request
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    urllib.request.urlretrieve(url, str(output_path))


def get_residue_coords(structure, chain_id, residue_list, atom="CA"):
    chain = structure[0][chain_id]
    coords = []
    for res in chain.get_residues():
        if res.id[0] == " " and res.id[1] in residue_list and atom in res:
            coords.append(res[atom].get_vector().get_array())
    return np.array(coords)


def orient_darpin_toward_target(darpin_struct, darpin_chain_id,
                                target_struct, target_chain_id,
                                target_residues, loop_ranges_renum,
                                offset=16.0):
    """Position and orient DARPin so its concave face points at the target site.

    1. Compute target site centroid and outward normal
    2. Compute DARPin centroid and concave-face direction (from centroid toward loops)
    3. Rotate DARPin so concave face points toward target
    4. Translate DARPin to offset distance from target site
    """
    target_coords = get_residue_coords(target_struct, target_chain_id, target_residues)
    target_centroid = target_coords.mean(axis=0)

    target_centered = target_coords - target_centroid
    _, _, vt = np.linalg.svd(target_centered)
    target_normal = vt[-1]

    darpin_model = darpin_struct[0]
    darpin_ch = darpin_model[darpin_chain_id]

    all_ca = []
    loop_ca = []
    for res in darpin_ch.get_residues():
        if res.id[0] != " " or "CA" not in res:
            continue
        coord = res["CA"].get_vector().get_array()
        all_ca.append(coord)
        rnum = res.id[1]
        for ls, le in loop_ranges_renum:
            if ls <= rnum <= le:
                loop_ca.append(coord)
                break

    all_ca = np.array(all_ca)
    loop_ca = np.array(loop_ca)
    darpin_centroid = all_ca.mean(axis=0)
    loop_centroid = loop_ca.mean(axis=0)

    concave_dir = loop_centroid - darpin_centroid
    concave_dir = concave_dir / np.linalg.norm(concave_dir)

    desired_dir = -target_normal
    desired_dir = desired_dir / np.linalg.norm(desired_dir)

    # Rotation: align concave_dir with desired_dir
    v = np.cross(concave_dir, desired_dir)
    c = np.dot(concave_dir, desired_dir)
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            R = np.eye(3)
        else:
            perp = np.array([1, 0, 0]) if abs(concave_dir[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(concave_dir, perp)
            axis = axis / np.linalg.norm(axis)
            R = -np.eye(3) + 2 * np.outer(axis, axis)
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    # Apply rotation around DARPin centroid, then translate
    target_position = target_centroid + target_normal * offset
    for atom in darpin_ch.get_atoms():
        coord = atom.get_vector().get_array()
        rotated = R @ (coord - darpin_centroid) + darpin_centroid
        translated = rotated + (target_position - darpin_centroid)
        atom.set_coord(translated)

    return darpin_struct


def create_complex_pdb(target_struct, darpin_struct, output_path,
                       target_chain="A", darpin_chain="B"):
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Residue import Residue

    new_struct = Structure("complex")
    new_model = Model(0)
    new_struct.add(new_model)

    target_ch = target_struct[0][target_chain]
    new_chain_a = Chain("A")
    for res in target_ch.get_residues():
        if res.id[0] == " ":
            new_res = Residue(res.id, res.resname, res.segid)
            for atom in res.get_atoms():
                new_res.add(atom.copy())
            new_chain_a.add(new_res)
    new_model.add(new_chain_a)

    darpin_ch = darpin_struct[0][darpin_chain]
    new_chain_b = Chain("B")
    resnum = 1
    for res in darpin_ch.get_residues():
        if res.id[0] == " ":
            new_res = Residue((" ", resnum, " "), res.resname, "")
            for atom in res.get_atoms():
                new_res.add(atom.copy())
            new_chain_b.add(new_res)
            resnum += 1
    new_model.add(new_chain_b)

    io = PDBIO()
    io.set_structure(new_struct)
    io.save(str(output_path))
    return resnum - 1


def build_renumber_map(structure, chain_id):
    """Build mapping from original residue numbers to 1-based sequential."""
    chain = structure[0][chain_id]
    orig_to_new = {}
    new_num = 1
    for res in chain.get_residues():
        if res.id[0] == " ":
            orig_to_new[res.id[1]] = new_num
            new_num += 1
    return orig_to_new


def convert_ranges(ranges_orig, orig_to_new):
    """Convert original residue ranges to renumbered ranges."""
    converted = []
    for start, end in ranges_orig:
        new_start = orig_to_new.get(start)
        new_end = orig_to_new.get(end)
        if new_start is not None and new_end is not None:
            converted.append((new_start, new_end))
    return converted


def build_inpaint_str(helix_ranges_renum, darpin_len, target_res_range="A0-265"):
    """Build the inpaint_str that fixes target + DARPin helices, leaves loops free."""
    parts = [target_res_range]
    for start, end in helix_ranges_renum:
        parts.append(f"B{start}-{end}")
    return "[" + "/0 ".join(parts) + "]"


def main():
    darpin_pdb_id = "2XEH"
    darpin_chain = "B"

    targets_dir = REPO_ROOT / "targets"
    targets_dir.mkdir(exist_ok=True)

    darpin_raw_path = targets_dir / f"{darpin_pdb_id}.pdb"
    if not darpin_raw_path.exists():
        print(f"Downloading DARPin structure {darpin_pdb_id}...")
        download_pdb(darpin_pdb_id, darpin_raw_path)

    parser = PDBParser(QUIET=True)
    darpin_struct = parser.get_structure("darpin", str(darpin_raw_path))

    # Save clean single-chain DARPin
    darpin_clean_path = targets_dir / "darpin_template.pdb"
    io = PDBIO()
    io.set_structure(darpin_struct)
    io.save(str(darpin_clean_path), ChainAndProteinSelect(darpin_chain))
    print(f"Saved clean DARPin: {darpin_clean_path}")

    darpin_struct = parser.get_structure("darpin", str(darpin_clean_path))
    darpin_ch_id = list(darpin_struct[0].get_chains())[0].id
    n_darpin_res = sum(1 for r in darpin_struct[0][darpin_ch_id].get_residues() if r.id[0] == " ")
    print(f"DARPin 2XEH NI3C Mut6: {n_darpin_res} residues, chain {darpin_ch_id}")

    # Build renumbering map
    orig_to_new = build_renumber_map(darpin_struct, darpin_ch_id)
    helix_ranges_renum = convert_ranges(HELIX_RANGES_ORIG, orig_to_new)
    loop_ranges_renum = convert_ranges(LOOP_RANGES_ORIG, orig_to_new)

    print(f"\nHelix ranges (renumbered 1-{n_darpin_res}):")
    for s, e in helix_ranges_renum:
        print(f"  B{s}-{e}")
    print(f"Loop ranges (binding surface):")
    for s, e in loop_ranges_renum:
        print(f"  B{s}-{e}")

    # Load target
    target_path = targets_dir / "15pgdh_chainA.pdb"
    if not target_path.exists():
        print(f"ERROR: Target not found: {target_path}")
        print("Run pipeline/prepare_target.py first.")
        sys.exit(1)

    target_struct = parser.get_structure("target", str(target_path))

    # Alpha-9 helix interface residues for positioning (hotspot region)
    cterminal_seam = [143, 144, 149, 156, 161, 164, 165]

    print(f"\nOrienting DARPin concave face toward alpha-9 helix interface...")
    darpin_struct = orient_darpin_toward_target(
        darpin_struct, darpin_ch_id,
        target_struct, "A",
        target_residues=cterminal_seam,
        loop_ranges_renum=loop_ranges_renum,
        offset=16.0,
    )

    # Create complex PDB
    complex_path = targets_dir / "darpin_on_15pgdh.pdb"
    darpin_len = create_complex_pdb(
        target_struct, darpin_struct, complex_path,
        target_chain="A", darpin_chain=darpin_ch_id,
    )
    print(f"\nSaved complex: {complex_path}")
    print(f"  Chain A: 15-PGDH target (266 residues)")
    print(f"  Chain B: DARPin 2XEH NI3C Mut6 ({darpin_len} residues)")

    # Build inpaint_str for RFdiffusion
    inpaint_str = build_inpaint_str(helix_ranges_renum, darpin_len)
    print(f"\n=== RFdiffusion parameters ===")
    print(f"  contigs:     [A0-265/0 B1-{darpin_len}]")
    print(f"  inpaint_str: {inpaint_str}")
    print(f"  hotspot_res: [A143,A144,A165]")
    print(f"  partial_T:   10")


if __name__ == "__main__":
    main()
