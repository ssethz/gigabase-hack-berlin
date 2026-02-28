#!/usr/bin/env python3
"""Prepare 2XEE NI3C Mut5 DARPin scaffold for partial diffusion with RFdiffusion.

Downloads the thermally-stabilized DARPin structure (Plückthun lab, 2010),
auto-selects the best chain by B-factor, renumbers residues, orients the
concave binding face toward the 15-PGDH C-terminal seam zone, and outputs
the inpaint_str for fixing helices while diffusing loops.
"""

import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.vectors import rotmat, Vector
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent

# 2XEE NI3C Mut5 helix ranges (from PDB HELIX records, consensus across chains)
# N-cap through IR3 identical to 2XEH; C-cap helix 9 ends at 156 (not 157).
HELIX_RANGES_ORIG = [
    (13, 25), (26, 36),    # N-cap helix pair
    (49, 58), (59, 69),    # Internal repeat 1 helix pair
    (82, 91), (92, 102),   # Internal repeat 2 helix pair
    (115, 124), (125, 135),# Internal repeat 3 helix pair
    (148, 156), (158, 169),# C-cap helix pair (156 for 2XEE Mut5)
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


def select_best_chain(structure) -> str:
    """Select the chain with the lowest mean CA B-factor (best electron density)."""
    chain_bfactors: dict[str, list[float]] = {}
    for chain in structure[0].get_chains():
        bfactors = []
        for res in chain.get_residues():
            if res.id[0] == " " and "CA" in res:
                bfactors.append(res["CA"].get_bfactor())
        if bfactors:
            chain_bfactors[chain.id] = bfactors

    best_chain = min(chain_bfactors, key=lambda c: np.mean(chain_bfactors[c]))
    for cid, bfs in sorted(chain_bfactors.items()):
        print(f"  Chain {cid}: mean CA B-factor = {np.mean(bfs):.1f}")
    print(f"  Selected chain {best_chain} (lowest B-factor)")
    return best_chain


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
                                surface_clearance=8.0):
    """Position and orient DARPin so its concave face points at the target site.

    Uses the outward direction from the target bulk centroid through the hotspot
    surface to ensure the DARPin is placed OUTSIDE the target.
    """
    # All target CA coords and hotspot CA coords
    all_target_ca = get_residue_coords(target_struct, target_chain_id,
                                       list(range(0, 300)))
    hotspot_coords = get_residue_coords(target_struct, target_chain_id,
                                        target_residues)

    bulk_centroid = all_target_ca.mean(axis=0)
    hotspot_centroid = hotspot_coords.mean(axis=0)

    # Outward direction: from bulk center through hotspot surface
    outward = hotspot_centroid - bulk_centroid
    outward = outward / np.linalg.norm(outward)

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

    # DARPin concave face direction
    concave_dir = loop_centroid - darpin_centroid
    concave_dir = concave_dir / np.linalg.norm(concave_dir)

    # We want concave face pointing INWARD (toward target) = opposite of outward
    desired_dir = -outward

    # Rotation matrix to align concave_dir with desired_dir
    v = np.cross(concave_dir, desired_dir)
    c = np.dot(concave_dir, desired_dir)
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    # Compute DARPin radius (max distance from centroid to any CA)
    darpin_radius = np.max(np.linalg.norm(all_ca - darpin_centroid, axis=1))

    # Place DARPin centroid along outward direction, cleared from hotspot surface
    placement = hotspot_centroid + outward * (darpin_radius + surface_clearance)

    for atom in darpin_ch.get_atoms():
        coord = atom.get_vector().get_array()
        rotated = R @ (coord - darpin_centroid) + darpin_centroid
        translated = rotated + (placement - darpin_centroid)
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
    """Build the inpaint_str that fixes target + DARPin helices, leaves loops free.

    RFdiffusion's inpaint_str uses '/' as separator (not '/0 ' which is contigs-only).
    Listed residues get their original structure restored at each denoising step (fixed).
    Unlisted residues (DARPin loops) undergo normal denoising (redesigned).
    """
    parts = [target_res_range]
    for start, end in helix_ranges_renum:
        parts.append(f"B{start}-{end}")
    return "[" + "/".join(parts) + "]"


def build_provide_seq(helix_ranges_renum, target_len=266):
    """Build provide_seq string to keep DARPin helix sequences during diffusion.

    provide_seq is 0-indexed over the full output (target + binder concatenated).
    Chain B starts at index target_len.
    """
    ranges = []
    for start, end in helix_ranges_renum:
        global_start = target_len + (start - 1)  # 1-indexed to 0-indexed
        global_end = target_len + (end - 1)
        ranges.append(f"{global_start}-{global_end}")
    return "[" + ",".join(ranges) + "]"


def select_hotspots(config_path: Path, top_n: int = 12, focus_ranges=None):
    """Select hotspot residues by contact count from the pipeline config.

    Prioritizes residues within focus_ranges (the friend-validated zones),
    then fills remaining slots from the broader interface.
    """
    import json

    with open(config_path) as f:
        config = json.load(f)

    details = config["interface"]["residue_details"]

    if focus_ranges is None:
        focus_ranges = [(101, 121), (148, 179)]

    def in_focus(rnum):
        return any(lo <= rnum <= hi for lo, hi in focus_ranges)

    scored = []
    for rnum_str, info in details.items():
        rnum = int(rnum_str)
        scored.append((rnum, info["n_contacts"], in_focus(rnum)))

    focused = sorted([s for s in scored if s[2]], key=lambda x: -x[1])
    other = sorted([s for s in scored if not s[2]], key=lambda x: -x[1])

    selected = [r[0] for r in focused[:top_n]]
    remaining = top_n - len(selected)
    if remaining > 0:
        selected.extend(r[0] for r in other[:remaining])

    return sorted(selected)


def write_rfdiff_params(params_path: Path, inpaint_str: str, hotspots: list[int],
                        darpin_len: int, contigs: str, provide_seq: str,
                        helix_ranges: list[tuple[int, int]],
                        loop_ranges: list[tuple[int, int]]):
    """Write RFdiffusion parameters to a sourceable shell file."""
    hotspot_str = "[" + ",".join(f"A{r}" for r in hotspots) + "]"
    helix_json = json.dumps(helix_ranges)
    loop_json = json.dumps(loop_ranges)
    with open(params_path, "w") as f:
        f.write("# Auto-generated by prepare_darpin.py -- do not edit manually\n")
        f.write(f'DARPIN_LEN={darpin_len}\n')
        f.write(f'CONTIGS="{contigs}"\n')
        f.write(f'INPAINT_STR="{inpaint_str}"\n')
        f.write(f'PROVIDE_SEQ="{provide_seq}"\n')
        f.write(f'HOTSPOTS="{hotspot_str}"\n')
        f.write(f"HELIX_RANGES='{helix_json}'\n")
        f.write(f"LOOP_RANGES='{loop_json}'\n")
    print(f"Wrote RFdiffusion params: {params_path}")


def main():
    darpin_pdb_id = "2XEE"

    targets_dir = REPO_ROOT / "targets"
    targets_dir.mkdir(exist_ok=True)

    darpin_raw_path = targets_dir / f"{darpin_pdb_id}.pdb"
    if not darpin_raw_path.exists():
        print(f"Downloading DARPin structure {darpin_pdb_id}...")
        download_pdb(darpin_pdb_id, darpin_raw_path)

    parser = PDBParser(QUIET=True)
    darpin_struct = parser.get_structure("darpin", str(darpin_raw_path))

    print(f"Selecting best chain from {darpin_pdb_id} by B-factor...")
    darpin_chain = select_best_chain(darpin_struct)

    # Save clean single-chain DARPin
    darpin_clean_path = targets_dir / "darpin_template.pdb"
    io = PDBIO()
    io.set_structure(darpin_struct)
    io.save(str(darpin_clean_path), ChainAndProteinSelect(darpin_chain))
    print(f"Saved clean DARPin: {darpin_clean_path}")

    darpin_struct = parser.get_structure("darpin", str(darpin_clean_path))
    darpin_ch_id = list(darpin_struct[0].get_chains())[0].id
    n_darpin_res = sum(1 for r in darpin_struct[0][darpin_ch_id].get_residues() if r.id[0] == " ")
    print(f"DARPin {darpin_pdb_id} NI3C Mut5: {n_darpin_res} residues, chain {darpin_ch_id}")

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

    # Interface residues for positioning (hotspot patches A148-179, A101-121)
    cterminal_seam = list(range(101, 122)) + list(range(148, 180))

    print(f"\nOrienting DARPin concave face toward alpha-9 helix interface...")
    darpin_struct = orient_darpin_toward_target(
        darpin_struct, darpin_ch_id,
        target_struct, "A",
        target_residues=cterminal_seam,
        loop_ranges_renum=loop_ranges_renum,
        surface_clearance=8.0,
    )

    # Create complex PDB
    complex_path = targets_dir / "darpin_on_15pgdh.pdb"
    darpin_len = create_complex_pdb(
        target_struct, darpin_struct, complex_path,
        target_chain="A", darpin_chain=darpin_ch_id,
    )
    print(f"\nSaved complex: {complex_path}")
    print(f"  Chain A: 15-PGDH target (266 residues)")
    print(f"  Chain B: DARPin {darpin_pdb_id} NI3C Mut5 ({darpin_len} residues)")

    # Build RFdiffusion parameters
    inpaint_str = build_inpaint_str(helix_ranges_renum, darpin_len)
    provide_seq = build_provide_seq(helix_ranges_renum, target_len=266)
    contigs = f"[A0-265/0 B1-{darpin_len}]"

    # Select hotspots: top 12 by contact count from the two target patches
    config_path = REPO_ROOT / "configs" / "pipeline_config.json"
    if config_path.exists():
        hotspots = select_hotspots(config_path, top_n=12,
                                   focus_ranges=[(101, 121), (148, 179)])
    else:
        hotspots = [105, 113, 116, 149, 153, 156, 160, 161, 164, 168, 171, 172]

    print(f"\n=== RFdiffusion parameters ===")
    print(f"  contigs:      {contigs}")
    print(f"  inpaint_str:  {inpaint_str}")
    print(f"  provide_seq:  {provide_seq}")
    print(f"  hotspot_res:  [{','.join(f'A{r}' for r in hotspots)}]")

    params_path = targets_dir / "rfdiff_params.sh"
    write_rfdiff_params(params_path, inpaint_str, hotspots, darpin_len, contigs,
                        provide_seq, helix_ranges_renum, loop_ranges_renum)


if __name__ == "__main__":
    main()
