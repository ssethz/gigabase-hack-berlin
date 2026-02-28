#!/usr/bin/env python3
"""Prepare ProteinMPNN input JSONLs from RFdiffusion backbone PDBs or mmCIF files.

Reads all PDB/CIF files from a backbone directory, identifies which residues
are designable vs fixed, and writes JSONL files for ProteinMPNN.

For CIF files with an `is_motif_atom_with_fixed_seq` column, fixed positions
are determined automatically from that field.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _structure_to_mpnn_dict(name: str, model) -> dict:
    """Extract chain sequences and backbone coordinates from a BioPython model."""
    chains = {}
    for chain in model.get_chains():
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        if not residues:
            continue

        coords = {"N": [], "CA": [], "C": [], "O": []}
        seq = []

        for res in residues:
            seq.append(THREE_TO_ONE.get(res.resname, "X"))
            for atom_name in ["N", "CA", "C", "O"]:
                if atom_name in res:
                    coords[atom_name].append(res[atom_name].get_vector().get_array().tolist())
                else:
                    coords[atom_name].append([0.0, 0.0, 0.0])

        chains[chain.id] = {
            "seq": "".join(seq),
            "coords": coords,
            "n_residues": len(seq),
        }

    return {"name": name, "chains": chains}


def pdb_to_mpnn_dict(pdb_path: Path) -> dict:
    """Convert a PDB file to ProteinMPNN input dict format."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    return _structure_to_mpnn_dict(pdb_path.stem, structure[0])


def cif_to_mpnn_dict(cif_path: Path) -> dict:
    """Convert an mmCIF file to ProteinMPNN input dict format."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    return _structure_to_mpnn_dict(cif_path.stem, structure[0])


def parse_cif_fixed_positions(cif_path: Path) -> dict[str, list[int]] | None:
    """Read is_motif_atom_with_fixed_seq from a CIF file.

    Returns a dict mapping chain_id -> list of 1-based residue positions that
    should be FIXED (i.e. not designed). Returns None if the field is absent.
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    cif_dict = MMCIF2Dict(str(cif_path))

    fixed_col = cif_dict.get("_atom_site.is_motif_atom_with_fixed_seq")
    if fixed_col is None:
        return None

    chain_ids = cif_dict["_atom_site.label_asym_id"]
    seq_ids = cif_dict["_atom_site.label_seq_id"]
    atom_ids = cif_dict["_atom_site.label_atom_id"]

    fixed_by_chain: dict[str, set[int]] = {}
    for chain_id, seq_id, atom_id, is_fixed in zip(chain_ids, seq_ids, atom_ids, fixed_col):
        if atom_id != "CA":
            continue
        if chain_id not in fixed_by_chain:
            fixed_by_chain[chain_id] = set()
        if is_fixed == "True":
            fixed_by_chain[chain_id].add(int(seq_id))

    return {cid: sorted(positions) for cid, positions in fixed_by_chain.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ProteinMPNN inputs from backbone PDBs or CIF files."
    )
    parser.add_argument("--backbone_dir", required=True, help="Directory with backbone PDB/CIF files")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--target_chain", default="A", help="Target chain to fix (default: A, ignored for CIF with fixed_seq field)")
    parser.add_argument("--fix_binder_helices", action="store_true",
                        help="Fix DARPin helix positions in binder chain (design loops only)")
    args = parser.parse_args()

    params_path = REPO_ROOT / "targets" / "rfdiff_params.sh"
    DARPIN_HELIX_POSITIONS = set()
    if args.fix_binder_helices:
        if params_path.exists():
            with open(params_path) as f:
                for line in f:
                    m = re.match(r"HELIX_RANGES='(.+)'", line.strip())
                    if m:
                        for s, e in json.loads(m.group(1)):
                            DARPIN_HELIX_POSITIONS.update(range(s, e + 1))
                        break
        if not DARPIN_HELIX_POSITIONS:
            print("WARNING: Could not read HELIX_RANGES from rfdiff_params.sh, using fallback")
            for s, e in [(1,13),(14,24),(37,46),(47,57),(70,79),(80,90),
                         (103,112),(113,123),(136,145),(146,157)]:
                DARPIN_HELIX_POSITIONS.update(range(s, e + 1))

    backbone_dir = Path(args.backbone_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_files = sorted(backbone_dir.glob("*.pdb")) + sorted(backbone_dir.glob("*.cif"))
    if not structure_files:
        print(f"No PDB or CIF files found in {backbone_dir}")
        return

    print(f"Processing {len(structure_files)} backbone files...")

    chains_jsonl = output_dir / "chains.jsonl"
    chain_ids_jsonl = output_dir / "chain_ids.jsonl"
    fixed_pos_jsonl = output_dir / "fixed_positions.jsonl"

    chain_id_dict = {}
    fixed_pos_dict = {}
    chain_info_all = {}

    with open(chains_jsonl, "w") as f_chains:
        for struct_path in structure_files:
            is_cif = struct_path.suffix.lower() == ".cif"

            try:
                if is_cif:
                    data = cif_to_mpnn_dict(struct_path)
                    cif_fixed = parse_cif_fixed_positions(struct_path)
                else:
                    data = pdb_to_mpnn_dict(struct_path)
                    cif_fixed = None
            except Exception as e:
                print(f"  Skipping {struct_path.name}: {e}")
                continue

            if not data["chains"]:
                continue

            name = data["name"]

            entry = {"name": name}
            concat_seq = ""
            for cid, cdata in data["chains"].items():
                concat_seq += cdata["seq"]
                entry[f"seq_chain_{cid}"] = cdata["seq"]
                coords_dict = {
                    f"N_chain_{cid}": cdata["coords"]["N"],
                    f"CA_chain_{cid}": cdata["coords"]["CA"],
                    f"C_chain_{cid}": cdata["coords"]["C"],
                    f"O_chain_{cid}": cdata["coords"]["O"],
                }
                entry[f"coords_chain_{cid}"] = coords_dict

            entry["seq"] = concat_seq
            entry["num_of_chains"] = len(data["chains"])
            f_chains.write(json.dumps(entry) + "\n")

            all_chains = list(data["chains"].keys())

            if cif_fixed is not None:
                # CIF with is_motif_atom_with_fixed_seq: use it directly
                fp = {}
                designed = []
                fixed = []
                for cid, cdata in data["chains"].items():
                    fixed_positions = cif_fixed.get(cid, [])
                    fp[cid] = fixed_positions
                    n_res = cdata["n_residues"]
                    if len(fixed_positions) == n_res:
                        fixed.append(cid)
                    elif len(fixed_positions) == 0:
                        designed.append(cid)
                    else:
                        designed.append(cid)
                print(f"  {name} (CIF): chains={all_chains}, "
                      f"designed={designed}, fixed={fixed}, "
                      f"sizes={[c['n_residues'] for c in data['chains'].values()]}")
            else:
                # PDB fallback: use target_chain / helix logic
                designed = [c for c in all_chains if c != args.target_chain]
                fixed = [c for c in all_chains if c == args.target_chain]
                fp = {}
                for cid, cdata in data["chains"].items():
                    if cid == args.target_chain:
                        fp[cid] = list(range(1, cdata["n_residues"] + 1))
                    elif args.fix_binder_helices:
                        fp[cid] = sorted(
                            p for p in DARPIN_HELIX_POSITIONS
                            if p <= cdata["n_residues"]
                        )
                    else:
                        fp[cid] = []
                print(f"  {name} (PDB): chains={all_chains}, "
                      f"sizes={[c['n_residues'] for c in data['chains'].values()]}")

            chain_id_dict[name] = (designed, fixed)
            fixed_pos_dict[name] = fp

            chain_info_all[name] = {
                "chain_ids": all_chains,
                "designed_chains": designed,
                "fixed_chains": fixed,
                "chain_sequences": {cid: cdata["seq"] for cid, cdata in data["chains"].items()},
                "chain_sizes": {cid: cdata["n_residues"] for cid, cdata in data["chains"].items()},
            }

    with open(chain_ids_jsonl, "w") as f:
        f.write(json.dumps(chain_id_dict) + "\n")

    with open(fixed_pos_jsonl, "w") as f:
        f.write(json.dumps(fixed_pos_dict) + "\n")

    chain_info_path = output_dir / "chain_info.json"
    with open(chain_info_path, "w") as f:
        json.dump(chain_info_all, f, indent=2)

    print(f"\nWrote: {chains_jsonl}")
    print(f"Wrote: {chain_ids_jsonl}")
    print(f"Wrote: {fixed_pos_jsonl}")
    print(f"Wrote: {chain_info_path}")


if __name__ == "__main__":
    main()
