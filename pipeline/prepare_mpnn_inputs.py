#!/usr/bin/env python3
"""Prepare ProteinMPNN input JSONLs from RFdiffusion backbone PDBs.

Reads all PDB files from a backbone directory (RFdiffusion output),
identifies the target chain (A = 15-PGDH) and binder chain (B),
and writes JSONL files for ProteinMPNN with the target fixed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
import warnings

warnings.filterwarnings("ignore")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def pdb_to_mpnn_dict(pdb_path: Path) -> dict:
    """Convert a PDB file to ProteinMPNN input dict format."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]

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

    return {
        "name": pdb_path.stem,
        "chains": chains,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ProteinMPNN inputs from RFdiffusion backbones."
    )
    parser.add_argument("--backbone_dir", required=True, help="Directory with backbone PDBs")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--target_chain", default="A", help="Target chain to fix (default: A)")
    parser.add_argument("--fix_binder_helices", action="store_true",
                        help="Fix DARPin helix positions in binder chain (design loops only)")
    args = parser.parse_args()

    # 2XEH NI3C Mut6 DARPin helix ranges (1-indexed, after renumbering)
    DARPIN_HELIX_POSITIONS = set()
    for s, e in [(1,13),(14,24),(37,46),(47,57),(70,79),(80,90),
                 (103,112),(113,123),(136,145),(146,157)]:
        DARPIN_HELIX_POSITIONS.update(range(s, e + 1))

    backbone_dir = Path(args.backbone_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(backbone_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"No PDB files found in {backbone_dir}")
        return

    print(f"Processing {len(pdb_files)} backbone PDBs...")

    chains_jsonl = output_dir / "chains.jsonl"
    chain_ids_jsonl = output_dir / "chain_ids.jsonl"
    fixed_pos_jsonl = output_dir / "fixed_positions.jsonl"

    chain_id_dict = {}
    fixed_pos_dict = {}

    with open(chains_jsonl, "w") as f_chains:
        for pdb_path in pdb_files:
            try:
                data = pdb_to_mpnn_dict(pdb_path)
            except Exception as e:
                print(f"  Skipping {pdb_path.name}: {e}")
                continue

            if not data["chains"]:
                continue

            name = data["name"]

            # ProteinMPNN expected format: concat seq, nested coords dicts
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

            # chain_id_dict: (designed_chains, fixed_chains)
            all_chains = list(data["chains"].keys())
            designed = [c for c in all_chains if c != args.target_chain]
            fixed = [c for c in all_chains if c == args.target_chain]
            chain_id_dict[name] = (designed, fixed)

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
            fixed_pos_dict[name] = fp

            print(f"  {name}: chains={all_chains}, "
                  f"sizes={[c['n_residues'] for c in data['chains'].values()]}")

    with open(chain_ids_jsonl, "w") as f:
        f.write(json.dumps(chain_id_dict) + "\n")

    with open(fixed_pos_jsonl, "w") as f:
        f.write(json.dumps(fixed_pos_dict) + "\n")

    print(f"\nWrote: {chains_jsonl}")
    print(f"Wrote: {chain_ids_jsonl}")
    print(f"Wrote: {fixed_pos_jsonl}")


if __name__ == "__main__":
    main()
