#!/usr/bin/env python3
"""Prepare 15-PGDH target for DARPin binder design pipeline.

Takes PDB 2GDZ, builds the biological dimer, extracts a clean monomer chain,
identifies interface hotspot residues, and writes config files for downstream
RFdiffusion partial diffusion / ProteinMPNN / Boltz-2 steps.
"""

import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
BIOMT_ROT = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
BIOMT_TRANS = np.array([0.0, 0.0, 195.771])

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class ProteinOnlySelect(Select):
    """Select only standard protein residues (no ligands, waters, etc.)."""
    def accept_residue(self, residue):
        return residue.id[0] == " "


class ChainSelect(Select):
    """Select a single chain, protein residues only."""
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id[0] == " "


def build_dimer(structure):
    """Apply BIOMT2 crystallographic symmetry to generate chain B."""
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue

    model = structure[0]
    chain_a = model["A"]
    chain_b = Chain("B")

    for residue in chain_a.get_residues():
        new_res = Residue(residue.id, residue.resname, residue.segid)
        for atom in residue.get_atoms():
            new_atom = atom.copy()
            coord = atom.get_vector().get_array()
            new_atom.set_coord(BIOMT_ROT @ coord + BIOMT_TRANS)
            new_res.add(new_atom)
        chain_b.add(new_res)

    model.add(chain_b)
    return structure


def find_interface_residues(structure, cutoff=5.0):
    """Find interface residues using heavy-atom distance cutoff between chains."""
    model = structure[0]
    chain_a = model["A"]
    chain_b = model["B"]

    atoms_a = [a for r in chain_a.get_residues() if r.id[0] == " "
               for a in r.get_atoms() if a.element != "H"]
    atoms_b = [a for r in chain_b.get_residues() if r.id[0] == " "
               for a in r.get_atoms() if a.element != "H"]

    ns_b = NeighborSearch(atoms_b)
    interface_residues = {}

    for atom in atoms_a:
        nearby = ns_b.search(atom.get_vector().get_array(), cutoff)
        if nearby:
            res = atom.get_parent()
            resnum = res.id[1]
            if resnum not in interface_residues:
                interface_residues[resnum] = {
                    "resname": res.resname,
                    "resname_1": THREE_TO_ONE.get(res.resname, "X"),
                    "n_contacts": 0,
                }
            interface_residues[resnum]["n_contacts"] += len(nearby)

    return interface_residues


def get_sequence(chain):
    """Extract amino acid sequence from a chain."""
    seq = []
    for res in chain.get_residues():
        if res.id[0] == " ":
            seq.append(THREE_TO_ONE.get(res.resname, "X"))
    return "".join(seq)


def main():
    pdb_path = REPO_ROOT / "2GDZ.pdb"
    if not pdb_path.exists():
        print(f"Downloading 2GDZ.pdb...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://files.rcsb.org/download/2GDZ.pdb", str(pdb_path)
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("15PGDH", str(pdb_path))

    # Build biological dimer
    print("Building biological dimer from BIOMT transformation...")
    structure = build_dimer(structure)

    targets_dir = REPO_ROOT / "targets"
    targets_dir.mkdir(exist_ok=True)
    configs_dir = REPO_ROOT / "configs"
    configs_dir.mkdir(exist_ok=True)

    io = PDBIO()
    io.set_structure(structure)

    # Save clean dimer (protein only)
    dimer_path = targets_dir / "15pgdh_dimer.pdb"
    io.save(str(dimer_path), ProteinOnlySelect())
    print(f"Saved dimer: {dimer_path}")

    # Save clean monomer chain A (protein only)
    monomer_path = targets_dir / "15pgdh_chainA.pdb"
    io.save(str(monomer_path), ChainSelect("A"))
    print(f"Saved monomer: {monomer_path}")

    # Extract sequence
    chain_a = structure[0]["A"]
    sequence = get_sequence(chain_a)
    seq_path = targets_dir / "15pgdh_sequence.fasta"
    with open(seq_path, "w") as f:
        f.write(f">15-PGDH_HPGD_P15428\n{sequence}\n")
    print(f"Saved sequence ({len(sequence)} aa): {seq_path}")

    # Find interface residues
    print("\nAnalyzing dimer interface (heavy atom cutoff = 5.0 A)...")
    interface_res = find_interface_residues(structure, cutoff=5.0)
    print(f"Found {len(interface_res)} interface residues on chain A")

    # Categorize by structural region
    alpha9_range = set(range(143, 173))
    lid_range = set(range(185, 218))

    alpha9_hotspots = sorted(r for r in interface_res if r in alpha9_range)
    lid_hotspots = sorted(r for r in interface_res if r in lid_range)
    other_hotspots = sorted(r for r in interface_res
                            if r not in alpha9_range and r not in lid_range)

    # Select primary hotspots for RFdiffusion (alpha9 core)
    # Focus on residues with highest contact counts
    sorted_by_contacts = sorted(
        interface_res.items(), key=lambda x: x[1]["n_contacts"], reverse=True
    )
    top_hotspots = [r for r, _ in sorted_by_contacts[:12]]

    print(f"\nAlpha9 helix interface residues: {alpha9_hotspots}")
    print(f"Lid domain interface residues: {lid_hotspots}")
    print(f"Other interface residues: {other_hotspots}")
    print(f"Top 12 hotspots by contact density: {sorted(top_hotspots)}")

    # Write pipeline config
    config = {
        "target": {
            "name": "15-PGDH",
            "gene": "HPGD",
            "uniprot": "P15428",
            "pdb": "2GDZ",
            "sequence": sequence,
            "n_residues": len(sequence),
        },
        "target_files": {
            "monomer_pdb": str(monomer_path),
            "dimer_pdb": str(dimer_path),
            "sequence_fasta": str(seq_path),
        },
        "interface": {
            "all_residues": sorted(interface_res.keys()),
            "alpha9_helix": alpha9_hotspots,
            "lid_domain": lid_hotspots,
            "other": other_hotspots,
            "residue_details": {
                str(k): v for k, v in sorted(interface_res.items())
            },
        },
        "hotspots": {
            "primary": sorted(top_hotspots),
            "primary_rfdiff_format": [f"A{r}" for r in sorted(top_hotspots)],
            "alpha9_focused": [f"A{r}" for r in alpha9_hotspots],
        },
        "design_params": {
            "max_binder_length": 250,
            "partial_T_values": [15, 25, 35],
            "mpnn_seqs_per_backbone": 8,
            "mpnn_temperature": 0.1,
            "boltz_diffusion_samples": 5,
        },
    }

    config_path = configs_dir / "pipeline_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved pipeline config: {config_path}")

    # Print summary for RFdiffusion command construction
    hotspot_str = ",".join(config["hotspots"]["primary_rfdiff_format"])
    print(f"\n{'='*60}")
    print("RFdiffusion hotspot string:")
    print(f"  ppi.hotspot_res=[{hotspot_str}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
