#!/usr/bin/env python3
"""Generate Boltz-2 YAML input files from ProteinMPNN sequence outputs.

Reads FASTA files from ProteinMPNN, pairs each designed binder sequence
with the 15-PGDH target sequence, and writes Boltz-2 YAML configs.
"""

import argparse
import json
import re
from pathlib import Path


def parse_mpnn_fasta(fasta_path: Path) -> list[dict]:
    """Parse a ProteinMPNN output FASTA file.
    
    Returns list of dicts with 'name', 'score', 'sequence', 'recovery'.
    """
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header and current_seq:
                    sequences.append({
                        "header": current_header,
                        "sequence": "".join(current_seq),
                    })
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_header and current_seq:
        sequences.append({
            "header": current_header,
            "sequence": "".join(current_seq),
        })

    return sequences


def write_boltz_yaml(target_seq: str, binder_seq: str, name: str, output_path: Path):
    """Write a Boltz-2 YAML input file for a target-binder complex."""
    yaml_content = f"""# Boltz-2 prediction: {name}
# Target: 15-PGDH (chain A), Binder: designed (chain B)
sequences:
  - protein:
      id: A
      sequence: {target_seq}
      msa: empty
  - protein:
      id: B
      sequence: {binder_seq}
      msa: empty
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Boltz-2 YAMLs from ProteinMPNN outputs."
    )
    parser.add_argument("--mpnn_dir", required=True,
                        help="ProteinMPNN output directory containing FASTA files")
    parser.add_argument("--config", required=True,
                        help="Pipeline config JSON (for target sequence)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for Boltz YAML files")
    parser.add_argument("--max_per_backbone", type=int, default=4,
                        help="Max sequences per backbone to validate (default: 4)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    target_seq = config["target"]["sequence"]
    mpnn_dir = Path(args.mpnn_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(mpnn_dir.glob("**/*.fa"))
    if not fasta_files:
        print(f"No FASTA files found in {mpnn_dir}")
        return

    total_yamls = 0
    manifest = []

    for fasta_path in fasta_files:
        sequences = parse_mpnn_fasta(fasta_path)
        if not sequences:
            continue

        backbone_name = fasta_path.stem

        # Skip the first entry (usually the original/input sequence)
        designed_seqs = sequences[1:]

        # Take top N by ProteinMPNN score (lower is better)
        designed_seqs = designed_seqs[:args.max_per_backbone]

        for i, seq_entry in enumerate(designed_seqs):
            full_seq = seq_entry["sequence"]
            # ProteinMPNN may concatenate chains with '/' or output only the designed chain
            chains = full_seq.split("/")
            binder_seq = chains[-1]

            if len(binder_seq) > 250:
                continue

            name = f"{backbone_name}_seq{i}"
            yaml_path = output_dir / f"{name}.yaml"
            write_boltz_yaml(target_seq, binder_seq, name, yaml_path)

            manifest.append({
                "name": name,
                "backbone": backbone_name,
                "binder_sequence": binder_seq,
                "binder_length": len(binder_seq),
                "yaml_path": str(yaml_path),
            })
            total_yamls += 1

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {total_yamls} Boltz-2 YAML files in {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
