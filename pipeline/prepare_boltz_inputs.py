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


def write_boltz_yaml(target_seq: str, binder_seq: str, name: str, output_path: Path,
                     socs_box_seq: str | None = None, linker_seq: str | None = None):
    """Write a Boltz-2 YAML input file for a target-binder complex.

    If socs_box_seq is provided, models the bioPROTAC ternary complex:
    chain A = 15-PGDH, chain B = DARPin-linker-SOCS_box fusion.
    """
    if socs_box_seq:
        linker = linker_seq or "GSGSGSGSG"
        fusion_seq = binder_seq + linker + socs_box_seq
        yaml_content = f"""# Boltz-2 prediction: {name} (bioPROTAC fusion)
# Target: 15-PGDH (chain A), Binder: DARPin-linker-SOCS2_box fusion (chain B)
sequences:
  - protein:
      id: A
      sequence: {target_seq}
      msa: empty
  - protein:
      id: B
      sequence: {fusion_seq}
      msa: empty
"""
    else:
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


def write_boltz_yaml_multichain(chain_sequences: dict[str, str], name: str, output_path: Path):
    """Write a Boltz-2 YAML for an arbitrary multi-chain complex."""
    lines = [f"# Boltz-2 prediction: {name}", "sequences:"]
    for chain_id, seq in chain_sequences.items():
        lines.append(f"  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {seq}")
        lines.append(f"      msa: empty")
    yaml_content = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Boltz-2 YAMLs from ProteinMPNN outputs."
    )
    parser.add_argument("--mpnn_dir", required=True,
                        help="ProteinMPNN output directory containing FASTA files")
    parser.add_argument("--config", default=None,
                        help="Pipeline config JSON (for target sequence; not needed with --chain_info)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for Boltz YAML files")
    parser.add_argument("--max_per_backbone", type=int, default=4,
                        help="Max sequences per backbone to validate (default: 4)")
    parser.add_argument("--fusion", action="store_true",
                        help="Model bioPROTAC fusion: DARPin-linker-SOCS_box + 15-PGDH")
    parser.add_argument("--linker", type=str, default=None,
                        help="Linker sequence for fusion (default: from config or GSGSGSGSG)")
    parser.add_argument("--chain_info", default=None,
                        help="chain_info.json from prepare_mpnn_inputs.py (enables multi-chain mode)")
    args = parser.parse_args()

    multichain_mode = args.chain_info is not None
    chain_info = None
    target_seq = None
    socs_box_seq = None
    linker_seq = args.linker

    if multichain_mode:
        with open(args.chain_info) as f:
            chain_info = json.load(f)
        print(f"Multi-chain mode: using chain_info from {args.chain_info}")
        print(f"  Backbones in chain_info: {list(chain_info.keys())}")
    else:
        if not args.config:
            print("ERROR: --config is required unless --chain_info is provided")
            return
        with open(args.config) as f:
            config = json.load(f)
        target_seq = config["target"]["sequence"]

        if args.fusion:
            socs_cfg = config.get("socs_box")
            if not socs_cfg:
                print("ERROR: --fusion requires 'socs_box' section in config")
                return
            socs_box_seq = socs_cfg["sequence"]
            if not linker_seq:
                linker_seq = socs_cfg["linker_options"].get("flexible", "GSGSGSGSG")
            print(f"Fusion mode: DARPin + {linker_seq} + SOCS2 box ({len(socs_box_seq)} aa)")

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

        designed_seqs = sequences[1:]
        designed_seqs = designed_seqs[:args.max_per_backbone]

        # Resolve chain_info for this backbone (try exact match, then prefix match)
        info = None
        if chain_info:
            info = chain_info.get(backbone_name)
            if info is None:
                for key in chain_info:
                    if backbone_name.startswith(key) or key.startswith(backbone_name):
                        info = chain_info[key]
                        break

        for i, seq_entry in enumerate(designed_seqs):
            full_seq = seq_entry["sequence"]
            chain_seqs = full_seq.split("/")

            if multichain_mode and info:
                chain_ids = info["chain_ids"]
                designed_chain_ids = info.get("designed_chains", [])
                fixed_chain_ids = info.get("fixed_chains", [])
                original_seqs = info.get("chain_sequences", {})

                if len(chain_seqs) == len(chain_ids):
                    # FASTA has all chains (separated by /)
                    chain_sequences = dict(zip(chain_ids, chain_seqs))
                elif len(chain_seqs) == len(designed_chain_ids):
                    # FASTA has only designed chain(s); fill fixed from chain_info
                    chain_sequences = {}
                    for cid in chain_ids:
                        if cid in fixed_chain_ids:
                            chain_sequences[cid] = original_seqs[cid]
                        else:
                            chain_sequences[cid] = chain_seqs[designed_chain_ids.index(cid)]
                elif len(chain_seqs) == 1:
                    # Single sequence: assume it's the first designed chain, fill rest from originals
                    chain_sequences = dict(original_seqs)
                    if designed_chain_ids:
                        chain_sequences[designed_chain_ids[0]] = chain_seqs[0]
                else:
                    print(f"  WARNING: {backbone_name} seq{i}: "
                          f"FASTA has {len(chain_seqs)} chains, cannot map to {len(chain_ids)} chain_ids")
                    continue

                name = f"{backbone_name}_seq{i}"
                yaml_path = output_dir / f"{name}.yaml"
                write_boltz_yaml_multichain(chain_sequences, name, yaml_path)

                binder_seqs = [chain_sequences[c] for c in designed_chain_ids if c in chain_sequences]
                binder_seq = binder_seqs[0] if binder_seqs else chain_seqs[0]
            else:
                binder_seq = chain_seqs[-1]

                if len(binder_seq) > 250:
                    continue

                name = f"{backbone_name}_seq{i}"
                if socs_box_seq:
                    name = f"{backbone_name}_seq{i}_fusion"
                yaml_path = output_dir / f"{name}.yaml"
                write_boltz_yaml(target_seq, binder_seq, name, yaml_path,
                                 socs_box_seq=socs_box_seq, linker_seq=linker_seq)

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
