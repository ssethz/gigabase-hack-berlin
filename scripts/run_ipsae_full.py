#!/usr/bin/env python3
"""Run Boltz on designs to get full PAE, then compute ipSAE scores."""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import gemmi

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def log(msg: str) -> None:
    """Print with timestamp and flush."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _extract_sequences_boltzgen(content: str) -> list[str]:
    """Extract sequences from boltzgen CIF format."""
    sequences = []
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('1 polypeptide') or line.startswith('2 polypeptide'):
            parts = line.split()
            if len(parts) >= 4:
                seq = parts[-1]
                if len(seq) > 10:
                    sequences.append(seq)
    return sequences


def _extract_sequences_mmcif(cif_path: Path) -> list[str]:
    """Extract sequences from standard mmCIF format (AF3, PDB) using _entity_poly_seq."""
    doc = gemmi.cif.read(str(cif_path))
    block = doc[0]

    entity_seq = block.find("_entity_poly_seq.", ["entity_id", "mon_id", "num"])
    if not entity_seq:
        return []

    seqs_by_entity: dict[str, list[tuple[int, str]]] = {}
    for row in entity_seq:
        eid = row[0]
        if eid not in seqs_by_entity:
            seqs_by_entity[eid] = []
        seqs_by_entity[eid].append((int(row[2]), row[1]))

    sequences = []
    for eid in sorted(seqs_by_entity.keys()):
        residues = sorted(seqs_by_entity[eid], key=lambda x: x[0])
        seq = ''.join(AA_3TO1.get(r[1], 'X') for r in residues)
        sequences.append(seq)

    return sequences


def _extract_sequences_from_structure(cif_path: Path) -> list[str]:
    """Extract sequences from CIF by reading atom records directly (Chai, RFdiffusion)."""
    try:
        st = gemmi.read_structure(str(cif_path))
        sequences = []
        for chain in st[0]:
            seq = ''.join(AA_3TO1.get(res.name, '') for res in chain if res.name in AA_3TO1)
            if seq:
                sequences.append(seq)
        return sequences
    except Exception:
        return []


def extract_sequences_from_cif(cif_path: Path) -> tuple[list[str], bool]:
    """Extract sequences from CIF file. Returns (sequences, is_boltzgen_format)."""
    with open(cif_path) as f:
        content = f.read()

    # Try boltzgen format first (has "N polypeptide ... SEQUENCE" lines)
    sequences = _extract_sequences_boltzgen(content)
    if sequences:
        return sequences, True

    # Try standard mmCIF format (AF3, PDB) with _entity_poly_seq
    sequences = _extract_sequences_mmcif(cif_path)
    if sequences:
        return sequences, False

    # Fall back to reading structure directly (Chai, RFdiffusion)
    return _extract_sequences_from_structure(cif_path), False


def create_boltz_yaml(
    design_cif: Path,
    target_cif: Path | None,
    output_yaml: Path,
    use_msa: bool,
    use_template: bool,
) -> None:
    """Create YAML for Boltz prediction."""
    sequences, is_boltzgen = extract_sequences_from_cif(design_cif)

    if not is_boltzgen and use_template:
        raise ValueError(f"AF3/mmCIF format detected. Template not supported for {design_cif.name}")

    if len(sequences) < 2:
        raise ValueError(f"Expected 2 sequences in {design_cif}, got {len(sequences)}")

    target_seq, nanobody_seq = sequences[0], sequences[1]

    target_msa = "" if use_msa else "\n      msa: empty"
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {target_seq}{target_msa}
  - protein:
      id: B
      sequence: {nanobody_seq}
      msa: empty
"""
    if use_template:
        yaml_content += f"""templates:
  - cif: {target_cif}
    chain_id: A
    force: true
    threshold: 1.0
"""
    output_yaml.write_text(yaml_content)


def run_boltz_batch(
    yaml_dir: Path, output_dir: Path, cache_dir: Path, use_msa: bool
) -> bool:
    """Run Boltz prediction on all YAMLs in directory (batched)."""
    cmd = [
        "boltz", "predict",
        str(yaml_dir),
        "--out_dir", str(output_dir),
        "--cache", str(cache_dir),
        "--write_full_pae",
        "--recycling_steps", "3",
        "--diffusion_samples", "1",
    ]
    if use_msa:
        cmd.append("--use_msa_server")
    log(f"Command: {' '.join(cmd)}")
    log("Starting Boltz (this may take a few minutes)...")
    sys.stdout.flush()
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_ipsae(pae_npz: Path, cif_file: Path, ipsae_script: Path) -> dict[str, str]:
    """Run ipSAE scoring and return chain-pair scores."""
    result = subprocess.run(
        [sys.executable, str(ipsae_script), str(pae_npz), str(cif_file), "15", "15"],
        capture_output=True,
        text=True,
        cwd=ipsae_script.parent,
    )
    if result.returncode != 0 and result.stderr:
        log(f"    ipSAE error: {result.stderr[:200]}")

    output_txt = cif_file.parent / f"{cif_file.stem}_15_15.txt"
    if not output_txt.exists():
        return {}

    with open(output_txt) as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        return {}

    header = lines[0].split()
    if not header:
        return {}

    for line in reversed(lines[1:]):
        parts = line.split()
        if len(parts) >= 5 and parts[4] == "max":
            return dict(zip(header, parts[:len(header)]))

    parts = lines[1].split()
    if len(parts) >= 5:
        return dict(zip(header, parts[:len(header)]))
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Full ipSAE pipeline with Boltz prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Template only (no MSA):
  python run_ipsae_full.py --run-dir /path/to/run --use-template /path/to/target.cif ...

  # MSA only:
  python run_ipsae_full.py --run-dir /path/to/run --use-msa ...

  # Both:
  python run_ipsae_full.py --run-dir /path/to/run --use-msa --use-template /path/to/target.cif ...
""",
    )
    parser.add_argument("--run-dir", required=True, help="Boltzgen run directory")
    parser.add_argument("--ipsae-script", required=True, help="Path to ipsae.py")
    parser.add_argument("--cache", required=True, help="Boltz cache directory")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top designs to process")
    parser.add_argument("--use-msa", action="store_true", help="Use MSA server for target sequence")
    parser.add_argument("--use-template", metavar="CIF", help="Use template forcing from this CIF")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    target_cif = Path(args.use_template) if args.use_template else None
    use_template = args.use_template is not None
    ipsae_script = Path(args.ipsae_script)
    cache_dir = Path(args.cache)

    final_designs_dir = run_dir / "final_ranked_designs" / f"final_{args.top_n}_designs"
    if not final_designs_dir.exists():
        candidates = list((run_dir / "final_ranked_designs").glob("final_*_designs"))
        if candidates:
            final_designs_dir = candidates[0]
        else:
            log(f"Error: No final designs found in {run_dir}")
            sys.exit(1)

    mode_str = f"msa{int(args.use_msa)}_tpl{int(use_template)}"
    ipsae_dir = run_dir / f"ipsae_analysis_{mode_str}"
    ipsae_dir.mkdir(exist_ok=True)
    boltz_input_dir = ipsae_dir / "boltz_inputs"
    boltz_input_dir.mkdir(exist_ok=True)
    boltz_output_dir = ipsae_dir / "boltz_outputs"
    boltz_output_dir.mkdir(exist_ok=True)
    log(f"Settings: use_msa={args.use_msa}, use_template={use_template}")

    design_cifs = sorted(final_designs_dir.glob("*.cif"))
    log(f"Found {len(design_cifs)} designs in {final_designs_dir}")

    log("=== Step 1/3: Creating Boltz input YAMLs ===")
    valid_designs = []
    for i, design_cif in enumerate(design_cifs, 1):
        name = design_cif.stem
        yaml_file = boltz_input_dir / f"{name}.yaml"
        try:
            create_boltz_yaml(
                design_cif, target_cif, yaml_file, args.use_msa, use_template
            )
            valid_designs.append(name)
            log(f"  [{i}/{len(design_cifs)}] {name}.yaml")
        except ValueError as e:
            log(f"  [{i}/{len(design_cifs)}] ERROR {name}: {e}")

    if not valid_designs:
        log("No valid designs to process!")
        sys.exit(1)

    log(f"=== Step 2/3: Running Boltz predict ({len(valid_designs)} designs) ===")
    start_time = time.time()
    if not run_boltz_batch(boltz_input_dir, boltz_output_dir, cache_dir, args.use_msa):
        log("Boltz prediction failed!")
        sys.exit(1)
    elapsed = time.time() - start_time
    log(f"Boltz completed in {elapsed:.1f}s ({elapsed / len(valid_designs):.1f}s per design)")

    log("=== Step 3/3: Computing ipSAE scores ===")
    predictions_dir = boltz_output_dir / f"boltz_results_{boltz_input_dir.name}" / "predictions"
    if not predictions_dir.exists():
        predictions_dir = boltz_output_dir / "predictions"
    log(f"Looking for predictions in: {predictions_dir}")

    results = []
    for i, name in enumerate(valid_designs, 1):
        design_out = predictions_dir / name

        pae_files = list(design_out.glob("pae_*.npz"))
        if not pae_files:
            pae_files = list(design_out.glob("*.npz"))
        if not pae_files:
            log(f"  [{i}/{len(valid_designs)}] No PAE file for {name}")
            continue
        pae_npz = pae_files[0]

        struct_files = list(design_out.glob("*_model_*.cif"))
        if not struct_files:
            struct_files = list(design_out.glob("*.cif"))
        if not struct_files:
            log(f"  [{i}/{len(valid_designs)}] No structure for {name}")
            continue
        struct_cif = struct_files[0]

        scores = run_ipsae(pae_npz, struct_cif, ipsae_script)
        if scores:
            scores["design"] = name
            results.append(scores)
            log(f"  [{i}/{len(valid_designs)}] {name}: ipSAE={scores.get('ipSAE', 'N/A')}, pDockQ={scores.get('pDockQ', 'N/A')}")
        else:
            log(f"  [{i}/{len(valid_designs)}] No ipSAE output for {name}")

    if results:
        output_csv = ipsae_dir / "ipsae_scores.csv"
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["design"] + [k for k in results[0].keys() if k != "design"])
            writer.writeheader()
            writer.writerows(results)

        results_sorted = sorted(
            results,
            key=lambda x: float(x.get("ipSAE", 0)),
            reverse=True
        )

        log("=" * 50)
        log("RESULTS - Top designs by ipSAE:")
        log("=" * 50)
        for r in results_sorted:
            log(f"  {r['design']}: ipSAE={r.get('ipSAE', 'N/A')}, pDockQ={r.get('pDockQ', 'N/A')}, LIS={r.get('LIS', 'N/A')}")

        log(f"Results saved to: {output_csv}")
    else:
        log("No results generated!")


if __name__ == "__main__":
    main()
