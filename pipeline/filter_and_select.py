#!/usr/bin/env python3
"""Filter and select top binder designs using multi-criteria ranking.

Reads the scored designs CSV, applies quality filters, performs structural
diversity clustering, and selects a final set for experimental validation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, Superimposer
import warnings

warnings.filterwarnings("ignore")


def load_scores(csv_path: Path) -> list[dict]:
    """Load scored designs from CSV."""
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["ipsae", "ipsae_d0chn", "ipsae_d0dom",
                        "confidence_score", "iptm", "protein_iptm",
                        "complex_plddt", "complex_iplddt",
                        "ptm", "complex_pde", "complex_ipde"]:
                try:
                    row[key] = float(row[key]) if row[key] and row[key] != "None" else None
                except (ValueError, KeyError):
                    row[key] = None
            results.append(row)
    return results


def apply_filters(designs: list[dict], min_iptm: float = 0.3,
                  min_iplddt: float = 50.0, min_ipsae: float = None) -> list[dict]:
    """Apply quality filters to designs.

    Designs missing required scores (iptm or complex_iplddt) are rejected.
    """
    filtered = []
    for d in designs:
        iptm = d.get("iptm")
        iplddt = d.get("complex_iplddt")
        if iptm is None or iptm < min_iptm:
            continue
        if iplddt is None or iplddt < min_iplddt:
            continue
        if min_ipsae is not None:
            ipsae = d.get("ipsae")
            if ipsae is None or ipsae < min_ipsae:
                continue
        filtered.append(d)
    return filtered


def compute_ca_rmsd(pdb1_path: str, pdb2_path: str, chain_id: str = "B") -> float:
    """Compute superimposed CA RMSD between binder chains of two structures."""
    parser = PDBParser(QUIET=True)

    try:
        s1 = parser.get_structure("s1", pdb1_path)
        s2 = parser.get_structure("s2", pdb2_path)
    except Exception:
        return float("inf")

    def get_ca_atoms(structure, cid):
        atoms = []
        for chain in structure[0].get_chains():
            if chain.id == cid:
                for res in chain.get_residues():
                    if res.id[0] == " " and "CA" in res:
                        atoms.append(res["CA"])
        return atoms

    ca1 = get_ca_atoms(s1, chain_id)
    ca2 = get_ca_atoms(s2, chain_id)

    if len(ca1) == 0 or len(ca2) == 0 or len(ca1) != len(ca2):
        return float("inf")

    sup = Superimposer()
    sup.set_atoms(ca1, ca2)
    return sup.rms


def greedy_diversity_selection(designs: list[dict], n_select: int,
                               rmsd_threshold: float = 3.0) -> list[dict]:
    """Select diverse designs using greedy max-min diversity.
    
    If structure files are not available, falls back to selecting top-ranked.
    """
    if not designs:
        return []

    has_structures = all(d.get("structure_file") and
                        Path(d["structure_file"]).exists()
                        for d in designs[:min(50, len(designs))])

    if not has_structures:
        print("  No structure files for diversity clustering, using top-ranked.")
        return designs[:n_select]

    selected = [designs[0]]  # Start with best-scored
    remaining = list(designs[1:])

    while len(selected) < n_select and remaining:
        best_candidate = None
        best_min_rmsd = -1

        for candidate in remaining[:50]:  # Only check top 50 for speed
            min_rmsd = float("inf")
            for sel in selected:
                try:
                    rmsd = compute_ca_rmsd(candidate["structure_file"],
                                          sel["structure_file"])
                    min_rmsd = min(min_rmsd, rmsd)
                except Exception:
                    pass

            if min_rmsd > best_min_rmsd:
                best_min_rmsd = min_rmsd
                best_candidate = candidate

        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    return selected


def main():
    parser = argparse.ArgumentParser(description="Filter and select top designs.")
    parser.add_argument("--scores_csv", required=True, help="Scored designs CSV")
    parser.add_argument("--output", required=True, help="Output JSON with selected designs")
    parser.add_argument("--n_select", type=int, default=20, help="Number of designs to select")
    parser.add_argument("--min_iptm", type=float, default=0.3, help="Minimum ipTM filter")
    parser.add_argument("--min_iplddt", type=float, default=50.0, help="Minimum ipLDDT filter")
    parser.add_argument("--min_ipsae", type=float, default=None, help="Minimum ipSAE filter")
    parser.add_argument("--diversity_rmsd", type=float, default=3.0,
                        help="RMSD threshold for diversity clustering")
    args = parser.parse_args()

    print(f"Loading scores from: {args.scores_csv}")
    designs = load_scores(args.scores_csv)
    print(f"Total designs: {len(designs)}")

    print(f"\nApplying filters (ipTM >= {args.min_iptm}, ipLDDT >= {args.min_iplddt}"
          f"{f', ipSAE >= {args.min_ipsae}' if args.min_ipsae else ''})...")
    filtered = apply_filters(designs, args.min_iptm, args.min_iplddt, args.min_ipsae)
    print(f"After filtering: {len(filtered)} designs")

    filtered.sort(
        key=lambda x: (x.get("ipsae") if x.get("ipsae") is not None else -1,
                       x.get("iptm", 0) or 0),
        reverse=True,
    )

    if not filtered:
        print("No designs passed filters. Relaxing to top designs by ipTM...")
        designs.sort(key=lambda x: x.get("iptm", 0) or 0, reverse=True)
        filtered = designs[:args.n_select]

    print(f"\nSelecting {args.n_select} diverse designs...")
    selected = greedy_diversity_selection(filtered, args.n_select, args.diversity_rmsd)
    print(f"Selected: {len(selected)} designs")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "n_total": len(designs),
        "n_filtered": len(filtered),
        "n_selected": len(selected),
        "filters": {
            "min_iptm": args.min_iptm,
            "min_iplddt": args.min_iplddt,
            "min_ipsae": args.min_ipsae,
        },
        "designs": selected,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nOutput: {output_path}")

    # Print final selection summary
    print(f"\n{'='*80}")
    print("FINAL SELECTED DESIGNS")
    print(f"{'='*80}")
    print(f"{'#':<4}{'Design':<35}{'ipSAE':>8}{'ipTM':>8}{'ipLDDT':>8}")
    print("-" * 70)
    for i, d in enumerate(selected, 1):
        ipsae = f"{d['ipsae']:.4f}" if d.get("ipsae") is not None else "N/A"
        iptm = f"{d['iptm']:.4f}" if d.get("iptm") is not None else "N/A"
        iplddt = f"{d['complex_iplddt']:.1f}" if d.get("complex_iplddt") is not None else "N/A"
        print(f"{i:<4}{d['design_name']:<35}{ipsae:>8}{iptm:>8}{iplddt:>8}")

    # Also write a simple FASTA of selected binder sequences
    fasta_path = output_path.parent / "selected_binders.fasta"
    # This would need the manifest to get sequences - left for orchestrator


if __name__ == "__main__":
    main()
