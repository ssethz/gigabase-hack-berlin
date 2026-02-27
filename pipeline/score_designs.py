#!/usr/bin/env python3
"""Score designed binders using ipSAE and Boltz-2 confidence metrics.

Processes Boltz-2 prediction outputs, runs ipSAE scoring, and compiles
a ranked results table for downstream filtering.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IPSAE_SCRIPT = Path("/cluster/project/krause/ssussex/gigabase/IPSAE/ipsae.py")


def find_boltz_outputs(output_dir: Path) -> list[dict]:
    """Find all Boltz-2 prediction outputs in a directory tree."""
    results = []

    for conf_file in sorted(output_dir.glob("**/confidence_*.json")):
        with open(conf_file) as f:
            confidence = json.load(f)

        # Find corresponding structure file
        stem = conf_file.stem.replace("confidence_", "")
        pdb_file = conf_file.parent / f"{stem}.pdb"
        cif_file = conf_file.parent / f"{stem}.cif"
        pae_file = conf_file.parent / f"pae_{stem}.npz"

        struct_file = pdb_file if pdb_file.exists() else (cif_file if cif_file.exists() else None)

        design_name = conf_file.parent.name
        sample_name = stem

        results.append({
            "design_name": design_name,
            "sample_name": sample_name,
            "confidence_file": str(conf_file),
            "structure_file": str(struct_file) if struct_file else None,
            "pae_file": str(pae_file) if pae_file.exists() else None,
            "confidence_score": confidence.get("confidence_score", 0),
            "ptm": confidence.get("ptm", 0),
            "iptm": confidence.get("iptm", 0),
            "protein_iptm": confidence.get("protein_iptm", 0),
            "complex_plddt": confidence.get("complex_plddt", 0),
            "complex_iplddt": confidence.get("complex_iplddt", 0),
            "complex_pde": confidence.get("complex_pde", 0),
            "complex_ipde": confidence.get("complex_ipde", 0),
        })

    return results


def _parse_ipsae_lines(lines) -> dict:
    """Parse ipSAE score lines looking for chain A-B interaction."""
    scores = {"ipsae": None, "ipsae_d0chn": None, "ipsae_d0dom": None}
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 8 and parts[0] == "A" and parts[1] == "B":
            if parts[4] == "max":
                scores["ipsae"] = float(parts[5])
                scores["ipsae_d0chn"] = float(parts[6])
                scores["ipsae_d0dom"] = float(parts[7])
                break
            elif parts[4] == "asym" and scores["ipsae"] is None:
                scores["ipsae"] = float(parts[5])
                scores["ipsae_d0chn"] = float(parts[6])
                scores["ipsae_d0dom"] = float(parts[7])
    return scores


def run_ipsae(pae_file: str, struct_file: str, pae_cutoff: int = 10, dist_cutoff: int = 10) -> dict:
    """Run ipSAE scoring on a single prediction."""
    if not Path(pae_file).exists() or not Path(struct_file).exists():
        return {"ipsae": None, "ipsae_d0chn": None, "ipsae_d0dom": None}

    try:
        result = subprocess.run(
            [sys.executable, str(IPSAE_SCRIPT), pae_file, struct_file,
             str(pae_cutoff), str(dist_cutoff)],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            return {"ipsae": None, "ipsae_d0chn": None, "ipsae_d0dom": None}

        # ipSAE writes to {struct_stem}_{pae}_{dist}.txt alongside the structure
        struct_path = Path(struct_file)
        pae_str = f"{int(pae_cutoff):02d}"
        dist_str = f"{int(dist_cutoff):02d}"
        score_file = struct_path.parent / f"{struct_path.stem}_{pae_str}_{dist_str}.txt"

        if score_file.exists():
            with open(score_file) as f:
                scores = _parse_ipsae_lines(f)
            if scores["ipsae"] is not None:
                return scores

        # Fallback: parse stdout from the subprocess
        if result.stdout:
            scores = _parse_ipsae_lines(result.stdout.splitlines())
            if scores["ipsae"] is not None:
                return scores

        return {"ipsae": None, "ipsae_d0chn": None, "ipsae_d0dom": None}

    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  ipSAE error: {e}")
        return {"ipsae": None, "ipsae_d0chn": None, "ipsae_d0dom": None}


def main():
    parser = argparse.ArgumentParser(description="Score designs with ipSAE + Boltz metrics.")
    parser.add_argument("--boltz_dir", required=True, help="Boltz-2 output directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--run_ipsae", action="store_true", help="Run ipSAE scoring (slower)")
    parser.add_argument("--pae_cutoff", type=int, default=10, help="PAE cutoff for ipSAE")
    parser.add_argument("--dist_cutoff", type=int, default=10, help="Distance cutoff for ipSAE")
    args = parser.parse_args()

    boltz_dir = Path(args.boltz_dir)
    print(f"Scanning Boltz outputs in: {boltz_dir}")
    results = find_boltz_outputs(boltz_dir)
    print(f"Found {len(results)} predictions")

    if args.run_ipsae:
        print(f"\nRunning ipSAE scoring (PAE={args.pae_cutoff}, dist={args.dist_cutoff})...")
        for i, r in enumerate(results):
            if r["pae_file"] and r["structure_file"]:
                print(f"  [{i+1}/{len(results)}] {r['design_name']}/{r['sample_name']}")
                ipsae_scores = run_ipsae(
                    r["pae_file"], r["structure_file"],
                    args.pae_cutoff, args.dist_cutoff,
                )
                r.update(ipsae_scores)
            else:
                r.update({"ipsae": None, "ipsae_d0chn": None, "ipsae_d0dom": None})

    # Sort by ipSAE (if available) then by iptm
    def sort_key(x):
        ipsae = x.get("ipsae")
        iptm = x.get("iptm", 0)
        return (ipsae if ipsae is not None else -1, iptm)

    results.sort(key=sort_key, reverse=True)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "design_name", "sample_name",
        "ipsae", "ipsae_d0chn", "ipsae_d0dom",
        "confidence_score", "iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt",
        "ptm", "complex_pde", "complex_ipde",
        "structure_file", "pae_file",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")
    print(f"\nTop 10 designs:")
    print(f"{'Rank':<6}{'Design':<35}{'ipSAE':>8}{'ipTM':>8}{'ipLDDT':>8}{'conf':>8}")
    print("-" * 80)
    for i, r in enumerate(results[:10], 1):
        ipsae = f"{r.get('ipsae'):>8.4f}" if r.get("ipsae") is not None else "     N/A"
        print(f"{i:<6}{r['design_name']:<35}{ipsae}{r['iptm']:>8.4f}"
              f"{r['complex_iplddt']:>8.2f}{r['confidence_score']:>8.4f}")


if __name__ == "__main__":
    main()
