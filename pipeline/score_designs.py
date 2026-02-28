#!/usr/bin/env python3
"""Score designed binders using ipSAE and Boltz-2 confidence metrics.

Processes Boltz-2 prediction outputs, runs ipSAE scoring, and compiles
a ranked results table for downstream filtering.
"""

import argparse
import csv
import json
import multiprocessing
import subprocess
import sys
from functools import partial
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


def _parse_ipsae_lines(lines, chain_pairs=None) -> dict:
    """Parse ipSAE score lines for specified chain pair interactions.

    Args:
        lines: iterable of score file lines
        chain_pairs: list of (chain1, chain2) tuples to extract, e.g. [("A","B"), ("A","C")].
                     Defaults to [("A","B")] for backward compatibility.

    Returns dict with keys like ipsae, ipsae_AB, ipsae_AC, etc.
    """
    if chain_pairs is None:
        chain_pairs = [("A", "B")]

    scores = {}
    found = {}

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue

        for c1, c2 in chain_pairs:
            pair_key = f"{c1}{c2}"
            if parts[0] == c1 and parts[1] == c2:
                if parts[4] == "max":
                    found[pair_key] = (float(parts[5]), float(parts[6]), float(parts[7]))
                elif parts[4] == "asym" and pair_key not in found:
                    found[pair_key] = (float(parts[5]), float(parts[6]), float(parts[7]))

    if len(chain_pairs) == 1:
        pair_key = f"{chain_pairs[0][0]}{chain_pairs[0][1]}"
        if pair_key in found:
            scores["ipsae"] = found[pair_key][0]
            scores["ipsae_d0chn"] = found[pair_key][1]
            scores["ipsae_d0dom"] = found[pair_key][2]
        else:
            scores["ipsae"] = None
            scores["ipsae_d0chn"] = None
            scores["ipsae_d0dom"] = None
    else:
        for c1, c2 in chain_pairs:
            pair_key = f"{c1}{c2}"
            suffix = f"_{pair_key}"
            if pair_key in found:
                scores[f"ipsae{suffix}"] = found[pair_key][0]
                scores[f"ipsae_d0chn{suffix}"] = found[pair_key][1]
                scores[f"ipsae_d0dom{suffix}"] = found[pair_key][2]
            else:
                scores[f"ipsae{suffix}"] = None
                scores[f"ipsae_d0chn{suffix}"] = None
                scores[f"ipsae_d0dom{suffix}"] = None

        # Composite: max ipSAE across all pairs for ranking
        pair_vals = [found[f"{c1}{c2}"][0] for c1, c2 in chain_pairs if f"{c1}{c2}" in found]
        scores["ipsae"] = max(pair_vals) if pair_vals else None
        scores["ipsae_d0chn"] = None
        scores["ipsae_d0dom"] = None

    return scores


def _empty_ipsae_scores(chain_pairs=None) -> dict:
    """Return an empty scores dict matching the chain_pairs structure."""
    if chain_pairs is None or len(chain_pairs) == 1:
        return {"ipsae": None, "ipsae_d0chn": None, "ipsae_d0dom": None}
    scores = {"ipsae": None, "ipsae_d0chn": None, "ipsae_d0dom": None}
    for c1, c2 in chain_pairs:
        suffix = f"_{c1}{c2}"
        scores[f"ipsae{suffix}"] = None
        scores[f"ipsae_d0chn{suffix}"] = None
        scores[f"ipsae_d0dom{suffix}"] = None
    return scores


def run_ipsae(pae_file: str, struct_file: str, pae_cutoff: int = 15,
              dist_cutoff: int = 15, chain_pairs=None) -> dict:
    """Run ipSAE scoring on a single prediction."""
    if not Path(pae_file).exists() or not Path(struct_file).exists():
        return _empty_ipsae_scores(chain_pairs)

    try:
        result = subprocess.run(
            [sys.executable, str(IPSAE_SCRIPT), pae_file, struct_file,
             str(pae_cutoff), str(dist_cutoff)],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            return _empty_ipsae_scores(chain_pairs)

        struct_path = Path(struct_file)
        pae_str = f"{int(pae_cutoff):02d}"
        dist_str = f"{int(dist_cutoff):02d}"
        score_file = struct_path.parent / f"{struct_path.stem}_{pae_str}_{dist_str}.txt"

        if score_file.exists():
            with open(score_file) as f:
                scores = _parse_ipsae_lines(f, chain_pairs=chain_pairs)
            if scores.get("ipsae") is not None:
                return scores

        if result.stdout:
            scores = _parse_ipsae_lines(result.stdout.splitlines(), chain_pairs=chain_pairs)
            if scores.get("ipsae") is not None:
                return scores

        return _empty_ipsae_scores(chain_pairs)

    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  ipSAE error: {e}")
        return _empty_ipsae_scores(chain_pairs)


def main():
    parser = argparse.ArgumentParser(description="Score designs with ipSAE + Boltz metrics.")
    parser.add_argument("--boltz_dir", required=True, help="Boltz-2 output directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--run_ipsae", action="store_true", help="Run ipSAE scoring (slower)")
    parser.add_argument("--pae_cutoff", type=int, default=15, help="PAE cutoff for ipSAE")
    parser.add_argument("--dist_cutoff", type=int, default=15, help="Distance cutoff for ipSAE")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers for ipSAE (0 = auto, 1 = sequential)")
    parser.add_argument("--chain_pairs", type=str, default=None,
                        help="Chain pairs for ipSAE scoring, e.g. 'AB,AC' (default: AB)")
    args = parser.parse_args()

    chain_pairs = None
    if args.chain_pairs:
        chain_pairs = [(p[0], p[1]) for p in args.chain_pairs.split(",")]
        print(f"ipSAE chain pairs: {chain_pairs}")

    boltz_dir = Path(args.boltz_dir)
    print(f"Scanning Boltz outputs in: {boltz_dir}")
    results = find_boltz_outputs(boltz_dir)
    print(f"Found {len(results)} predictions")

    if args.run_ipsae:
        scoreable = [(i, r) for i, r in enumerate(results) if r["pae_file"] and r["structure_file"]]
        n_workers = args.workers if args.workers > 0 else min(len(scoreable), multiprocessing.cpu_count())
        print(f"\nRunning ipSAE scoring (PAE={args.pae_cutoff}, dist={args.dist_cutoff}, workers={n_workers})...")

        ipsae_fn = partial(run_ipsae, pae_cutoff=args.pae_cutoff, dist_cutoff=args.dist_cutoff,
                           chain_pairs=chain_pairs)

        if n_workers > 1 and len(scoreable) > 1:
            work_items = [(r["pae_file"], r["structure_file"]) for _, r in scoreable]
            with multiprocessing.Pool(n_workers) as pool:
                ipsae_results = pool.starmap(ipsae_fn, work_items)
            for (idx, r), scores in zip(scoreable, ipsae_results):
                r.update(scores)
                print(f"  [{idx+1}/{len(results)}] {r['design_name']}/{r['sample_name']} "
                      f"ipSAE={scores.get('ipsae', 'N/A')}")
        else:
            for idx, r in scoreable:
                print(f"  [{idx+1}/{len(results)}] {r['design_name']}/{r['sample_name']}")
                scores = ipsae_fn(r["pae_file"], r["structure_file"])
                r.update(scores)

        empty = _empty_ipsae_scores(chain_pairs)
        for r in results:
            if "ipsae" not in r:
                r.update(empty)

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
    ]
    if chain_pairs and len(chain_pairs) > 1:
        for c1, c2 in chain_pairs:
            fieldnames.extend([f"ipsae_{c1}{c2}", f"ipsae_d0chn_{c1}{c2}", f"ipsae_d0dom_{c1}{c2}"])
    fieldnames.extend([
        "confidence_score", "iptm", "protein_iptm",
        "complex_plddt", "complex_iplddt",
        "ptm", "complex_pde", "complex_ipde",
        "structure_file", "pae_file",
    ])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")
    print(f"\nTop 10 designs:")
    header = f"{'Rank':<6}{'Design':<35}{'ipSAE':>8}"
    if chain_pairs and len(chain_pairs) > 1:
        for c1, c2 in chain_pairs:
            header += f"{'ip'+c1+c2:>8}"
    header += f"{'ipTM':>8}{'ipLDDT':>8}{'conf':>8}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results[:10], 1):
        ipsae = f"{r.get('ipsae'):>8.4f}" if r.get("ipsae") is not None else "     N/A"
        row = f"{i:<6}{r['design_name']:<35}{ipsae}"
        if chain_pairs and len(chain_pairs) > 1:
            for c1, c2 in chain_pairs:
                val = r.get(f"ipsae_{c1}{c2}")
                row += f"{val:>8.4f}" if val is not None else "     N/A"
        row += f"{r['iptm']:>8.4f}{r['complex_iplddt']:>8.2f}{r['confidence_score']:>8.4f}"
        print(row)


if __name__ == "__main__":
    main()
