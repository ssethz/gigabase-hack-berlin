#!/usr/bin/env python3
"""Extract and rank ProteinMPNN sequence scores from FASTA outputs.

Parses all MPNN FASTA files in a directory tree, extracts per-sequence
scores (score, global_score, seq_recovery), and writes a ranked CSV.
"""

import argparse
import csv
import re
from pathlib import Path


def parse_mpnn_fastas(mpnn_dir: Path) -> list[dict]:
    """Parse all MPNN FASTA files and extract scores."""
    results = []

    for fasta_path in sorted(mpnn_dir.glob("**/*.fa")):
        backbone_name = fasta_path.stem
        seq_idx = -1

        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith(">"):
                    if seq_idx >= 0 and results:
                        chains = line.split("/")
                        results[-1]["designed_seq"] = chains[0] if len(chains) == 1 else line
                        designed = [c for i, c in enumerate(chains)
                                    if results[-1].get("_designed_idx") is None or i in results[-1].get("_designed_idx", [])]
                        results[-1]["designed_seq_length"] = sum(len(c) for c in chains)
                    continue

                seq_idx += 1
                header = line[1:]

                entry = {
                    "backbone": backbone_name,
                    "seq_idx": seq_idx,
                    "name": f"{backbone_name}_seq{seq_idx}",
                    "score": None,
                    "global_score": None,
                    "seq_recovery": None,
                    "temperature": None,
                    "sample": None,
                    "designed_seq": "",
                    "designed_seq_length": 0,
                }

                if seq_idx == 0:
                    m = re.search(r"score=([\d.]+)", header)
                    if m:
                        entry["score"] = float(m.group(1))
                    m = re.search(r"global_score=([\d.]+)", header)
                    if m:
                        entry["global_score"] = float(m.group(1))
                    entry["seq_recovery"] = None
                    entry["temperature"] = None
                    entry["sample"] = 0
                else:
                    m = re.search(r"score=([\d.]+)", header)
                    if m:
                        entry["score"] = float(m.group(1))
                    m = re.search(r"global_score=([\d.]+)", header)
                    if m:
                        entry["global_score"] = float(m.group(1))
                    m = re.search(r"seq_recovery=([\d.]+)", header)
                    if m:
                        entry["seq_recovery"] = float(m.group(1))
                    m = re.search(r"T=([\d.]+)", header)
                    if m:
                        entry["temperature"] = float(m.group(1))
                    m = re.search(r"sample=(\d+)", header)
                    if m:
                        entry["sample"] = int(m.group(1))

                results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract and rank ProteinMPNN sequence scores."
    )
    parser.add_argument("--mpnn_dir", required=True,
                        help="MPNN output directory (searched recursively for .fa files)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--skip_original", action="store_true",
                        help="Skip the original (seq_idx=0) entry per backbone")
    args = parser.parse_args()

    mpnn_dir = Path(args.mpnn_dir)
    print(f"Scanning MPNN outputs in: {mpnn_dir}")

    results = parse_mpnn_fastas(mpnn_dir)
    print(f"Found {len(results)} total sequences")

    if args.skip_original:
        results = [r for r in results if r["seq_idx"] != 0]
        print(f"After skipping originals: {len(results)} designed sequences")

    results.sort(key=lambda x: x["score"] if x["score"] is not None else float("inf"))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank", "name", "backbone", "seq_idx", "sample",
        "score", "global_score", "seq_recovery", "temperature",
        "designed_seq_length", "designed_seq",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
            writer.writerow(r)

    print(f"\nResults written to: {output_path}")
    print(f"\nTop 20 by MPNN score (lower = better):")
    print(f"{'Rank':<6}{'Name':<30}{'Score':>8}{'Global':>8}{'Recovery':>10}")
    print("-" * 65)
    for r in results[:20]:
        score = f"{r['score']:>8.4f}" if r['score'] is not None else "     N/A"
        gscore = f"{r['global_score']:>8.4f}" if r['global_score'] is not None else "     N/A"
        rec = f"{r['seq_recovery']:>10.4f}" if r['seq_recovery'] is not None else "       N/A"
        print(f"{r['rank']:<6}{r['name']:<30}{score}{gscore}{rec}")


if __name__ == "__main__":
    main()
