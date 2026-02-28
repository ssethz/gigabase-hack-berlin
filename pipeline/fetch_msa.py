#!/usr/bin/env python3
"""Pre-fetch MSA a3m files for fixed/target chain sequences via ColabFold API.

Reads chain_info JSON to identify unique fixed-chain sequences, fetches MSA
for each via the ColabFold MMseqs2 server, and writes a mapping file so
downstream Boltz YAML generation can reference the pre-computed a3m files.
"""

import argparse
import hashlib
import json
import time
import urllib.request
import urllib.error
from pathlib import Path


COLABFOLD_TICKET_URL = "https://api.colabfold.com/ticket/msa"
COLABFOLD_RESULT_URL = "https://api.colabfold.com/result/ticket/"
POLL_INTERVAL = 5
MAX_POLL_TIME = 600


def sequence_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()[:12]


def fetch_msa_colabfold(sequence: str, output_path: Path, timeout: int = MAX_POLL_TIME) -> Path:
    """Fetch MSA for a single sequence via the ColabFold MMseqs2 API.

    Submits a ticket, polls until ready, downloads the a3m result.
    """
    payload = json.dumps({"q": f">query\n{sequence}", "mode": "all"}).encode()
    req = urllib.request.Request(
        COLABFOLD_TICKET_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(f"  Submitting MSA request ({len(sequence)} aa)...")
    resp = urllib.request.urlopen(req)
    ticket = json.loads(resp.read().decode())
    ticket_id = ticket["id"]
    print(f"  Ticket: {ticket_id}")

    start = time.time()
    while time.time() - start < timeout:
        time.sleep(POLL_INTERVAL)
        try:
            status_resp = urllib.request.urlopen(f"{COLABFOLD_RESULT_URL}{ticket_id}")
            status = json.loads(status_resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise

        if status.get("status") == "COMPLETE":
            break
        if status.get("status") == "ERROR":
            raise RuntimeError(f"ColabFold MSA request failed: {status}")
        elapsed = int(time.time() - start)
        print(f"  Waiting... ({elapsed}s, status={status.get('status', 'unknown')})")
    else:
        raise TimeoutError(f"MSA fetch timed out after {timeout}s for ticket {ticket_id}")

    a3m_url = f"{COLABFOLD_RESULT_URL}{ticket_id}/0"
    a3m_resp = urllib.request.urlopen(a3m_url)
    a3m_content = a3m_resp.read()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(a3m_content)

    n_seqs = a3m_content.decode(errors="replace").count(">")
    print(f"  Saved: {output_path} ({n_seqs} sequences, {len(a3m_content)} bytes)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Pre-fetch MSA a3m files for fixed chains via ColabFold API."
    )
    parser.add_argument("--chain_info", required=True,
                        help="chain_info_merged.json from prepare_mpnn_inputs.py")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for a3m files and msa_map.json")
    parser.add_argument("--timeout", type=int, default=MAX_POLL_TIME,
                        help=f"Max seconds to wait per MSA request (default: {MAX_POLL_TIME})")
    args = parser.parse_args()

    with open(args.chain_info) as f:
        chain_info = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_seqs: dict[str, str] = {}
    for backbone_name, info in chain_info.items():
        fixed_chains = info.get("fixed_chains", [])
        chain_sequences = info.get("chain_sequences", {})
        for cid in fixed_chains:
            seq = chain_sequences.get(cid)
            if seq and seq not in unique_seqs:
                unique_seqs[seq] = sequence_hash(seq)

    print(f"Found {len(unique_seqs)} unique fixed-chain sequence(s) to fetch MSA for")

    msa_map: dict[str, str] = {}
    for seq, seq_hash in unique_seqs.items():
        a3m_path = output_dir / f"{seq_hash}.a3m"

        if a3m_path.exists() and a3m_path.stat().st_size > 0:
            print(f"  {seq_hash}: already cached ({a3m_path})")
            msa_map[seq] = str(a3m_path.resolve())
            continue

        print(f"Fetching MSA for {seq_hash} ({len(seq)} aa)...")
        try:
            fetch_msa_colabfold(seq, a3m_path, timeout=args.timeout)
            msa_map[seq] = str(a3m_path.resolve())
        except Exception as e:
            print(f"  ERROR fetching MSA for {seq_hash}: {e}")
            print(f"  This chain will fall back to msa: empty")

    map_path = output_dir / "msa_map.json"
    with open(map_path, "w") as f:
        json.dump(msa_map, f, indent=2)
    print(f"\nMSA map written: {map_path} ({len(msa_map)} entries)")


if __name__ == "__main__":
    main()
