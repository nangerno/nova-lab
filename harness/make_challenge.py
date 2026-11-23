"""Write a frozen challenge input.json for local A/B runs.

Default fixture mirrors Blueprint epoch 20779 accessions (F6T8J6 + two antitargets)
with molecule_validation settings from nova-blueprint/config/config.yaml.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "challenges" / "fixed_v1" / "input.json"

FILTERS = {
    "max_heavy_atoms": 50,
    "min_heavy_atoms": 20,
    "max_molecular_weight": 700,
    "min_molecular_weight": 100,
    "max_hbd": 5,
    "max_hba": 10,
    "max_logp": 7.0,
    "min_rotatable_bonds": 1,
    "max_rotatable_bonds": 10,
    "min_rings": 1,
    "max_ring_size": 12,
    "max_spiro_atoms": 2,
    "max_aromatic_rings": 5,
    "max_aliphatic_rings": 5,
    "max_fsp3": 0.8,
    "max_triple_bonds": 1,
    "max_amide_bonds": 5,
    "max_stereocenters": 6,
    "max_halogens": 6,
    "max_iodine": 1,
    "max_bromine": 2,
    "max_fluorine": 9,
    "max_phosphorus": 2,
    "max_nitrogen": 9,
    "max_oxygen": 9,
    "max_sulfur": 2,
    "max_aromatic_nitrogen": 6,
    "max_aromatic_carbon": 28,
    "max_aromatic_fraction": 0.92,
    "min_heteroatom_fraction": 0.05,
    "max_heteroatom_fraction": 0.65,
    "max_bertz_ct": 1950,
    "max_kappa1": 35,
    "max_kappa2": 17,
    "max_kappa3": 12,
    "max_nitrile": 2,
    "max_hydroxyl": 5,
}


def _fasta_lines(text: str) -> str:
    return "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))


def fetch_uniprotkb(accession: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    with urllib.request.urlopen(url, timeout=60) as response:
        seq = _fasta_lines(response.read().decode())
    if len(seq) < 20:
        raise RuntimeError(f"UniProtKB {accession} returned empty/short sequence")
    return seq


def fetch_uniparc(upi: str) -> str:
    url = f"https://rest.uniprot.org/uniparc/{upi}.fasta"
    with urllib.request.urlopen(url, timeout=60) as response:
        seq = _fasta_lines(response.read().decode())
    if len(seq) < 20:
        raise RuntimeError(f"UniParc {upi} returned empty/short sequence")
    return seq


def build_fixed_v1() -> dict:
    # Epoch 20779 dashboard accessions. A0A8K1X527 is deleted in UniProtKB;
    # sequence recovered from UniParc UPI001E74D9DE.
    target = fetch_uniprotkb("F6T8J6")
    anti = [
        fetch_uniprotkb("A0A8C9UVP0"),
        fetch_uniparc("UPI001E74D9DE"),
    ]
    return {
        "config": {
            "antitarget_weight": 0.9,
            "entropy_min_threshold": 0.35,
            "tanimoto_max_threshold": 0.9,
            "num_molecules": 100,
            "filters": FILTERS,
        },
        "challenge": {
            "target_sequences": [target],
            "antitarget_sequences": anti,
            "meta": {
                "fixture": "fixed_v1",
                "source_epoch": 20779,
                "target_accession": "F6T8J6",
                "antitarget_accessions": ["A0A8C9UVP0", "A0A8K1X527"],
                "antitarget_uniparc": ["", "UPI001E74D9DE"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--from-json",
        type=Path,
        help="Optional pre-fetched sequences JSON with keys target, antis",
    )
    args = parser.parse_args()

    if args.from_json:
        raw = json.loads(args.from_json.read_text())
        payload = {
            "config": {
                "antitarget_weight": 0.9,
                "entropy_min_threshold": 0.35,
                "tanimoto_max_threshold": 0.9,
                "num_molecules": 100,
                "filters": FILTERS,
            },
            "challenge": {
                "target_sequences": [raw["target"]],
                "antitarget_sequences": list(raw["antis"]),
                "meta": {"fixture": "fixed_v1", "source": str(args.from_json)},
            },
        }
    else:
        payload = build_fixed_v1()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    n_t = len(payload["challenge"]["target_sequences"][0])
    n_a = [len(s) for s in payload["challenge"]["antitarget_sequences"]]
    print(f"wrote {args.out} target_len={n_t} antitarget_lens={n_a}")


if __name__ == "__main__":
    main()
