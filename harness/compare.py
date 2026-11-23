"""Compare baseline vs variant run artifacts.

Primary metric: portfolio mean selectivity from miner_ledger last_write.value
(fallback: mean of scored.jsonl y for molecules in result.json when ledger missing).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _scored_index(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "name" in row and "y" in row:
                out[row["name"]] = float(row["y"])
    return out


def summarize(run_dir: Path) -> dict:
    ledger = _load_json(run_dir / "miner_ledger.json") or {}
    result = _load_json(run_dir / "result.json") or {}
    molecules = list(result.get("molecules") or [])
    scored = _scored_index(run_dir / "scored.jsonl")

    last = ledger.get("last_write") or {}
    portfolio_mean = last.get("value")
    if portfolio_mean is None and molecules:
        values = [scored[name] for name in molecules if name in scored]
        portfolio_mean = sum(values) / len(values) if values == molecules or (
            values and len(values) == len(molecules)
        ) else None
        if portfolio_mean is None and values:
            portfolio_mean = sum(values) / len(values)

    pairs_requested = float(ledger.get("pairs_requested") or 0)
    oracle_seconds = float(ledger.get("oracle_seconds") or 0)
    pairs_per_sec = (pairs_requested / oracle_seconds) if oracle_seconds > 0 else None

    return {
        "portfolio_mean": portfolio_mean,
        "entropy": last.get("entropy"),
        "scored": ledger.get("scored"),
        "requests": ledger.get("http_requests"),
        "oracle_seconds": oracle_seconds,
        "pairs_requested": pairs_requested,
        "pairs_per_sec": pairs_per_sec,
        "writes": ledger.get("writes"),
        "fallback": ledger.get("fallback"),
        "errors": ledger.get("errors"),
        "dup_inchikey": ledger.get("dup_inchikey"),
        "products_invalid": ledger.get("products_invalid"),
        "elapsed": ledger.get("elapsed"),
        "n_result": len(molecules),
        "scored_rxn": ledger.get("scored_rxn"),
        "sources": ledger.get("sources"),
        "hyper": ledger.get("hyper"),
    }


def compare(baseline: dict, variant: dict) -> dict:
    def delta(key: str):
        b, v = baseline.get(key), variant.get(key)
        if isinstance(b, (int, float)) and isinstance(v, (int, float)):
            return v - b
        return None

    primary_b = baseline.get("portfolio_mean")
    primary_v = variant.get("portfolio_mean")
    decision = "inconclusive"
    if isinstance(primary_b, (int, float)) and isinstance(primary_v, (int, float)):
        if primary_v > primary_b + 1e-6:
            decision = "variant_wins"
        elif primary_v < primary_b - 1e-6:
            decision = "baseline_wins"
        else:
            decision = "tie"

    return {
        "decision": decision,
        "primary_delta": delta("portfolio_mean"),
        "deltas": {
            "portfolio_mean": delta("portfolio_mean"),
            "scored": delta("scored"),
            "oracle_seconds": delta("oracle_seconds"),
            "pairs_per_sec": delta("pairs_per_sec"),
            "entropy": delta("entropy"),
            "errors": delta("errors"),
            "dup_inchikey": delta("dup_inchikey"),
            "products_invalid": delta("products_invalid"),
        },
        "baseline": baseline,
        "variant": variant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id_dir", type=Path, help="runs/<id> containing baseline/ and variant/")
    parser.add_argument("--out", type=Path, help="optional path for comparison.json")
    args = parser.parse_args()

    baseline = summarize(args.run_id_dir / "baseline")
    variant = summarize(args.run_id_dir / "variant")
    report = compare(baseline, variant)
    text = json.dumps(report, indent=2)
    print(text)
    out = args.out or (args.run_id_dir / "comparison.json")
    out.write_text(text + "\n")
    print(f"decision={report['decision']} primary_delta={report['primary_delta']}", flush=True)


if __name__ == "__main__":
    main()
