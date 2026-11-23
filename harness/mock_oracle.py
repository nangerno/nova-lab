"""CPU mock oracle for harness wiring tests (no GPU / no Nesso weights).

Speaks the same HTTP surface the broker expects:
  POST /v1/proteins  Authorization: Bearer …
  POST /v1/score     Authorization: Bearer …
  GET  /health

Scores are deterministic hashes of (smiles, target) — useful for A/B plumbing,
not for ranking real chemistry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _affinity(smiles: str, target: str) -> float:
    digest = hashlib.sha256(f"{smiles}|{target}".encode()).hexdigest()
    # Map to roughly [-2, 4] log10(IC50/uM)-like range.
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return -2.0 + 6.0 * unit


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    token = "local-dev-token"
    prepared: set[str] = set()

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._json(200, {"alive": 1, "total": 1, "gpus": [], "rss_gb": 0.0})
        self._json(404, {"detail": "not found"})

    def do_POST(self) -> None:
        if not self._auth_ok():
            return self._json(401, {"detail": "unauthorized"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return self._json(400, {"detail": "bad request"})

        if self.path == "/v1/proteins":
            sequences = body.get("sequences") or []
            labels = body.get("labels") or [None] * len(sequences)
            proteins = []
            for i, seq in enumerate(sequences):
                if not isinstance(seq, str) or not seq:
                    return self._json(422, {"detail": "bad sequence"})
                self.prepared.add(seq)
                proteins.append(
                    {
                        "sequence_hash": hashlib.sha1(seq.encode()).hexdigest()[:16],
                        "residues": len(seq),
                        "depth": 0,
                        "cached": False,
                        "label": labels[i] if i < len(labels) else None,
                    }
                )
            return self._json(200, {"proteins": proteins, "warm_s": 0.0, "elapsed_s": 0.0})

        if self.path == "/v1/score":
            targets = body.get("targets") or []
            smiles = body.get("smiles") or []
            if not targets or not smiles:
                return self._json(422, {"detail": "targets and smiles required"})
            for t in targets:
                if t not in self.prepared:
                    return self._json(409, {"detail": "target not prepared"})
            if len(targets) * len(smiles) > 512:
                return self._json(413, {"detail": "batch too large"})
            results = []
            for smi in smiles:
                scores = []
                for t in targets:
                    scores.append(
                        {
                            "affinity_pred_value": _affinity(smi, t),
                            "mock": True,
                        }
                    )
                results.append({"smiles": smi, "scores": scores})
            return self._json(
                200,
                {
                    "request_id": "mock",
                    "elapsed_s": 0.01,
                    "predictions": len(targets) * len(smiles),
                    "results": results,
                },
            )

        self._json(404, {"detail": "not found"})

    def _auth_ok(self) -> bool:
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {self.token}"

    def _json(self, status: int, obj: dict) -> None:
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--token", default="local-dev-token")
    args = parser.parse_args()
    Handler.token = args.token
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock oracle on http://{args.host}:{args.port} token={args.token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
