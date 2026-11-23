"""Unix-socket oracle broker for local A/B runs.

Same contract as nova-blueprint/sandbox/broker.py: miner talks to ORACLE_SOCKET;
this process stamps miner/epoch and forwards to ORACLE_URL with ORACLE_TOKEN.
"""
from __future__ import annotations

import http.client
import json
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

TIMEOUT_S = 900
REGISTER_TIMEOUT_S = 1800
MAX_PREDICTIONS = 512
# Cloudflare trycloudflare.com often returns 524 if the origin is busy > ~100s.
RETRYABLE = {502, 503, 504, 520, 522, 524}


def _oracle_url() -> str:
    return os.environ.get("ORACLE_URL", "http://127.0.0.1:8080")


def _oracle_token() -> str:
    return os.environ.get("ORACLE_TOKEN", "local-dev-token")


def _post(path: str, body: dict, timeout: float) -> tuple[int, bytes]:
    # Read env at call time so .env loaded in run_ab.main() is visible.
    url = urlparse(_oracle_url())
    opener = (
        http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    )
    connection = opener(url.netloc, timeout=timeout)
    try:
        connection.request(
            "POST",
            path,
            json.dumps(body),
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_oracle_token()}",
            },
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _post_retry(
    path: str,
    body: dict,
    timeout: float,
    *,
    attempts: int = 6,
    label: str = "",
) -> tuple[int, bytes]:
    """Retry transient Cloudflare / gateway failures."""
    last_status, last_payload = 0, b""
    for attempt in range(1, attempts + 1):
        try:
            status, payload = _post(path, body, timeout)
        except (TimeoutError, ConnectionError, OSError) as exc:
            wait = min(60.0, 2.0 ** attempt)
            print(
                f"oracle {label or path} attempt {attempt}/{attempts} "
                f"transport error: {type(exc).__name__}: {exc}; retry in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
            continue
        if status == 200 or status not in RETRYABLE:
            return status, payload
        last_status, last_payload = status, payload
        wait = min(60.0, 2.0 ** attempt)
        print(
            f"oracle {label or path} attempt {attempt}/{attempts} "
            f"HTTP {status}; retry in {wait:.0f}s",
            flush=True,
        )
        time.sleep(wait)
    return last_status, last_payload


def register_proteins(sequences: list[str], labels: list[str] | None = None) -> list[dict]:
    """Register one sequence at a time (cheaper under Cloudflare 524 limits)."""
    labels = list(labels) if labels is not None else [None] * len(sequences)
    if len(labels) < len(sequences):
        labels = labels + [None] * (len(sequences) - len(labels))
    proteins: list[dict] = []
    for i, seq in enumerate(sequences):
        body: dict = {"sequences": [seq]}
        if labels[i] is not None:
            body["labels"] = [labels[i]]
        tag = f"proteins[{i+1}/{len(sequences)}]"
        print(f"registering {tag} residues={len(seq)}…", flush=True)
        status, payload = _post_retry(
            "/v1/proteins", body, REGISTER_TIMEOUT_S, attempts=8, label=tag
        )
        if status != 200:
            raise RuntimeError(
                f"oracle refused {tag} ({status}): {payload[:200]!r}. "
                f"If this is Cloudflare 524, restart nova-oracle on the GPU host "
                f"or set ORACLE_URL to a direct (non-tunnel) address."
            )
        chunk = json.loads(payload)["proteins"]
        proteins.extend(chunk)
        cached = chunk[0].get("cached") if chunk else None
        print(f"  {tag} ok cached={cached}", flush=True)
    return proteins


def proteins_ready(sequences: list[str]) -> bool:
    """True if a tiny score against these sequences succeeds (already prepared)."""
    status, payload = _post_retry(
        "/v1/score",
        {
            "miner": "nova-lab-preflight",
            "epoch": "preflight",
            "targets": list(sequences),
            "smiles": ["CCO"],
        },
        timeout=180,
        attempts=2,
        label="preflight-score",
    )
    if status == 200:
        return True
    if status == 409:
        return False
    raise RuntimeError(
        f"oracle preflight score failed ({status}): {payload[:200]!r}. "
        f"Health may be up while /v1/score is wedged behind Cloudflare — "
        f"restart the GPU oracle or use a direct ORACLE_URL."
    )


def ensure_proteins(sequences: list[str], labels: list[str] | None = None) -> None:
    """Skip registration when targets are already warm; otherwise register."""
    print("preflight: checking whether proteins are already prepared…", flush=True)
    try:
        if proteins_ready(sequences):
            print("preflight: proteins ready (skip register)", flush=True)
            return
    except RuntimeError as exc:
        # Soft-fallback: still try register; caller sees register errors if both fail.
        print(f"preflight: score probe failed ({exc}); will try register", flush=True)
    print(f"registering {len(sequences)} proteins with oracle…", flush=True)
    register_proteins(sequences, labels)


def _score_chunk(miner: str, epoch: str, targets: list, smiles: list) -> tuple[int, bytes]:
    """Score one smiles chunk with retries; used by the unix-socket broker."""
    return _post_retry(
        "/v1/score",
        {
            "miner": miner,
            "epoch": epoch,
            "targets": list(targets),
            "smiles": list(smiles),
        },
        TIMEOUT_S,
        attempts=5,
        label=f"score n={len(smiles)}",
    )


def _max_smiles_per_request(n_targets: int) -> int:
    """Keep Cloudflare-friendly batches. Override with ORACLE_SCORE_CHUNK."""
    raw = os.environ.get("ORACLE_SCORE_CHUNK")
    if raw:
        return max(1, int(raw))
    # Default: at most 24 pairs so trycloudflare.com is less likely to 524.
    return max(1, 24 // max(1, n_targets))


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        if self.path != "/v1/score":
            return self._send(404, b'{"detail":"not found"}')
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            targets, smiles = body["targets"], body["smiles"]
        except (ValueError, KeyError, TypeError):
            return self._send(400, b'{"detail":"bad request"}')

        chunk = _max_smiles_per_request(len(targets))
        if len(smiles) <= chunk:
            status, payload = _score_chunk(
                self.server.miner, self.server.epoch, targets, smiles
            )
            return self._send(status, payload)

        # Split large miner batches into tunnel-safe chunks; stitch positional results.
        results = []
        for start in range(0, len(smiles), chunk):
            part = smiles[start : start + chunk]
            status, payload = _score_chunk(
                self.server.miner, self.server.epoch, targets, part
            )
            if status != 200:
                return self._send(status, payload)
            results.extend(json.loads(payload)["results"])
        merged = {
            "request_id": "broker-chunked",
            "elapsed_s": 0.0,
            "predictions": sum(
                len(row.get("scores") or []) for row in results if isinstance(row, dict)
            ),
            "results": results,
        }
        self._send(200, json.dumps(merged).encode())

    def _send(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def get_request(self):
        return self.socket.accept()[0], ("local", 0)

    def handle_error(self, request, client_address) -> None:
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


class Broker:
    def __init__(self, socket_path: str | Path, miner: str, epoch: str):
        self.socket_path = str(socket_path)
        self.miner = miner
        self.epoch = epoch
        self._server: _Server | None = None

    def __enter__(self) -> "Broker":
        Path(self.socket_path).unlink(missing_ok=True)
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)
        server = _Server(self.socket_path, _Handler)
        try:
            server.miner = self.miner
            server.epoch = self.epoch
            os.chmod(self.socket_path, 0o666)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        except BaseException:
            server.server_close()
            Path(self.socket_path).unlink(missing_ok=True)
            raise
        self._server = server
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._server is not None:
                self._server.shutdown()
        finally:
            try:
                if self._server is not None:
                    self._server.server_close()
            finally:
                Path(self.socket_path).unlink(missing_ok=True)
