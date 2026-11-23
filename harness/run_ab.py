"""Run baseline and/or variant miners under the same seed, challenge, and budget.

Usage examples:
  # CPU smoke with mock oracle (starts mock if ORACLE_URL unreachable)
  python harness/run_ab.py --budget 30 --seed 1 --labels baseline --mock

  # Full short A/B against a running local nova-oracle
  export ORACLE_URL=http://127.0.0.1:8080
  export ORACLE_TOKEN=local-dev-token
  python harness/run_ab.py --budget 600 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

from broker import Broker, ensure_proteins  # noqa: E402
from compare import compare, summarize  # noqa: E402


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ if the key is not already set."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _python() -> str:
    """Prefer the lab's 3.12 venv so miner deps match the sandbox."""
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    marker = ROOT / ".python-path"
    if marker.is_file():
        path = marker.read_text().strip()
        if path and Path(path).is_file():
            return path
    return sys.executable


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _wait_http(url: str, token: str, timeout: float = 2.0) -> bool:
    import http.client
    from urllib.parse import urlparse

    parsed = urlparse(url)
    opener = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = None
    try:
        conn = opener(parsed.netloc, timeout=timeout)
        conn.request("GET", "/health", headers={"Authorization": f"Bearer {token}"})
        resp = conn.getresponse()
        resp.read()
        return resp.status == 200
    except OSError:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _start_mock(host: str, port: int, token: str) -> subprocess.Popen:
    return subprocess.Popen(
        [_python(), str(HARNESS / "mock_oracle.py"), "--host", host, "--port", str(port), "--token", token],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _prepare_workdir(label_dir: Path, miner_src: Path, challenge: Path) -> Path:
    work = label_dir / "workspace"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(miner_src, work, ignore=shutil.ignore_patterns("FROZEN.txt", "__pycache__", "*.pyc"))
    shutil.copy2(challenge, work / "input.json")
    (label_dir / "output").mkdir(parents=True, exist_ok=True)
    return work


def _run_one(
    *,
    label: str,
    miner_src: Path,
    challenge: Path,
    run_dir: Path,
    budget: float,
    seed: int,
    workers: int | None,
    socket_path: Path,
    epoch: str,
) -> dict:
    label_dir = run_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    work = _prepare_workdir(label_dir, miner_src, challenge)
    output = label_dir / "output"
    env = os.environ.copy()
    env["WORKDIR"] = str(work)
    env["OUTPUT_DIR"] = str(output)
    env["ORACLE_SOCKET"] = str(socket_path)
    env["NOVA_TIME_BUDGET_SEC"] = str(budget)
    env["NOVA_SEED"] = str(seed)
    if workers is not None:
        env["NOVA_WORKERS"] = str(workers)
    # Keep BLAS single-threaded as the champion does.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"

    log_path = label_dir / "miner.log"
    started = time.time()
    with Broker(socket_path, miner=f"local-{label}", epoch=epoch):
        with log_path.open("w") as log:
            proc = subprocess.Popen(
                [_python(), "miner.py"],
                cwd=str(work),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                # Hard kill a little after the miner budget so a stuck process cannot hang A/B.
                proc.wait(timeout=budget + 90)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                log.write("\n[harness] killed miner after budget+grace\n")

    # Copy artifacts next to log for easy compare paths
    for name in ("result.json", "miner_ledger.json", "scored.jsonl"):
        src = output / name
        if src.is_file():
            shutil.copy2(src, label_dir / name)

    summary = summarize(label_dir)
    summary["label"] = label
    summary["exit_wall_s"] = round(time.time() - started, 1)
    summary["returncode"] = proc.returncode
    (label_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, default=ROOT / "challenges" / "fixed_v1" / "input.json")
    parser.add_argument("--budget", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--labels", nargs="+", default=["baseline", "variant"], choices=["baseline", "variant"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mock", action="store_true", help="Start mock_oracle if needed")
    parser.add_argument("--oracle-url", default=os.environ.get("ORACLE_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--oracle-token", default=os.environ.get("ORACLE_TOKEN", "local-dev-token"))
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()
    _load_dotenv(ROOT / ".env")
    # Re-resolve defaults after .env so CLI/env still win when explicitly set.
    if "--oracle-url" not in sys.argv:
        args.oracle_url = os.environ.get("ORACLE_URL", args.oracle_url)
    if "--oracle-token" not in sys.argv:
        args.oracle_token = os.environ.get("ORACLE_TOKEN", args.oracle_token)

    if not args.challenge.is_file():
        raise SystemExit(f"challenge missing: {args.challenge} (run harness/make_challenge.py)")

    os.environ["ORACLE_URL"] = args.oracle_url
    os.environ["ORACLE_TOKEN"] = args.oracle_token
    # Tunnel-friendly defaults unless the caller already set them.
    os.environ.setdefault("NOVA_BATCH", "8")
    os.environ.setdefault("ORACLE_SCORE_CHUNK", "8")

    mock_proc = None
    if args.mock and not _wait_http(args.oracle_url, args.oracle_token):
        from urllib.parse import urlparse

        parsed = urlparse(args.oracle_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080
        print(f"starting mock oracle on {host}:{port}", flush=True)
        mock_proc = _start_mock(host, port, args.oracle_token)
        for _ in range(50):
            if _wait_http(args.oracle_url, args.oracle_token):
                break
            time.sleep(0.1)
        else:
            raise SystemExit("mock oracle failed to become healthy")

    if not _wait_http(args.oracle_url, args.oracle_token):
        raise SystemExit(f"oracle health failed at {args.oracle_url}")

    challenge = json.loads(args.challenge.read_text())
    sequences = list(challenge["challenge"]["target_sequences"]) + list(
        challenge["challenge"].get("antitarget_sequences") or []
    )
    labels_meta = ["target"] + [
        f"antitarget_{i}" for i in range(len(challenge["challenge"].get("antitarget_sequences") or []))
    ]

    if not args.skip_register:
        ensure_proteins(sequences, labels_meta)
    else:
        print("skipping protein registration (--skip-register)", flush=True)

    run_id = args.run_id or f"{_utc_id()}_seed{args.seed}_b{int(args.budget)}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "seed": args.seed,
        "budget": args.budget,
        "challenge": str(args.challenge),
        "labels": args.labels,
        "oracle_url": args.oracle_url,
        "mock": bool(mock_proc),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    summaries = {}
    try:
        for label in args.labels:
            miner_src = ROOT / label
            if not (miner_src / "miner.py").is_file():
                raise SystemExit(f"missing miner at {miner_src}")
            sock = run_dir / f"oracle_{label}.sock"
            print(f"==> running {label} budget={args.budget}s seed={args.seed}", flush=True)
            summaries[label] = _run_one(
                label=label,
                miner_src=miner_src,
                challenge=args.challenge,
                run_dir=run_dir,
                budget=args.budget,
                seed=args.seed,
                workers=args.workers,
                socket_path=sock,
                epoch=run_id,
            )
            print(
                f"    {label}: portfolio_mean={summaries[label].get('portfolio_mean')} "
                f"scored={summaries[label].get('scored')} elapsed={summaries[label].get('elapsed')}",
                flush=True,
            )
    finally:
        if mock_proc is not None:
            mock_proc.terminate()
            try:
                mock_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mock_proc.kill()

    if "baseline" in summaries and "variant" in summaries:
        report = compare(summaries["baseline"], summaries["variant"])
        (run_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"decision={report['decision']} primary_delta={report['primary_delta']}",
            flush=True,
        )
    print(f"artifacts: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
