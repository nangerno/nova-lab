# Local oracle bring-up

The champion miner uses `nova_miner.utils.oracle.Oracle(ORACLE_SOCKET)`.
Validators normally expose that socket via `nova-blueprint/sandbox/broker.py`.
This lab reuses the same pattern: broker → HTTP oracle with bearer token.

You do **not** need a Metanova-issued token. Self-host `nova-oracle` and choose any bearer string.

## Option A — mock (CPU, wiring only)

```bash
python3 harness/mock_oracle.py --port 8080 --token local-dev-token
# or let run_ab start it:
python3 harness/run_ab.py --mock --budget 30 --seed 1
```

Scores are deterministic hashes. Use this to verify deps, broker, and A/B plumbing.
Do **not** use mock scores to decide chemistry improvements.

## Option B — real Nesso (GPU)

1. Clone and provision:

```bash
git clone https://github.com/metanova-labs/nova-oracle.git
cd nova-oracle
# follow oracle/models/nesso/README.md and deploy/provision.sh for nesso
cp oracle/models/nesso/env.example .env
```

2. Set in `.env` (example):

```
ORACLE_MODEL=nesso
ORACLE_GPUS=0
ORACLE_HOST=127.0.0.1
ORACLE_PORT=8080
ORACLE_TOKEN=local-dev-token
```

3. Start:

```bash
./deploy/run.sh
python3 tests/smoke.py
```

4. Point the lab at it:

```bash
export ORACLE_URL=http://127.0.0.1:8080
export ORACLE_TOKEN=local-dev-token
cd /root/nova-lab
python3 harness/run_ab.py --budget 600 --seed 1
```

`run_ab.py` registers challenge proteins via `/v1/proteins` before each series.

## Cloudflare tunnels (`*.trycloudflare.com`)

`/health` can be fine while `/v1/proteins` and `/v1/score` return **HTTP 524**
(origin took too long). Nesso batches and first-time protein embedding often
exceed the tunnel’s ~100s proxy limit.

Mitigations already in the harness:

- register one protein at a time, with retries
- split score batches (`ORACLE_SCORE_CHUNK`, default 8)
- default `NOVA_BATCH=8` for miner requests
- skip register when a preflight score already succeeds

If **every** score/register 524s for minutes:

1. Restart `nova-oracle` on the GPU host (shards may be wedged).
2. Prefer a **direct** `ORACLE_URL=http://<gpu-ip>:8000` (no Cloudflare).
3. Re-run: `NOVA_BATCH=8 python harness/run_ab.py --budget 600 --seed 1`

## Contract reminders

- Pair cap: `len(targets) * len(smiles) ≤ 512` (same as sandbox broker).
- Score payload: `results[i]["scores"][t]` with `affinity_pred_value`.
- Miner selectivity: mean target pIC50 − `antitarget_weight` × mean antitarget pIC50,
  where pIC50 = clip(6 − affinity_pred_value, 1, 11).
