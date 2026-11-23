# nova-lab

Local A/B harness around the frozen NOVA Blueprint champion `nesso-selectivity-v2`.

## Layout

| path | role |
| --- | --- |
| `baseline/` | Exact champion copy. **Do not edit.** |
| `variant/` | Working tree for one experiment at a time |
| `harness/` | Broker, mock oracle, A/B runner, compare |
| `challenges/fixed_v1/` | Frozen `input.json` for a series |
| `runs/` | Timestamped baseline/variant outputs |
| `EXPERIMENTS.md` | Ordered backlog + promote/reject log |

## One-time setup

```bash
cd /root/nova-lab
bash scripts/setup_deps.sh          # creates .venv (Python 3.12) + molecules.sqlite
source .venv/bin/activate
python harness/make_challenge.py
```

## Oracle

Miners talk to a unix socket (`ORACLE_SOCKET`). The harness broker forwards to HTTP.

**CPU smoke (no GPU):**

```bash
source .venv/bin/activate
python harness/run_ab.py --mock --budget 30 --seed 1 --workers 2
```

**Real Nesso (GPU):** self-host [metanova-labs/nova-oracle](https://github.com/metanova-labs/nova-oracle) with `ORACLE_MODEL=nesso` and any local `ORACLE_TOKEN`, then:

```bash
export ORACLE_URL=http://127.0.0.1:8080
export ORACLE_TOKEN=local-dev-token
python harness/run_ab.py --budget 600 --seed 1
```

See [docs/ORACLE.md](docs/ORACLE.md).

## Compare only

```bash
python harness/compare.py runs/<run_id>
```

Primary metric: portfolio mean measured selectivity. Promote `variant/` only when it wins; then stack the next experiment on that winner. Details in `EXPERIMENTS.md`.
