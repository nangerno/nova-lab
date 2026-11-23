# Experiments

Rules:

1. Never edit `baseline/`.
2. One change in `variant/` per experiment.
3. Short budget default: **600s**, fixed challenge `challenges/fixed_v1`, ≥2 seeds before promote.
4. Primary metric: **portfolio mean**. Secondary: `#scored`, pairs/sec, entropy, errors.
5. Win both seeds → `runs/promoted/` + keep change in `variant/`. Else revert.
6. Short-budget wins need a **3600s** confirmation before competition submit.

## Backlog

| ID | Axis | Status | Change |
| --- | --- | --- | --- |
| E1a | Throughput | **rejected** | early score gate |
| E1b | Throughput | **promoted** | `fit_within` 1.15 → 1.05 |
| E2 | Acquisition | **rejected** | Thompson `(1-0.85p)` |
| E3 | Local search | **promoted** | `RETRIEVE_SHARE` 0.50 → 0.75 |
| E4 | Portfolio packing | **promoted** | `max_swaps` 400 → 800 |
| E5 | Structural bets | **queued** | only if 3600s tip still flat |

## Current tip (`variant/`)

Champion + **E1b + E3 + E4**:

1. `miner.py`: latency safety `1.05`
2. `miner.py`: default `RETRIEVE_SHARE=0.75`
3. `portfolio.py`: `max_swaps=800`

Snapshots: `runs/promoted/E1b`, `E3`, `E4`.

## Log

| Date | Run ID | Exp | Seed | Δ / decision |
| --- | --- | --- | --- | --- |
| 2026-11-22 | `E1_seed1/2_b600` | E1a | 1–2 | mixed → reject |
| 2026-11-22 | `E1b_seed1/2_b600` | E1b | 1–2 | +0.87 / +0.13 → **promote** |
| 2026-11-22 | `E2_seed1/2_b600` | E2 | 1–2 | mixed → reject |
| 2026-11-22 | `E3_seed1_b600` | E3 | 1 | +0.373 |
| 2026-11-22 | `E3_seed2_b600` | E3 | 2 | inconclusive (baseline no portfolio) |
| 2026-11-22 | `E3_seed2b_b600` | E3 | 2 | +0.065 → **promote** |
| 2026-11-23 | `E4_seed1_b600` | E4 | 1 | +0.103 |
| 2026-11-23 | `E4_seed2_b600` | E4 | 2 | +0.018 → **promote** |
| 2026-11-23 | `TIP_confirm_seed1_b3600` | tip=E1b+E3+E4 | 1 @3600s | **+0.176 variant_wins** (scored 789→781) |

## Long-run confirmation

`TIP_confirm_seed1_b3600`: tip beat frozen champion by **+0.176** portfolio mean at full budget.

Optional: second 3600s seed before submit; then E5 only if still flat.
