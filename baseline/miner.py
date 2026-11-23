"""Online Nesso selectivity search, generated and learned entirely in the sandbox.

Uniform initial chemistry, online GP acquisition, structural neighbourhood search,
and measured-score portfolio selection under the validator's exact validity rules.
"""
from __future__ import annotations

import os
import time

T_START = time.time()
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import math
import multiprocessing as mp
import queue
import random
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chem
import portfolio
import portfolio_acquisition
import retrieve
from model import GP
from space import Space, make_name
from nova_miner.utils.oracle import Oracle, OracleError, combine
from submission_gate import reject_reason

_CFG: dict | None = None
BUDGET = float(os.environ.get("NOVA_TIME_BUDGET_SEC", 3600))
SAFETY = 12.0
WORKDIR = os.environ.get("WORKDIR", HERE)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
MAX_PAIRS = 512
POOL_CAP = 60000
RESERVE_CAP = 6000
ELITES = 40
DIV_ELITE_SHARE = 0.25
EXPLORE_SHARE = 0.25
EXPLORE_FULL_UNTIL = 0.35
EXPLORE_OFF_AT = 0.60
EXPLORE_PER_RXN = 3
EXPLORE_MAX_PER_RXN = 60
PROBE_SEEDS = 4
PROBE_ROUNDS = 2
PROBE_LOCAL_SHARE = 0.30
LOCAL_START, LOCAL_RAMP, LOCAL_CAP = 0.25, 0.02, 0.70
RETRIEVE_SHARE = min(1.0, max(0.0, float(os.environ.get("NOVA_RETRIEVE", "0.50"))))
RETRIEVE_TRIES = 8
RXN_TEMP = 0.50
PORTFOLIO_SHARE = min(0.25, max(0.0, float(os.environ.get("NOVA_PORTFOLIO_SHARE", "0.25"))))
SEED = int(os.environ.get("NOVA_SEED", 0)) or int(T_START * 1000) % (2 ** 31)


def log(message: str) -> None:
    print(f"[{time.time() - T_START:7.1f}s] {message}", flush=True)


def deadline() -> float:
    return T_START + BUDGET - SAFETY


def cpu_budget() -> int:
    """Docker CPU quota can be smaller than the visible affinity mask."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    try:
        with open("/sys/fs/cgroup/cpu.max") as handle:
            quota, period = handle.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as handle:
            quota = int(handle.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as handle:
            period = int(handle.read())
        if quota > 0 and period > 0:
            n = min(n, max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, n)


def write_json_atomic(path: str, obj) -> bool:
    """A failed checkpoint never replaces the last fully validated result."""
    if _CFG is not None and os.path.basename(path) == "result.json":
        try:
            why = reject_reason(list(obj["molecules"]), _CFG)
        except Exception as exc:
            why = f"gate error {exc!r}"
        if why is not None:
            log(f"result withheld: {why}")
            return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as handle:
            json.dump(obj, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(path + ".tmp", path)
        return True
    except (OSError, ValueError) as exc:
        log(f"write failed {path}: {exc}")
        return False


def score_row(row, n_targets: int, n_antitargets: int, weight: float):
    """(selectivity, per-protein pIC50) or None for an incomplete/nonfinite read.

    Check the raw affinity before clipping: infinity must never become a valid
    boundary pIC50. A failed antitarget is unavailable evidence, not selectivity.
    """
    if not isinstance(row, dict) or n_targets < 1 or not math.isfinite(weight):
        return None
    metrics = row.get("scores")
    if not isinstance(metrics, list) or len(metrics) != n_targets + n_antitargets:
        return None
    values = []
    for metric in metrics:
        if not isinstance(metric, dict):
            return None
        raw = metric.get("affinity_pred_value")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            return None
        value = combine(metric)
        if not math.isfinite(value):
            return None
        values.append(value)
    target = sum(values[:n_targets]) / n_targets
    anti = sum(values[n_targets:]) / n_antitargets if n_antitargets else 0.0
    result = target - weight * anti
    return (result, values) if math.isfinite(result) else None


class LatencyModel:
    """Request duration learned from this run, with pairs as the work unit."""

    def __init__(self, proteins: int):
        self.proteins = proteins
        self.obs = []

    def add(self, molecules: int, seconds: float) -> None:
        if molecules > 0 and seconds > 0 and math.isfinite(seconds):
            self.obs.append((molecules * self.proteins, seconds))

    def predict(self, molecules: int) -> float:
        if molecules <= 0:
            return 0.0
        if not self.obs:
            # A startup allowance, not an assumed model speed. Replaced after
            # the first response; the HTTP timeout also obeys the run deadline.
            return max(1.0, min(120.0, BUDGET * 0.10))
        pairs = molecules * self.proteins
        recent = self.obs[-8:]
        ns = np.asarray([n for n, _ in recent], float)
        ss = np.asarray([s for _, s in recent], float)
        if len(set(ns)) > 1:
            slope, intercept = np.polyfit(ns, ss, 1)
            estimate = max(0.0, intercept) + max(0.0, slope) * pairs
        else:
            estimate = float(np.median(ss / ns)) * pairs
        return max(0.01, float(estimate))

    def fit_within(self, seconds: float, cap: int) -> int:
        lo, hi = 0, max(0, cap)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if 1.15 * self.predict(mid) <= seconds:
                lo = mid
            else:
                hi = mid - 1
        return lo


class Miner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.n_final = int(cfg["num_molecules"])
        self.tan_max = float(cfg.get("tanimoto_max_threshold", 1.0))
        self.ent_min = float(cfg["entropy_min_threshold"])
        self.targets = list(cfg["target_sequences"])
        self.antitargets = list(cfg.get("antitarget_sequences") or [])
        self.anti_w = float(cfg["antitarget_weight"])
        if not self.targets or self.n_final < 1 or not math.isfinite(self.anti_w) or self.anti_w < 0:
            raise ValueError("invalid challenge target count, molecule count, or antitarget weight")
        self.rng = random.Random(SEED)
        self.nprng = np.random.default_rng(SEED)
        started = time.time()
        self.space = Space()
        self.rxns = [r for r in self.space.rxn_ids if all(self.space.pool_sizes(r))]
        allowed = cfg.get("allowed_reaction")
        if allowed is not None:
            self.rxns = [r for r in self.rxns if str(allowed) == f"rxn:{r}"]
        if not self.rxns:
            raise ValueError("challenge has no available allowed reaction")
        log(f"space: {len(self.rxns)} allowed reactions, {len(self.space.smiles)} blocks in {time.time()-started:.1f}s")
        self.oracle = Oracle(os.environ.get("ORACLE_SOCKET", "/run/oracle.sock"))
        self.max_pairs = MAX_PAIRS
        self.latency = LatencyModel(len(self.targets) + len(self.antitargets))
        self.model = GP(chem.FEAT_BITS)
        self.pool = {}
        self.keys_seen = set()
        self.scored = []
        self.scored_keys = set()
        self.inflight = {}
        self.elites = []
        self.probe_seeds = []
        self.rxn_weight = {r: 1.0 for r in self.rxns}
        self.ledger = dict(tuples=0, products_valid=0, dup_inchikey=0, requests=0, http_requests=0,
                           pairs_requested=0, pairs_delivered=0, failed=0, retried=0,
                           oracle_seconds=0.0, rounds=[], writes=0, fallback=False, errors=0, evicted=0)
        self.ledger["portfolio_acquisition"] = dict(gate_checks=0, gate_active=0, calls=0,
            proposed=0, rows_analyzed=0, feasible_candidates=0, gate_seconds=0.0, selection_seconds=0.0)
        self.fitted_n = 0
        self.post_s = 5.0
        self.written_names = None
        self.written_value = -math.inf
        self.workers = max(1, min(cpu_budget() - 1, 24))
        if os.environ.get("NOVA_WORKERS"):
            self.workers = max(1, min(self.workers, int(os.environ["NOVA_WORKERS"])))
        chem.SPACE = self.space
        self.jobs_seen = set()
        self.ret_fail = dict(calls=0, ok=0, not_ready=0, no_nbrs=0, all_seen=0)
        self.pexec = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("fork"),
                                         initializer=chem.init_worker,
                                         initargs=(cfg["filters"], self.space.reactions))
        # Fork before any retrieval or Oracle thread starts.
        list(self.pexec.map(int, range(self.workers)))
        self.ret = None
        self.gen_futs = []
        log(f"ours_portfolio workers={self.workers} seed={SEED} pair_cap={self.max_pairs} portfolio_share={PORTFOLIO_SHARE}")

    def _tuple(self, rxn: int) -> list[int]:
        return [int(self.space.pool[m][self.nprng.integers(len(self.space.pool[m]))])
                for m in self.space.roles(rxn)]

    def _job(self, n: int) -> list[tuple]:
        """Global exploration and local changes around current-run measured elites."""
        local = int(n * min(LOCAL_CAP, LOCAL_START + LOCAL_RAMP * len(self.ledger["rounds"]))) if self.elites else 0
        weights = np.asarray([self.rxn_weight[r] for r in self.rxns], float)
        weights = 0.85 * weights / weights.sum() + 0.15 / len(weights)
        jobs = []
        for _ in range(n - local):
            rxn = self.rxns[int(self.nprng.choice(len(self.rxns), p=weights))]
            jobs.append((rxn, self._tuple(rxn), "global"))
        probes = self.probe_seeds
        n_probe = int(local * PROBE_LOCAL_SHARE) if probes else 0
        for i in range(local):
            elite = (probes[i % len(probes)] if i < n_probe else
                     self.elites[min(int(self.nprng.exponential(len(self.elites) / 3)), len(self.elites) - 1)])
            rxn, ids = elite["rxn"], list(elite["ids"])
            if self.ret is not None and self.rng.random() < RETRIEVE_SHARE:
                neighbour = self._retrieve(rxn, ids)
                if neighbour is not None:
                    jobs.append((rxn, neighbour, "retrieve"))
                    continue
            mates = [r for r in self.elites if r["rxn"] == rxn and r["ids"] != elite["ids"]]
            if mates and self.rng.random() >= 0.75:
                mate = self.rng.choice(mates)
                ids = [a if self.rng.random() < 0.5 else b for a, b in zip(ids, mate["ids"])]
                source = "cross"
            else:
                position = self.rng.randrange(len(ids))
                role = self.space.roles(rxn)[position]
                ids[position] = int(self.space.pool[role][self.nprng.integers(len(self.space.pool[role]))])
                source = "mutate"
            jobs.append((rxn, ids, source))
        return [(rxn, ids, [self.space.smiles[i] for i in ids],
                 [self.space.mask[i] for i in ids], source) for rxn, ids, source in jobs]

    def _retrieve(self, rxn: int, ids: list[int]):
        roles = self.space.roles(rxn)
        self.ret_fail["calls"] += 1
        why = "not_ready"
        for _ in range(3):
            position = self.rng.randrange(len(ids))
            role = roles[position]
            neighbours = self.ret.nearest(role, ids[position])
            if not neighbours:
                why = "no_nbrs" if self.ret.ready(role) else "not_ready"
                continue
            why = "all_seen"
            top = neighbours[:retrieve.NN_K]
            weights = 1.0 / (np.arange(len(top)) + 4.0)
            weights /= weights.sum()
            order = [int(top[int(self.nprng.choice(len(top), p=weights))]) for _ in range(RETRIEVE_TRIES)]
            order += [int(block) for block in top]
            for block in order:
                if block == ids[position]:
                    continue
                candidate = list(ids)
                candidate[position] = block
                key = (rxn, tuple(candidate))
                if key in self.jobs_seen:
                    continue
                if len(self.jobs_seen) > 600000:
                    self.jobs_seen.clear()
                self.jobs_seen.add(key)
                self.ret_fail["ok"] += 1
                return candidate
        self.ret_fail[why] += 1
        return None

    def pump_generation(self) -> None:
        done = [f for f in self.gen_futs if f.done()]
        self.gen_futs = [f for f in self.gen_futs if not f.done()]
        for future in done:
            try:
                records, stats = future.result()
            except Exception as exc:
                self.ledger["errors"] += 1
                log(f"generation job failed: {exc}")
                continue
            self.ledger["tuples"] += stats["tuples"]
            self.ledger["products"] = self.ledger.get("products", 0) + stats["products"]
            self.ledger["products_invalid"] = self.ledger.get("products_invalid", 0) + stats["invalid"]
            self.ledger["gen_errors"] = self.ledger.get("gen_errors", 0) + stats["errors"]
            for record in records:
                self.ledger["products_valid"] += 1
                if record["inchikey"] in self.keys_seen:
                    self.ledger["dup_inchikey"] += 1
                    continue
                record["name"] = make_name(record["rxn"], record["ids"], record["k"])
                self.keys_seen.add(record["inchikey"])
                self.pool[record["name"]] = record
        if len(self.pool) > POOL_CAP:
            self._evict()
        if len(self.pool) < POOL_CAP:
            while len(self.gen_futs) < 2 * self.workers:
                self.gen_futs.append(self.pexec.submit(chem.expand, self._job(120)))

    def _evict(self) -> None:
        records = list(self.pool.values())
        if self.model.trained:
            mean, _ = self.model.predict(records, with_std=False)
            order = np.argsort(mean)
        else:
            order = self.nprng.permutation(len(records))
        drop = order[:len(records) - int(POOL_CAP * 0.7)]
        for i in drop:
            self.pool.pop(records[i]["name"], None)
        self.ledger["evicted"] += len(drop)

    def choose_batch(self, n: int) -> list[dict]:
        candidates = list(self.pool.values())
        if not candidates or n <= 0:
            return []
        if not self.model.trained:
            by_rxn = {}
            for record in candidates:
                by_rxn.setdefault(record["rxn"], []).append(record)
            for records in by_rxn.values():
                self.rng.shuffle(records)
            keys = list(by_rxn)
            self.rng.shuffle(keys)
            picked = []
            while len(picked) < n and any(by_rxn.values()):
                for key in keys:
                    if by_rxn[key] and len(picked) < n:
                        picked.append(by_rxn[key].pop())
            return picked
        explore = self._explore_picks(candidates, int(round(n * self._explore_share())))
        used = {record["name"] for record in explore}
        candidates = [record for record in candidates if record["name"] not in used]
        need = n - len(explore)
        if need <= 0 or not candidates:
            return explore[:n]
        mean, _ = self.model.predict(candidates, with_std=False)
        top = np.argsort(-mean)[:3000]
        extra = self.nprng.choice(len(candidates), min(len(candidates), 800), replace=False)
        indices = np.unique(np.concatenate([top, extra]))
        subset = [candidates[i] for i in indices]
        mu, sd = self.model.predict(subset, with_std=True)
        progress = min(1.0, (time.time() - T_START) / BUDGET)
        draws = mu + sd * self.nprng.standard_normal(len(subset)) * (1.0 - 0.6 * progress)
        from rdkit import DataStructs
        portfolio_indices = []
        if PORTFOLIO_SHARE > 0 and self.written_names:
            lookup = {record["name"]: record for record in self.scored}
            incumbents = [lookup[name] for name in self.written_names if name in lookup]
            gate_started = time.time()
            leader_entropy = portfolio_acquisition.greedy_entropy(self.scored, self.n_final, self.tan_max)
            diagnostics = self.ledger.setdefault("portfolio_acquisition", dict(gate_checks=0,
                gate_active=0, calls=0, proposed=0, rows_analyzed=0, feasible_candidates=0,
                gate_seconds=0.0, selection_seconds=0.0))
            diagnostics["gate_checks"] += 1
            diagnostics["gate_seconds"] += time.time() - gate_started
            diagnostics["last_leader_entropy"] = leader_entropy
            diagnostics["gate_basis"] = "measured_greedy_identity_morgan"
            # Use a different acquisition only after the measured high-score set
            # demonstrates the entropy bottleneck this slice is intended to help.
            if (len(incumbents) == self.n_final and leader_entropy is not None
                    and leader_entropy <= self.ent_min + 0.002):
                diagnostics["gate_active"] += 1
                started = time.time()
                portfolio_indices, stats = portfolio_acquisition.pick(
                    subset, mu, sd * (1.0 - 0.6 * progress), incumbents,
                    min(need, int(round(n * PORTFOLIO_SHARE))), self.tan_max, self.ent_min,
                    self.nprng, random_indices=np.flatnonzero(np.isin(indices, extra)))
                elapsed = time.time() - started
                diagnostics["calls"] += 1
                diagnostics["proposed"] += len(portfolio_indices)
                diagnostics["rows_analyzed"] += stats["candidates"]
                diagnostics["feasible_candidates"] += stats["feasible"]
                diagnostics["selection_seconds"] += elapsed
                diagnostics["last"] = dict(stats, seconds=elapsed, leader_entropy=leader_entropy)
                log(f"portfolio acquisition selected={len(portfolio_indices)} "
                    f"candidates={stats['candidates']} feasible={stats['feasible']} "
                    f"leader_entropy={leader_entropy:.6f} seconds={elapsed:.3f}")
        portfolio_set = set(portfolio_indices)
        picked = [subset[i] for i in portfolio_indices]
        fingerprints = [portfolio.rule_fp(record) for record in picked]
        for record in picked:
            record["acquisition"] = "portfolio_ei"
        for j in np.argsort(-draws):
            if int(j) in portfolio_set:
                continue
            record = subset[j]
            fp = portfolio.rule_fp(record)
            if fingerprints and np.count_nonzero(
                    np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fingerprints)) >= self.tan_max) >= 2:
                continue
            picked.append(record)
            record["acquisition"] = "baseline"
            fingerprints.append(fp)
            if len(picked) >= need:
                break
        for record in explore:
            record["acquisition"] = "reaction_ucb"
        return explore + picked

    def _explore_share(self) -> float:
        progress = (time.time() - T_START) / BUDGET
        if progress <= EXPLORE_FULL_UNTIL:
            return EXPLORE_SHARE
        if progress >= EXPLORE_OFF_AT:
            return 0.0
        return EXPLORE_SHARE * (EXPLORE_OFF_AT - progress) / (EXPLORE_OFF_AT - EXPLORE_FULL_UNTIL)

    def _explore_picks(self, candidates: list[dict], count: int) -> list[dict]:
        if count <= 0 or len(self.ledger["rounds"]) < 2:
            return []
        counts = _count(record["rxn"] for record in self.scored)
        dominant = max(counts, key=counts.get) if counts else None
        under = {rxn for rxn in self.rxns if rxn != dominant and counts.get(rxn, 0) < EXPLORE_MAX_PER_RXN}
        pool = [record for record in candidates if record["rxn"] in under]
        if not pool:
            return []
        if len(pool) > 3000:
            pool = [pool[i] for i in self.nprng.choice(len(pool), 3000, replace=False)]
        mean, sd = self.model.predict(pool, with_std=True)
        picked, per_rxn = [], {}
        for j in np.argsort(-(mean + 2.0 * sd)):
            record = pool[j]
            if per_rxn.get(record["rxn"], 0) >= EXPLORE_PER_RXN:
                continue
            per_rxn[record["rxn"]] = per_rxn.get(record["rxn"], 0) + 1
            picked.append(record)
            if len(picked) >= count:
                break
        return picked

    def _score(self, batch):
        started = time.time()
        rows, error = self._score_chunk(batch, self.targets + self.antitargets, 0)
        return rows, time.time() - started, error

    def _score_chunk(self, batch, proteins, depth):
        if len(batch) > self._batch_cap():
            rows, error, start = [], None, 0
            while start < len(batch):
                # A left-hand request can reduce the cap again. Recheck before
                # sending the remaining work, preserving every original slot.
                count = min(self._batch_cap(), len(batch) - start)
                part, problem = self._score_chunk(batch[start:start+count], proteins, depth)
                rows.extend(part if part is not None else [None] * count)
                error = error or problem
                start += count
            return rows, error
        error = None
        for attempt in range(2):
            room = deadline() - time.time() - self.post_s
            if room <= 1:
                return None, "deadline"
            self.oracle.timeout = min(900.0, room)
            self.ledger["http_requests"] += 1
            self.ledger["pairs_requested"] += len(batch) * len(proteins)
            try:
                rows = self.oracle.score(targets=proteins, smiles=[record["smiles"] for record in batch])
                if not isinstance(rows, list):
                    raise ValueError("Oracle results is not a list")
                # Preserve sub-request positions even for a truncated response.
                return rows[:len(batch)] + [None] * max(0, len(batch) - len(rows)), None
            except OracleError as exc:
                error = f"{exc.status}: {exc.detail}"
                if exc.status == 413 and len(batch) > 1 and depth < 8:
                    return self._halve(batch, proteins, depth, "413")
                if exc.status in (401, 409, 422):
                    break
            except (OSError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, TimeoutError) and len(batch) > 1 and depth < 8:
                    return self._halve(batch, proteins, depth, "timeout")
            if attempt == 0:
                time.sleep(min(1.0, max(0.0, deadline() - time.time())))
        return None, error

    def _halve(self, batch, proteins, depth, why):
        half = max(1, len(batch) // 2)
        self.max_pairs = min(self.max_pairs, max(len(proteins), half * len(proteins)))
        log(f"Oracle {why}: reducing cap to {self.max_pairs} pairs")
        left, error_left = self._score_chunk(batch[:half], proteins, depth + 1)
        right, error_right = self._score_chunk(batch[half:], proteins, depth + 1)
        left = left if left is not None else [None] * half
        right = right if right is not None else [None] * (len(batch) - half)
        return left + right, error_left or error_right

    def ingest(self, batch, rows, seconds, error) -> None:
        self.ledger["requests"] += 1
        self.ledger["oracle_seconds"] += seconds
        got = 0
        new = []
        for i, record in enumerate(batch):
            self.inflight.pop(record["name"], None)
            self.pool.pop(record["name"], None)
            row = rows[i] if isinstance(rows, list) and i < len(rows) else None
            decoded = score_row(row, len(self.targets), len(self.antitargets), self.anti_w)
            if isinstance(row, dict) and row.get("smiles", record["smiles"]) != record["smiles"]:
                decoded = None
            if decoded is not None:
                value, pic50 = decoded
                if record["inchikey"] in self.scored_keys:
                    continue
                record["y"] = value
                record["pic50"] = pic50
                record["round"] = len(self.ledger["rounds"])
                self.scored_keys.add(record["inchikey"])
                self.scored.append(record)
                self.ledger["pairs_delivered"] += len(pic50)
                new.append(record)
                got += 1
            else:
                self.ledger["failed"] += 1
                if not record.get("retried"):
                    record["retried"] = True
                    self.ledger["retried"] += 1
                    self.pool[record["name"]] = record
        if got:
            self.latency.add(len(batch), seconds)
        try:
            with open(os.path.join(OUTPUT_DIR, "scored.jsonl"), "a") as handle:
                for record in new:
                    payload = {key: record[key] for key in
                        ("name", "smiles", "y", "pic50", "round", "rxn", "k", "source")}
                    payload["acquisition"] = record.get("acquisition", "baseline")
                    handle.write(json.dumps(payload, allow_nan=False) + "\n")
        except OSError:
            pass
        best = max((record["y"] for record in self.scored), default=None)
        self.ledger["rounds"].append(dict(t=round(time.time() - T_START, 1), n=len(batch), got=got,
                                          seconds=round(seconds, 1), error=error, scored=len(self.scored),
                                          best=best, pool=len(self.pool)))
        log(f"round {len(self.ledger['rounds'])}: {got}/{len(batch)} in {seconds:.1f}s; "
            f"scored={len(self.scored)} best={best} pool={len(self.pool)}")

    def refit(self) -> None:
        if len(self.scored) < 12 or len(self.scored) == self.fitted_n:
            return
        started = time.time()
        tune = len(self.ledger["rounds"]) % 3 == 1 or not self.model.trained
        if not self.model.fit(self.scored, [record["y"] for record in self.scored], tune=tune):
            return
        self.fitted_n = len(self.scored)
        # Outcomes are deterministic within the serving environment; measured
        # values choose local parents and the final set. GP is acquisition only.
        order = np.argsort(-np.asarray([record["y"] for record in self.scored]))
        top = [self.scored[i] for i in order[:self.n_final]]
        counts = _count(record["rxn"] for record in top)
        dominant = max(counts, key=counts.get)
        diverse = int(ELITES * DIV_ELITE_SHARE)
        self.elites = [self.scored[i] for i in order[:ELITES - diverse]]
        chosen = {record["inchikey"] for record in self.elites}
        self.elites += [self.scored[i] for i in order if self.scored[i]["rxn"] != dominant
                        and self.scored[i]["inchikey"] not in chosen][:diverse]
        chosen = {record["inchikey"] for record in self.elites}
        self.elites += [self.scored[i] for i in order if self.scored[i]["inchikey"] not in chosen][:ELITES-len(self.elites)]
        by_rxn = {}
        for record in self.scored:
            by_rxn.setdefault(record["rxn"], []).append(record)
        seeds = []
        quantile = np.quantile([record["y"] for record in self.scored], 0.75)
        for rxn, records in by_rxn.items():
            if rxn == dominant or len(records) >= EXPLORE_MAX_PER_RXN:
                continue
            best = max(records, key=lambda record: record["y"])
            if len(self.ledger["rounds"]) - best["round"] <= PROBE_ROUNDS and best["y"] >= quantile:
                seeds.append(best)
        self.probe_seeds = sorted(seeds, key=lambda record: -record["y"])[:PROBE_SEEDS]
        all_values = np.asarray([record["y"] for record in self.scored])
        high = float(np.quantile(all_values, 0.9))
        scale = float(all_values.std()) or 1.0
        optimistic = []
        for rxn in self.rxns:
            values = sorted((record["y"] for record in by_rxn.get(rxn, [])), reverse=True)
            optimistic.append((float(np.mean(values[:3])) if values else high) + scale / math.sqrt(1 + len(values)))
        optimistic = np.asarray(optimistic)
        weights = np.exp((optimistic - optimistic.max()) / max(RXN_TEMP * scale, 1e-12))
        self.rxn_weight = {rxn: float(weight) for rxn, weight in zip(self.rxns, weights)}
        log(f"refit {len(self.scored)} observations in {time.time()-started:.1f}s; hyper={self.model.hyper}")

    def write_portfolio(self, final: bool = False) -> None:
        records = self.scored
        if len(records) < self.n_final:
            return
        values = [record["y"] for record in records]
        budget = max(2.0, self.post_s if final else 0.08 * self.latency.predict(self._batch_cap()))
        room = T_START + BUDGET - 4.0 - time.time()
        if room <= 0:
            return
        budget = min(budget, max(0.05, room * 0.5))
        indices, entropy = portfolio.select(records, values, self.n_final, self.tan_max, self.ent_min,
                                             seconds=budget, max_reserve=min(len(records), RESERVE_CAP))
        if indices is None:
            return
        value = float(np.mean([values[i] for i in indices]))
        if value <= self.written_value + 1e-12:
            return
        names = [records[i]["name"] for i in indices]
        if write_json_atomic(os.path.join(OUTPUT_DIR, "result.json"), {"molecules": names}):
            self.written_names, self.written_value = names, value
            self.ledger["writes"] += 1
            self.ledger["last_write"] = dict(t=round(time.time()-T_START, 1), value=value,
                entropy=entropy, rxns=len({records[i]["rxn"] for i in indices}), measured=True)
            log(f"wrote measured portfolio mean={value:.4f} entropy={entropy:.4f}" + (" final" if final else ""))

    def write_fallback(self) -> bool:
        records = list(self.pool.values())
        if len(records) < 4 * self.n_final:
            return False
        indices, entropy = portfolio.select(records, self.nprng.random(len(records)), self.n_final,
                                             self.tan_max, self.ent_min)
        if indices is None:
            return False
        if write_json_atomic(os.path.join(OUTPUT_DIR, "result.json"),
                             {"molecules": [records[i]["name"] for i in indices]}):
            self.ledger["fallback"] = True
            log(f"wrote generated fallback entropy={entropy:.4f}")
            return True
        return False

    def dump_ledger(self) -> None:
        ledger = dict(self.ledger, scored=len(self.scored), pool=len(self.pool),
                      elapsed=round(time.time()-T_START, 1), workers=self.workers, seed=SEED,
                      hyper=list(self.model.hyper) if self.model.trained else None,
                      scored_rxn={str(k): v for k, v in _count(r["rxn"] for r in self.scored).items()},
                      sources=_count(record.get("source") for record in self.scored),
                      retrieve=dict(self.ret.stats) if self.ret is not None else None,
                      retrieve_calls=dict(self.ret_fail))
        write_json_atomic(os.path.join(OUTPUT_DIR, "miner_ledger.json"), ledger)

    def _batch_cap(self) -> int:
        cap = max(1, self.max_pairs // (len(self.targets) + len(self.antitargets)))
        return min(cap, max(1, int(os.environ.get("NOVA_BATCH", 96))))

    def _round_size(self, after: float = 0.0) -> int:
        room = deadline() - time.time() - after - 1.5 * self.post_s
        return self.latency.fit_within(room, self._batch_cap())

    def _driver(self) -> None:
        while True:
            batch = self.q_send.get()
            room = deadline() - time.time() - 1.5 * self.post_s
            count = min(len(batch), self._batch_cap(), self.latency.fit_within(room, len(batch)))
            if count <= 0:
                self.q_done.put(([], None, 0.0, "deadline", batch))
                continue
            batch, leftover = batch[:count], batch[count:]
            self.active = (batch, time.time())
            try:
                rows, seconds, error = self._score(batch)
            except Exception as exc:
                rows, seconds, error = None, time.time()-self.active[1], f"{type(exc).__name__}: {exc}"
            self.q_done.put((batch, rows, seconds, error, leftover))
            self.active = None

    def process_completed(self, fit: bool = True) -> None:
        """Drain responses already available, without waiting for an active call."""
        while True:
            try:
                batch, rows, seconds, error, leftover = self.q_done.get_nowait()
            except queue.Empty:
                break
            for record in leftover:
                self.inflight.pop(record["name"], None)
                self.pool[record["name"]] = record
            if not batch:
                continue
            self.ingest(batch, rows, seconds, error)
            started = time.time()
            # Publish measured improvements even if surrogate fitting fails.
            try:
                self.write_portfolio(final=not fit)
                if fit and time.time() + 1.5*self.post_s < deadline():
                    self.refit()
            except Exception:
                self.ledger["errors"] += 1
                log("post-round failure:\n" + traceback.format_exc())
            self.post_s = 0.7*self.post_s + 0.3*(time.time()-started)
            self.dump_ledger()

    def run(self) -> None:
        self.q_send = queue.Queue(maxsize=1)
        self.q_done = queue.Queue()
        self.active = None
        threading.Thread(target=self._driver, daemon=True).start()
        fallback_done, fallback_at, stop_sending = False, 0, False
        while time.time() < deadline():
            self.pump_generation()
            if not fallback_done and len(self.pool) >= max(4*self.n_final, fallback_at):
                fallback_done = self.write_fallback()
                fallback_at = len(self.pool) + self.n_final
            if self.ret is None and RETRIEVE_SHARE > 0 and len(self.pool) >= 4*self.n_final:
                self.ret = retrieve.Retriever(self.space, log, self.pexec)
            self.process_completed()
            active = self.active
            if not stop_sending and self.q_send.empty():
                if active is None:
                    ready, after = self.q_done.empty(), 0.0
                else:
                    expected = self.latency.predict(len(active[0]))
                    flown = time.time() - active[1]
                    after = max(0.0, expected - flown)
                    ready = flown > 0.5*expected or not self.model.trained
                if ready:
                    count = self._round_size(after)
                    if count <= 0:
                        # An active call can finish before its latency estimate.
                        # Reconsider the remaining budget after that response.
                        if active is None and self.q_done.empty():
                            stop_sending = True
                    elif len(self.pool) >= min(4*count, 400) or time.time()-T_START > 60:
                        try:
                            batch = self.choose_batch(count)[:count]
                        except Exception:
                            self.ledger["errors"] += 1
                            log("batch choice failed:\n" + traceback.format_exc())
                            batch = []
                        if batch:
                            for record in batch:
                                self.pool.pop(record["name"], None)
                                self.inflight[record["name"]] = record
                            self.q_send.put(batch)
            if stop_sending and not self.inflight and self.q_send.empty() and self.q_done.empty():
                break
            time.sleep(0.05)
        try:
            self.process_completed(fit=False)
            if time.time() + self.post_s < T_START + BUDGET - 4:
                self.write_portfolio(final=True)
        except Exception:
            log("final write failed:\n" + traceback.format_exc())
        self.dump_ledger()
        log(f"done: scored={len(self.scored)} requests={self.ledger['http_requests']}")


def _count(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> None:
    with open(os.path.join(WORKDIR, "input.json")) as handle:
        raw = json.load(handle)
    cfg = {**raw.get("config", {}), **raw.get("challenge", {})}
    global _CFG
    _CFG = cfg
    log(f"targets={len(cfg['target_sequences'])} antitargets={len(cfg.get('antitarget_sequences') or [])} budget={BUDGET:.0f}s")
    miner = Miner(cfg)
    try:
        miner.run()
    finally:
        try:
            for process in list(getattr(miner.pexec, "_processes", {}).values()):
                process.terminate()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    main()
