"""Structural neighbours built during this run from the sandbox database.

One shared packed-fingerprint table covers all blocks; each role gathers a
contiguous view and exact Tanimoto retrieval is cached per reference block.
Building and queries have CPU/memory limits and are optional: candidate
search continues with mutation and crossover while retrieval is unavailable.
No affinity labels, preferred blocks, or external molecular lists are loaded.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

FP_BITS = 2048
FP_RADIUS = 2
NN_K = 64                      # neighbours kept per reference block
CHUNK = 20_000                 # blocks per parallel fingerprinting task
ROLE_CAP = 6_000_000           # materialised role rows; 256 B each, so ~1.5 GB of role matrices
QUERY_SHARE = 0.10             # at most this share of elapsed wall time spent in cold queries
QUERY_GRACE = 8.0              # ...plus a flat allowance, so the first rounds are not starved

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def _pack(ids, smiles):
    """-> (ids RDKit could read, their packed fingerprints), in the order given."""
    keep, rows = [], []
    for mid in ids:
        mid = int(mid)
        mol = Chem.MolFromSmiles(smiles[mid])
        if mol is None:                          # a block RDKit cannot read is not a candidate
            continue
        rows.append(np.packbits(_GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)))
        keep.append(mid)
    return (np.asarray(keep, dtype=np.int64),
            np.vstack(rows) if rows else np.zeros((0, FP_BITS // 8), dtype=np.uint8))


def fp_chunk(ids):
    """Worker side of the parallel build; runs in the miner's generation pool.

    `chem.SPACE` is assigned before that pool is created, so every forked worker already holds the
    block table and only the ids have to travel.
    """
    import chem
    return _pack(ids, chem.SPACE.smiles)


class BlockTable:
    """Packed Morgan fingerprints of every distinct building block, built once per run."""

    def __init__(self, smiles: dict, pexec=None):
        ids = sorted(int(m) for m in smiles)
        if pexec is None or len(ids) < 2 * CHUNK:
            keep, fp = _pack(ids, smiles)
        else:
            parts = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
            got = list(pexec.map(fp_chunk, parts))
            n = sum(len(k) for k, _ in got)
            keep = np.empty(n, dtype=np.int64)
            fp = np.empty((n, FP_BITS // 8), dtype=np.uint8)   # filled in place: no vstack copy
            at = 0
            for k, m in got:
                keep[at:at + len(k)] = k
                fp[at:at + len(k)] = m
                at += len(k)
        self.ids = keep
        self.fp = fp
        self.bits = np.bitwise_count(self.fp).sum(1).astype(np.int32)
        self.row = {int(m): i for i, m in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)


class RoleIndex:
    """One role's blocks as a contiguous matrix gathered from the shared table."""

    def __init__(self, role: int, pool, table: BlockTable):
        self.role = int(role)
        rows = [table.row[int(m)] for m in pool if int(m) in table.row]
        take = np.asarray(rows, dtype=np.int64)
        self.ids = table.ids[take]
        self.fp = table.fp[take]                 # a copy, so the query scans contiguous memory
        self.bits = table.bits[take]
        self.row = {int(m): i for i, m in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def nearest(self, mid: int, k: int = NN_K) -> list[int]:
        """The k blocks of this role most similar to `mid`, nearest first, `mid` excluded.

        Exact Tanimoto over the whole pool: popcount(a & b) / (|a| + |b| - popcount(a & b))."""
        i = self.row.get(int(mid))
        if i is None or len(self.ids) < 2:
            return []
        ref, rb = self.fp[i], int(self.bits[i])
        inter = np.bitwise_count(np.bitwise_and(self.fp, ref)).sum(1).astype(np.int32)
        union = rb + self.bits - inter
        sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        sim[i] = -1.0
        k = min(int(k), len(sim) - 1)
        top = np.argpartition(-sim, k)[:k]
        return [int(self.ids[j]) for j in top[np.argsort(-sim[top])]]


class Retriever:
    """The shared block table, role matrices materialised on demand, and a neighbour cache.

    Call `nearest(role, mid)`; an empty list means "not available, make the ordinary move".
    """

    def __init__(self, space, log=None, pexec=None):
        self.space = space
        self.log = log or (lambda *_a: None)
        self.pexec = pexec
        self.table: BlockTable | None = None
        self._idx: dict[int, RoleIndex] = {}
        self._nn: dict[tuple, list[int]] = {}
        self._rows = 0
        self.t0 = time.time()
        self.spent = 0.0
        self.stats = {"queries": 0, "hits": 0, "cold": 0, "not_ready": 0, "budget": 0,
                      "roles": 0, "built_s": 0.0, "query_s": 0.0}
        self._thread = threading.Thread(target=self._build, name="retrieve", daemon=True)
        self._thread.start()

    def _build(self) -> None:
        try:
            t = time.time()
            table = BlockTable(self.space.smiles, self.pexec)
            self.stats["built_s"] = time.time() - t
            self.table = table                   # published last: readers see it complete or absent
            self.log(f"retrieve: block table {len(table)} blocks in {self.stats['built_s']:.1f}s")
        except Exception as e:                                                 # noqa: BLE001
            self.log(f"retrieve: block table failed, operator disabled: {type(e).__name__}: {e}")

    # ---- availability --------------------------------------------------------

    def request(self, roles) -> None:
        """Kept for the caller's sake: the table covers every role, so there is nothing to queue."""

    def ready(self, role: int) -> bool:
        return self.table is not None

    def _role(self, role: int) -> RoleIndex | None:
        idx = self._idx.get(role)
        if idx is not None:
            return idx
        pool = self.space.pool.get(role)
        if self.table is None or pool is None or len(pool) < 2:
            return None
        if self._rows + len(pool) > ROLE_CAP and self._idx:
            # Drop the matrices rather than refuse the role: they are a gather away from the table,
            # so rebuilding one costs a memcpy, while refusing it costs the operator for a whole
            # reaction (the bug that made the first share curve meaningless).
            self._idx.clear()
            self._nn.clear()
            self._rows = 0
        idx = RoleIndex(role, pool, self.table)
        self._idx[role] = idx
        self._rows += len(idx)
        self.stats["roles"] = len(self._idx)
        return idx

    # ---- queries -------------------------------------------------------------

    def nearest(self, role: int, mid: int, k: int = NN_K) -> list[int]:
        role, mid = int(role), int(mid)
        self.stats["queries"] += 1
        key = (role, mid)
        got = self._nn.get(key)
        if got is not None:
            self.stats["hits"] += 1
            return got
        if self.table is None:
            self.stats["not_ready"] += 1
            return []
        # A cold query scans the whole packed pool: 68 ms over 462k blocks, in the process that
        # also fits the GP and submits generation work. Cache hits are free and dominate once the
        # elites settle, so the cost is bounded by wall-clock share rather than by a call count.
        if self.spent > QUERY_GRACE + QUERY_SHARE * (time.time() - self.t0):
            self.stats["budget"] += 1
            return []
        t = time.time()
        idx = self._role(role)
        out = idx.nearest(mid, k) if idx is not None else []
        dt = time.time() - t
        self.spent += dt
        self.stats["query_s"] += dt
        if idx is None:
            self.stats["not_ready"] += 1
            return []
        if len(self._nn) > 200_000:
            self._nn.clear()
        self._nn[key] = out
        self.stats["cold"] += 1
        return out
