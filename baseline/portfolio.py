"""Choose the 100 molecules to submit, under the validator's whole-entry rules.

Rules (neurons/validator/validity.py): 100 distinct ids; every molecule passes
the filters (guaranteed upstream); no two share an InChIKey; no pair has
Morgan-r2-2048 Tanimoto >= tanimoto_max; atom-pair-2048 entropy > threshold.

The validator's entry score is the mean measured selectivity, so `values`
are finite target-minus-antitarget scores observed during this run.

Method: greedy by value under the Tanimoto/InChIKey rules; then single swaps
(member i out, candidate j in). For all (i, j) at once the entropy change is

  2048*dH[i,j] = R_j.g0 - A_i.g1 + (A_i * (g1 - g0)).R_j
  g0_b = h(c_b+1) - h(c_b),  g1_b = h(c_b) - h(c_b-1),  h(k) = binary entropy of k/n

(one matrix product), so a repair step picks the swap buying the most entropy
per unit of value lost, and an improvement step the swap buying the most value
while staying above the line. Entropy is the same formula as
nova_miner.utils.molecules.compute_fingerprint_entropy (tests/test_portfolio.py).
"""
from __future__ import annotations

import time

import numpy as np
from rdkit import DataStructs

N_AP = 2048


def _h(k: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(k, dtype=np.float64) / n
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    q = p[m]
    out[m] = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    return out


def entropy_from_counts(counts: np.ndarray, n: int) -> float:
    return float(_h(counts, n).mean())


def ap_matrix(recs) -> np.ndarray:
    M = np.zeros((len(recs), N_AP), dtype=np.float32)
    for i, r in enumerate(recs):
        M[i, r["ap_bits"]] = 1.0
    return M


def rule_fp(rec):
    fp = rec.get("_rule_fp")
    if fp is None:
        fp = DataStructs.ExplicitBitVect(2048)
        fp.SetBitsFromList([int(b) for b in rec["rule_bits"]])
        rec["_rule_fp"] = fp
    return fp


def select(recs, values, n: int, tanimoto_max: float, entropy_min: float,
           margin: float = 0.002, max_reserve: int = 1500, max_swaps: int = 400,
           seconds: float = 0.0):
    """-> (indices into recs, entropy) or (None, None) if no admissible set was found.

    `seconds` (>0) bounds the swap loop. If repairs cannot finish in that budget,
    return no portfolio and preserve the previous validated checkpoint."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values)
    chosen, fps, keys = [], [], {}
    reserve = []
    for i in order:
        i = int(i)
        r = recs[i]
        if len(chosen) < n:
            if r["inchikey"] in keys:
                continue
            fp = rule_fp(r)
            if fps and max(DataStructs.BulkTanimotoSimilarity(fp, fps)) >= tanimoto_max:
                reserve.append(i)          # may become feasible when its neighbour leaves
                continue
            keys[r["inchikey"]] = len(chosen)
            chosen.append(i); fps.append(fp)
        else:
            reserve.append(i)
        if len(chosen) >= n and len(reserve) >= max_reserve:
            break
    if len(chosen) < n:
        return None, None
    reserve = reserve[:max_reserve]
    m = len(reserve)
    target = entropy_min + margin
    A = ap_matrix([recs[i] for i in chosen]).astype(np.float64)
    c = A.sum(0)
    H = entropy_from_counts(c, n)
    if m == 0:
        return (chosen, H) if H > target else (None, None)
    R = ap_matrix([recs[j] for j in reserve]).astype(np.float64)
    rfps = [rule_fp(recs[j]) for j in reserve]
    # conflict[j, i]: reserve j too similar to member i
    conflict = np.zeros((m, n), dtype=bool)
    for k, fp in enumerate(fps):
        conflict[:, k] = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, rfps)) >= tanimoto_max
    res_keys = [recs[j]["inchikey"] for j in reserve]
    vres = values[reserve]

    def feasible():
        ccount = conflict.sum(1)
        # swap (i, j) is Tanimoto-feasible iff j's only conflicts are with i
        ok = (ccount[None, :] - conflict.T) == 0                      # n x m
        dup = np.array([keys.get(k, -1) for k in res_keys])           # member sharing j's key
        has = dup >= 0
        ok[:, has] &= (np.arange(n)[:, None] == dup[None, has])
        return ok

    swaps = 0
    t_end = (time.time() + seconds) if seconds > 0 else None
    while swaps < max_swaps:
        if t_end is not None and time.time() > t_end:
            break
        vsel = values[chosen]
        g0 = _h(c + 1, n) - _h(c, n)
        g1 = _h(c, n) - _h(c - 1, n)
        dH = (R @ g0)[None, :] - (A @ g1)[:, None] + (A * (g1 - g0)) @ R.T
        dH /= N_AP
        ok = feasible()
        loss = vsel[:, None] - vres[None, :]
        if H <= target:
            cand = ok & (dH > 1e-12)
            if not cand.any():
                break
            better = cand & (loss <= 0)
            if better.any():
                score = np.where(better, dH, -np.inf)
            else:
                score = np.where(cand, dH / np.maximum(loss, 1e-9), -np.inf)
        else:
            cand = ok & (loss < -1e-12) & (H + dH > target)
            if not cand.any():
                break
            score = np.where(cand, -loss, -np.inf)
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        # apply swap: member i <-> reserve j
        out_idx, in_idx = chosen[i], reserve[j]
        del keys[recs[out_idx]["inchikey"]]
        keys[res_keys[j]] = i
        c = c - A[i] + R[j]
        A[i], R[j] = R[j].copy(), A[i].copy()
        chosen[i], reserve[j] = in_idx, out_idx
        res_keys[j] = recs[out_idx]["inchikey"]
        vres[j] = values[out_idx]
        fps[i], rfps[j] = rfps[j], fps[i]
        # new member i's conflicts with every reserve item; reserve j (old member) vs members
        conflict[:, i] = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], rfps)) >= tanimoto_max
        conflict[j, :] = np.asarray(DataStructs.BulkTanimotoSimilarity(rfps[j], fps)) >= tanimoto_max
        H = entropy_from_counts(c, n)
        swaps += 1
    if H <= target:
        return None, None
    return chosen, H
