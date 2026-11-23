"""Online expected gain from feasible replacements of a measured portfolio.

The geometry comes only from the current validated portfolio and unscored
in-run candidates. There is no preferred chemical property or fitted state.
"""
from __future__ import annotations

import numpy as np
from rdkit import DataStructs
from scipy.special import ndtr

import portfolio

MAX_CANDIDATES = 768


def greedy_entropy(records, n, tanimoto_max):
    """Entropy before repair, using portfolio.select's exact measured greedy pass.

    Return None if that pass cannot construct n identity- and Morgan-feasible
    members. Only the sort scans all observations; the selected set stops at n.
    """
    if n < 1 or len(records) < n:
        return None
    values = np.asarray([record["y"] for record in records], dtype=np.float64)
    chosen, fps, keys = [], [], set()
    for i in np.argsort(-values):
        record = records[int(i)]
        if record["inchikey"] in keys:
            continue
        fp = portfolio.rule_fp(record)
        if fps and max(DataStructs.BulkTanimotoSimilarity(fp, fps)) >= tanimoto_max:
            continue
        keys.add(record["inchikey"])
        chosen.append(record)
        fps.append(fp)
        if len(chosen) == n:
            return portfolio.entropy_from_counts(portfolio.ap_matrix(chosen).sum(0), n)
    return None


def replacement_options(candidates, incumbents, tanimoto_max, entropy_min, margin=0.002):
    """Return candidate-by-member feasible swaps and their exact entropy.

    Each option describes a single replacement of the current measured set;
    a collection of individually feasible replacements is not a joint promise.
    """
    n = len(incumbents)
    if not candidates or n < 2:
        return np.zeros((len(candidates), n), bool), np.zeros((len(candidates), n))
    A = portfolio.ap_matrix(incumbents).astype(np.float64)
    R = portfolio.ap_matrix(candidates).astype(np.float64)
    counts = A.sum(0)
    H = portfolio.entropy_from_counts(counts, n)
    g0 = portfolio._h(counts + 1, n) - portfolio._h(counts, n)
    g1 = portfolio._h(counts, n) - portfolio._h(counts - 1, n)
    delta = (R @ g0)[:, None] - (A @ g1)[None, :] + R @ (A * (g1 - g0)).T
    entropy = H + delta / portfolio.N_AP
    fps = [portfolio.rule_fp(record) for record in incumbents]
    conflicts = np.asarray([
        np.asarray(DataStructs.BulkTanimotoSimilarity(portfolio.rule_fp(record), fps)) >= tanimoto_max
        for record in candidates
    ])
    feasible = (conflicts.sum(1)[:, None] - conflicts) == 0
    # Observed molecules must never consume another successful Oracle read.
    keys = {record["inchikey"] for record in incumbents}
    feasible &= np.asarray([record["inchikey"] not in keys for record in candidates])[:, None]
    feasible &= entropy > entropy_min + margin
    return feasible, entropy


def expected_improvement(mean, sd, incumbent_values):
    """Marginal Gaussian EI in molecule-score units; no observation noise added."""
    improvement = np.asarray(mean, float)[:, None] - np.asarray(incumbent_values, float)[None, :]
    scale = np.maximum(np.asarray(sd, float), 0.0)[:, None]
    safe_scale = np.maximum(scale, 1e-12)
    z = improvement / safe_scale
    ei = improvement * ndtr(z) + scale * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return np.where(scale > 1e-12, np.maximum(ei, 0.0), np.maximum(improvement, 0.0))


def pick(candidates, mean, sd, incumbents, count, tanimoto_max, entropy_min,
         rng, random_indices=()):
    """Indices into candidates plus aggregate diagnostics, bounded to 768 rows.

    At least half the geometric shortlist can come from the GP-scored random
    exploration slice rather than only high means. Each chosen candidate and
    replacement seat is used once; final measured selection rechecks all rules.
    """
    mean, sd = np.asarray(mean, float), np.asarray(sd, float)
    stats = dict(candidates=0, feasible=0, selected=0)
    if count <= 0 or not candidates or len(incumbents) < 2:
        return [], stats
    if len(candidates) != len(mean) or len(mean) != len(sd):
        raise ValueError("portfolio acquisition prediction alignment")
    finite = np.isfinite(mean) & np.isfinite(sd) & (sd >= 0)
    random_indices = np.asarray(random_indices, dtype=int)
    random_indices = random_indices[(random_indices >= 0) & (random_indices < len(candidates))]
    random_indices = np.unique(random_indices[finite[random_indices]])
    if len(random_indices) > MAX_CANDIDATES // 2:
        random_indices = rng.choice(random_indices, MAX_CANDIDATES // 2, replace=False)
    random_set = set(int(i) for i in random_indices)
    optimistic = np.argsort(-(mean + 2 * sd))
    other = [int(i) for i in optimistic if finite[i] and int(i) not in random_set]
    indices = np.asarray(list(random_indices) + other[:MAX_CANDIDATES - len(random_indices)], dtype=int)
    work = [candidates[i] for i in indices]
    stats["candidates"] = len(work)
    if not work:
        return [], stats
    feasible, _ = replacement_options(work, incumbents, tanimoto_max, entropy_min)
    stats["feasible"] = int(feasible.any(1).sum())
    values = [record["y"] for record in incumbents]
    ei = expected_improvement(mean[indices], sd[indices], values)
    scores = np.where(feasible, ei, -np.inf)
    picked, fps = [], []
    quota = min(count, len(work), len(incumbents))
    for _ in range(len(work)):
        if len(picked) >= quota:
            break
        i, j = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if not np.isfinite(scores[i, j]) or scores[i, j] <= 1e-12:
            break
        fp = portfolio.rule_fp(work[i])
        if fps and max(DataStructs.BulkTanimotoSimilarity(fp, fps)) >= tanimoto_max:
            scores[i, :] = -np.inf
            continue
        picked.append(int(indices[i]))
        fps.append(fp)
        scores[i, :] = -np.inf
        scores[:, j] = -np.inf
    stats["selected"] = len(picked)
    return picked, stats
