"""Online surrogate: exact GP over molecular fingerprints and descriptors.

Kernel  k(x, x') = a * T(f, f') + b * exp(-|d - d'|^2 / (2 l^2 D))  (+ noise on the diagonal)
  T  = continuous Tanimoto on count-Morgan-r2 vectors  f.f' / (|f|^2 + |f'|^2 - f.f')
  d  = standardised descriptors (heavy atoms, logP, TPSA, ...), D = their count

Fingerprint and generic descriptor similarities carry no preferred molecular
properties or reaction identities. Responses and descriptors are standardised;
(a, b, l, regularization) are fitted by marginal likelihood on observations
from this run. Everything is numpy/scipy, with no fitted state carried in.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, solve_triangular

GRID = [(a, 1.0 - a, l, nz) for a in (0.35, 0.6, 0.85) for l in (0.7, 1.4) for nz in (0.01, 0.15)]


def to_csr(recs, n_bits: int) -> sparse.csr_matrix:
    """Count fingerprints of records -> CSR float32 (n x n_bits)."""
    idx = [r["fp_idx"] for r in recs]
    val = [r["fp_val"] for r in recs]
    lens = np.fromiter((len(i) for i in idx), dtype=np.int64, count=len(idx))
    indptr = np.concatenate([[0], np.cumsum(lens)])
    ind = np.concatenate(idx) if idx else np.zeros(0, np.int32)
    dat = np.concatenate(val).astype(np.float32) if val else np.zeros(0, np.float32)
    return sparse.csr_matrix((dat, ind, indptr), shape=(len(recs), n_bits))


def tanimoto(A: sparse.csr_matrix, B: sparse.csr_matrix, a_sq=None, b_sq=None) -> np.ndarray:
    dot = (A @ B.T).toarray().astype(np.float64)
    a_sq = np.asarray(A.multiply(A).sum(1)).ravel() if a_sq is None else a_sq
    b_sq = np.asarray(B.multiply(B).sum(1)).ravel() if b_sq is None else b_sq
    return dot / np.maximum(a_sq[:, None] + b_sq[None, :] - dot, 1e-9)


def _sqdist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x2 = (X * X).sum(1)[:, None]
    y2 = (Y * Y).sum(1)[None, :]
    return np.maximum(x2 + y2 - 2.0 * X @ Y.T, 0.0)


class GP:
    def __init__(self, n_bits: int, max_obs: int = 3500):
        self.n_bits = n_bits
        self.max_obs = max_obs
        self.trained = False
        self.hyper = GRID[4]

    # ---- fitting -----------------------------------------------------------

    def fit(self, recs, y, tune: bool = True) -> bool:
        y = np.asarray(y, dtype=np.float64)
        if len(recs) < 12:
            self.trained = False
            return False
        if len(recs) > self.max_obs:
            # keep the best half and a random half of the rest: the top of the
            # space is what selection has to rank correctly
            order = np.argsort(-y)
            top = order[: self.max_obs // 2]
            rest = np.random.default_rng(len(y)).choice(order[self.max_obs // 2:], self.max_obs - len(top), replace=False)
            keep = np.sort(np.concatenate([top, rest]))
            recs = [recs[i] for i in keep]
            y = y[keep]
        X = to_csr(recs, self.n_bits)
        D = np.stack([r["desc"] for r in recs]).astype(np.float64)
        self.d_mu = D.mean(0)
        self.d_sd = D.std(0) + 1e-6
        Dz = (D - self.d_mu) / self.d_sd
        self.y_mu = float(y.mean())
        self.y_sd = float(y.std()) or 1.0
        yz = (y - self.y_mu) / self.y_sd
        x_sq = np.asarray(X.multiply(X).sum(1)).ravel()
        T = tanimoto(X, X, x_sq, x_sq)
        S = _sqdist(Dz, Dz) / Dz.shape[1]
        grid = GRID if (tune or not self.trained) else [self.hyper]
        best = None
        for (a, b, l, nz) in grid:
            K = a * T + b * np.exp(-S / (2 * l * l))
            K[np.diag_indices_from(K)] += nz
            try:
                c, low = cho_factor(K, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                continue
            alpha = cho_solve((c, low), yz, check_finite=False)
            nll = 0.5 * yz @ alpha + np.log(np.diag(c)).sum()
            if best is None or nll < best[0]:
                best = (nll, (a, b, l, nz), c, alpha)
        if best is None:
            self.trained = False
            return False
        _, self.hyper, self.L, self.alpha = best
        self.X, self.x_sq, self.Dz = X, x_sq, Dz
        self.trained = True
        return True

    # ---- prediction ---------------------------------------------------------

    def _kstar(self, recs):
        a, b, l, _ = self.hyper
        Xs = to_csr(recs, self.n_bits)
        Ds = (np.stack([r["desc"] for r in recs]).astype(np.float64) - self.d_mu) / self.d_sd
        T = tanimoto(Xs, self.X, None, self.x_sq)
        S = _sqdist(Ds, self.Dz) / Ds.shape[1]
        return a * T + b * np.exp(-S / (2 * l * l))

    def predict(self, recs, with_std: bool = True, chunk: int = 4000):
        n = len(recs)
        if not self.trained or n == 0:
            return np.zeros(n), np.ones(n)
        mu = np.empty(n)
        sd = np.empty(n)
        a, b, _, _ = self.hyper
        prior = a + b
        for s in range(0, n, chunk):
            Ks = self._kstar(recs[s:s + chunk])
            mu[s:s + chunk] = Ks @ self.alpha
            if with_std:
                v = solve_triangular(self.L, Ks.T, lower=True, check_finite=False)
                sd[s:s + chunk] = np.sqrt(np.maximum(prior - (v * v).sum(0), 1e-9))
        mu = mu * self.y_sd + self.y_mu
        sd = sd * self.y_sd if with_std else np.full(n, np.nan)
        return mu, sd

    def noise_sd(self) -> float:
        """Fitted surrogate residual scale; this is not a repeated-read noise model."""
        return math.sqrt(self.hyper[3]) * self.y_sd if self.trained else float("nan")
