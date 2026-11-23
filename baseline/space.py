"""The combinatorial space, held in memory, resolved exactly like the validator.

`get_smiles_from_reaction` opens SQLite twice per call and recomputes every
product for each `k`; this module reads the database once at startup and keeps
every building block's SMILES and role mask, so a tuple's full product list
costs one RDKit reaction. The control flow of `react_molecules` and
`react_three_components` is mirrored line for line and the chemistry itself is
delegated to the official helpers, so a name resolves to the same SMILES the
validator will compute (checked by tests/test_space_parity.py).
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np

import nova_miner
from nova_miner.combinatorial_db import reactions as R

import chem

DB_PATH = str(Path(nova_miner.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")


class Space:
    """Reactions, role pools and building blocks of the pinned database."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        self.reactions: dict[int, dict] = {}
        for rid, name, smarts, a, b, c in conn.execute(
                "SELECT rxn_id, name, smarts, roleA, roleB, roleC FROM reactions"):
            self.reactions[int(rid)] = {
                "name": name, "smarts": smarts, "A": R.as_mask(a), "B": R.as_mask(b),
                "C": R.as_mask(c) if c is not None else None}
        self.rxn_ids = sorted(self.reactions)
        roles = sorted({m for r in self.reactions.values()
                        for m in (r["A"], r["B"], r["C"]) if m})
        self.smiles: dict[int, str] = {}
        self.mask: dict[int, int] = {}
        members: dict[int, list[int]] = {m: [] for m in roles}
        for mid, smi, rm in conn.execute("SELECT mol_id, smiles, role_mask FROM molecules"):
            m = R.as_mask(rm)
            if not m:
                continue
            hit = False
            for role in roles:
                if (m & role) == role:
                    members[role].append(mid)
                    hit = True
            if hit:
                self.smiles[mid] = smi
                self.mask[mid] = m
        conn.close()
        self.pool = {role: np.asarray(ids, dtype=np.int64) for role, ids in members.items()}

    # ---- structure of a reaction -------------------------------------------

    def roles(self, rxn: int) -> list[int]:
        r = self.reactions[rxn]
        return [r["A"], r["B"]] + ([r["C"]] if r["C"] else [])

    def arity(self, rxn: int) -> int:
        return 3 if self.reactions[rxn]["C"] else 2

    def pool_sizes(self, rxn: int) -> list[int]:
        return [len(self.pool[m]) for m in self.roles(rxn)]

    # ---- resolution ---------------------------------------------------------

    def products(self, rxn: int, ids) -> list[str]:
        """Every product of a tuple, index = the `k` of its id. [] if it does not react.

        Mirrors reactions.react_molecules / react_three_components exactly."""
        info = self.reactions.get(int(rxn))
        if info is None or any(i not in self.smiles for i in ids):
            return []
        return chem.products(int(rxn), [self.smiles[i] for i in ids], [self.mask[i] for i in ids], info)


def make_name(rxn: int, ids, k: int = 0) -> str:
    """The canonical id for product k. k=0 carries no suffix (the validator's default),
    so a product is never submitted under two strings."""
    base = "rxn:" + ":".join(str(int(x)) for x in (rxn, *ids))
    return base if k == 0 else f"{base}:k{k}"


def parse_name(name: str):
    parsed = R.parse_product_name(name)
    return parsed  # (rxn, [ids], k) or None
