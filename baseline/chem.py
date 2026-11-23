"""Per-molecule chemistry, run inside worker processes.

A candidate is only kept if the validator's own filter set passes, computed with
the official `nova_miner.utils.filters.passes`. Everything the main process
needs later is computed here once: generic molecular descriptors, the
surrogate's features, the Morgan bits the pairwise-Tanimoto rule uses and the
atom-pair bits the entropy rule uses, and the InChIKey the duplicate rule uses.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from nova_miner.combinatorial_db import reactions as R
from nova_miner.utils.filters import passes
from nova_miner.utils import molecules as _official_molecules

RDLogger.DisableLog("rdApp.*")

# The generator the validator's diversity rule uses (utils/molecules.py: GetAtomPairGenerator(fpSize=2048)).
# Taken from the official module when it exposes it, so the two can never drift; rebuilt identically otherwise.
_ATOM_PAIR_GEN = getattr(_official_molecules, "_ATOM_PAIR_GEN", None)
if _ATOM_PAIR_GEN is None:
    from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator
    _ATOM_PAIR_GEN = GetAtomPairGenerator(fpSize=getattr(_official_molecules, "FINGERPRINT_BITS", 2048))

# Validator rule fingerprints (utils/molecules.py: Morgan r2 2048 for Tanimoto).
_MORGAN_RULE = GetMorganGenerator(radius=2, fpSize=2048)
# Surrogate fingerprint: count-based Morgan r2, folded to FEAT_BITS.
FEAT_BITS = 2048
_MORGAN_COUNT = GetMorganGenerator(radius=2, fpSize=FEAT_BITS)

DESC_NAMES = ("heavy_atoms", "mol_wt", "logp", "tpsa", "hbd", "hba", "rot_bonds",
              "rings", "arom_rings", "fsp3", "heteroatoms", "frac_arom", "charge_abs", "halogens")
N_DESC = len(DESC_NAMES)

_FILTERS: dict | None = None
_REACTIONS: dict | None = None
SPACE = None            # set in the parent before workers fork; read-only in workers


def init_worker(filters: dict, reactions: dict) -> None:
    global _FILTERS, _REACTIONS
    _FILTERS, _REACTIONS = filters, reactions
    import logging
    logging.disable(logging.CRITICAL)   # official helpers log every non-reacting tuple


def _descriptors(mol) -> np.ndarray:
    ha = mol.GetNumHeavyAtoms()
    arom = sum(a.GetIsAromatic() for a in mol.GetAtoms())
    hal = sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in mol.GetAtoms())
    charge = sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
    return np.array([
        ha, Descriptors.MolWt(mol), Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol), Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol), rdMolDescriptors.CalcFractionCSP3(mol),
        ha - sum(a.GetSymbol() == "C" for a in mol.GetAtoms()), arom / max(ha, 1), charge, hal,
    ], dtype=np.float32)


def featurize(smiles: str) -> dict | None:
    """Everything about a product the miner needs, or None if the validator would reject it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not passes(mol, _FILTERS):
        return None
    counts = _MORGAN_COUNT.GetCountFingerprintAsNumPy(mol)       # folded to FEAT_BITS
    idx = np.flatnonzero(counts).astype(np.int32)
    val = np.minimum(counts[idx], 255).astype(np.uint8)
    descriptors = _descriptors(mol)
    if not np.isfinite(descriptors).all():
        return None
    return {
        "smiles": smiles,
        "ha": mol.GetNumHeavyAtoms(),
        "inchikey": Chem.MolToInchiKey(mol),
        "fp_idx": idx, "fp_val": val,
        "rule_bits": np.fromiter(_MORGAN_RULE.GetFingerprint(mol).GetOnBits(), dtype=np.int16),
        "ap_bits": np.fromiter(_ATOM_PAIR_GEN.GetFingerprint(mol).GetOnBits(), dtype=np.int16),
        "desc": descriptors,
    }


def products(rxn: int, smiles: list[str], masks: list[int], info: dict | None = None) -> list[str]:
    """Every product of a tuple, list index = the `k` of its id; [] if it does not react.

    Mirrors reactions.react_molecules / react_three_components line for line and
    delegates the chemistry to the official helpers (tests/test_space_parity.py)."""
    info = info or _REACTIONS[rxn]
    smarts, roleA, roleB, roleC = info["smarts"], info["A"], info["B"], info["C"]
    if len(smiles) == 2:
        r1, r2 = R.validate_and_order_reactants(smiles[0], smiles[1], masks[0], masks[1], roleA, roleB)
        if not r1 or not r2:
            return []
        if rxn == 1:
            p = R.combine_triazole_synthons(r1, r2)
            return [p] if p else []
        return R.enumerate_smarts_products(r1, r2, smarts)
    if len(smiles) != 3:
        return []
    res = R.validate_and_order_reactants(smiles[0], smiles[1], masks[0], masks[1], roleA, roleB,
                                         smiles[2], masks[2], roleC)
    if not all(res):
        return []
    r1, r2, r3 = res
    if rxn == 3:
        cooh = R.combine_triazole_synthons(r1, r2)
        return R.enumerate_smarts_products(cooh, r3, "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]") if cooh else []
    if rxn == 5:
        inter = R.perform_smarts_reaction(r1, r2, "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]")
        return R.enumerate_smarts_products(inter, r3, "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]") if inter else []
    return []


def expand(jobs: list[tuple]) -> tuple[list[dict], dict]:
    """Expand in-run reaction tuples into every valid product variant."""
    out = []
    stat = {"tuples": len(jobs), "products": 0, "invalid": 0, "errors": 0, "last_error": None}
    for rxn, ids, smis, masks, source in jobs:
        try:
            prods = products(rxn, smis, masks)
        except Exception as e:  # noqa: BLE001 - one odd tuple, not a failed job
            stat["errors"] += 1; stat["last_error"] = f"products: {type(e).__name__}: {e}"
            continue
        for k, smi in enumerate(prods):
            stat["products"] += 1
            try:
                rec = featurize(smi)
            except Exception as e:  # noqa: BLE001
                stat["errors"] += 1; stat["last_error"] = f"featurize: {type(e).__name__}: {e}"
                continue
            if rec is None:
                stat["invalid"] += 1
                continue
            rec.update(rxn=rxn, ids=tuple(ids), k=k, source=source)
            out.append(rec)
    return out, stat
