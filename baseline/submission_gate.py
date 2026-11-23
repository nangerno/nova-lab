"""The validator's validity rules, applied before result.json is overwritten.

A result.json the validator rejects scores nothing for the whole epoch, and a
later valid write can only happen if the search keeps improving.  So a
checkpoint is published only when it passes every rule the validator applies
(upstream neurons/validator/validity.py):

  * exactly ``num_molecules`` names, no duplicate names;
  * every name resolves to a SMILES that RDKit parses and passes the filters;
  * no two molecules share an InChIKey;
  * no pair has Morgan (r=2, 2048-bit) Tanimoto >= ``tanimoto_max_threshold``;
  * atom-pair fingerprint entropy strictly above ``entropy_min_threshold``;
  * every name uses the allowed reaction, when the challenge names one.

The library functions are used where ``nova_miner`` ships them; the Tanimoto
check mirrors upstream ``utils.molecules.find_too_similar_pairs``, which the
sandbox library does not include.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

try:
    from nova_miner.utils.molecules import (
        compute_fingerprint_entropy, find_chemically_identical, get_smiles)
    from nova_miner.utils.filters import check as check_filters
    from nova_miner.utils.filters import validate_thresholds
except ImportError:  # pragma: no cover - the sandbox always ships nova_miner
    compute_fingerprint_entropy = find_chemically_identical = None
    get_smiles = check_filters = validate_thresholds = None

_MORGAN = GetMorganGenerator(radius=2, fpSize=2048)


def too_similar_pairs(smiles_list, threshold):
    """(i, j, similarity) for every pair at or above ``threshold``."""
    if threshold >= 1.0 + 1e-12 or len(smiles_list) < 2:
        return []
    fps, indices = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(_MORGAN.GetFingerprint(mol))
        indices.append(i)
    offending = []
    for k in range(len(fps) - 1):
        for offset, sim in enumerate(BulkTanimotoSimilarity(fps[k], fps[k + 1:])):
            if sim >= threshold:
                offending.append((indices[k], indices[k + 1 + offset], sim))
    return offending


def _reaction_allowed(name, allowed):
    """Mirror of upstream utils.reactions.is_reaction_allowed."""
    if name.startswith("rxn:"):
        try:
            return allowed == f"rxn:{int(name.split(':')[1])}"
        except (IndexError, ValueError):
            return False
    return allowed == "savi"


def reject_reason(names, config):
    """None if the validator would accept ``names``, else the first failure."""
    if get_smiles is None:
        return "nova_miner unavailable"
    required = int(config["num_molecules"])
    if len(names) != required:
        return f"count {len(names)} != {required}"
    if len(set(names)) != len(names):
        return "duplicate names"
    allowed = config.get("allowed_reaction")
    thresholds = validate_thresholds(config.get("filters"))
    smiles = []
    for name in names:
        if allowed is not None and not _reaction_allowed(str(name), allowed):
            return f"{name} uses a disallowed reaction"
        # Resolve exactly as the validator does, not from the search's own
        # construction, so a construction difference cannot pass the gate.
        smi = get_smiles(name)
        if not smi:
            return f"{name} has no SMILES"
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return f"{name} does not parse"
        failures = check_filters(mol, thresholds)
        if failures:
            return f"{name} fails filter {failures[0]}"
        smiles.append(smi)
    if find_chemically_identical(smiles):
        return "chemically identical molecules"
    pairs = too_similar_pairs(smiles, config.get("tanimoto_max_threshold", 1.0))
    if pairs:
        i, j, sim = pairs[0]
        return f"{len(pairs)} pairs at Tanimoto >= threshold ({names[i]}~{names[j]} {sim:.3f})"
    entropy = compute_fingerprint_entropy(smiles)
    if not entropy > config["entropy_min_threshold"]:
        return f"entropy {entropy:.4f} <= {config['entropy_min_threshold']}"
    return None
