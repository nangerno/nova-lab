# fixed_v1

Frozen challenge for the first experiment series. Do not edit mid-series.

- Source epoch: Blueprint **20779**
- Target: `F6T8J6` (359 aa)
- Antitargets: `A0A8C9UVP0` (328 aa), `A0A8K1X527` via UniParc `UPI001E74D9DE` (320 aa; UniProtKB entry deleted)
- Config: `antitarget_weight=0.9`, `entropy_min_threshold=0.35`, `tanimoto_max_threshold=0.9`, `num_molecules=100`, filters from nova-blueprint `config.yaml`

Regenerate only if intentionally starting a new series:

```bash
python harness/make_challenge.py --out challenges/fixed_v1/input.json
```
