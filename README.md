# entity-leakage

Toolkit, corrected splits, and analysis scripts for **"Entity-Level Leakage in
Code Intelligence Benchmarks: Predicting, Measuring, and Correcting
Contamination in Derived-Example Datasets"** (under review).

The toolkit predicts train/test contamination from provenance metadata alone
(before any model is trained), measures it on shipped splits, generates
leakage-safe entity-level splits, and decomposes protocol-level score
inflation into exposure, exploitability, and stratum bias.

## Layout

```
leakage_audit/
  graph.py        sharing graph via union-find; leakage clusters
  metrics.py      closed-form contamination predictor (Prop. 1); measured contamination
  splits.py       leakage-safe cluster k-fold (LPT-balanced) + verifier
  predictor.py    inflation decomposition (Prop. 2); memorization-ceiling probes
  adapters/       bigclonebench (CodeXGLUE format), generic_csv
scripts/
  validate_theory_synthetic.py   Monte Carlo agreement with the closed form (no data needed)
  feasibility_check.py           per-benchmark rho + cluster statistics (Table 5)
  convert_*.py                   corpus converters (Devign, D2A, SStuBs, Tufano, Quatrain)
  prep_bcb.py                    BigCloneBench 30K pool + both split labelings (Sec. 5.6)
  make_devign_splits.py          Devign published + entity-clustered splits
  decompose_patch2.py            exposure/exploitability/stratum-bias decomposition (Sec. 7.1)
  confound_checks.py             bootstrap CIs, composition-matched Delta (Sec. 10.1)
  dose_response.py               multi-seed dose-response with per-bin Wilson CIs (Table 13)
splits/
  corrected entity-level split files for every audited benchmark (see below)
THEORY.md         formal model (paper Section 3)
```

The audit core has no dependencies beyond the Python standard library.
Training scripts require PyTorch and HuggingFace Transformers.

## Corrected splits

`splits/` contains, for each audited benchmark, the entity-level
leakage-safe split used in the paper's controlled comparisons, as
`example_id -> {train, valid, test}` JSON maps:

- Devign: commit-clustered split (published split included for reference)
- Patch assessment (Quatrain corpus): bug-level split and the patch-level
  split it is compared against
- BigCloneBench: 30,000-pair pool with function-disjoint (safe) and
  pair-level (unsafe) labelings

Each split verifies as leakage-safe (or reproduces the paper's measured
contamination) via `leakage_audit.metrics.measured_contamination`.

## Reproducing the paper's tables

| Paper artifact | Command |
| --- | --- |
| Table 5 (exposure audit) | `python scripts/feasibility_check.py <benchmark> <path>` per corpus |
| Tables 6-9 (protocol comparisons) | training scripts under `scripts/` with the split files in `splits/` |
| Section 7.1 decomposition | `python scripts/decompose_patch2.py --contam <map>` |
| Section 10.1 bootstrap + composition matching | `python scripts/confound_checks.py patch\|devign ...` |
| Table 13 (dose-response, all seeds) | `python scripts/dose_response.py --measure jaccard` |
| Appendix A (Monte Carlo) | `python scripts/validate_theory_synthetic.py` |

Per-seed prediction files for every trained run are released with the
artifact so that all stratified quantities are recomputable without GPUs.

## Auditing your own benchmark

```
python scripts/feasibility_check.py csv my_benchmark.csv --test-fraction 0.2
```

CSV format: `example_id,artifact_ids,label,split` with `;`-separated
artifact ids, split in {train,valid,test} (optional). The output reports
cluster statistics, the closed-form predicted contamination for an
example-level split, and (if a split column is present) measured
contamination.
