# entity-leakage

Toolkit + theory for **"Entity-Level Leakage in Code Intelligence
Benchmarks: A Theory of Derived-Example Datasets and a Cross-Benchmark
Audit"** (target: TOSEM).

## Layout

```
leakage_audit/
  graph.py        sharing graph via union-find; leakage clusters
  metrics.py      Theorem 1 predictor; measured contamination
  splits.py       leakage-safe cluster k-fold (LPT-balanced) + verifier
  predictor.py    Theorem 2 mixture model; memorization-ceiling probes
  adapters/       bigclonebench (CodeXGLUE format), generic_csv
scripts/
  validate_theory_synthetic.py   Monte Carlo validation (no data needed)
  feasibility_check.py           per-benchmark rho + cluster stats
THEORY.md         formal model (paper Section 3 skeleton)
```

No dependencies beyond the standard library.

## Validated so far (2026-07-18)

**Theorem 1** — |predicted − measured| < 0.001 across synthetic cluster
regimes; CodeNet-like regime predicts ρ≈0.991 vs 97.3–98.4% measured in
the EMSE paper.

**Theorem 2** — mixture identity holds on a simulated memorizer
(Δ=+0.33 in the CodeNet-like regime, matching the observed +0.329).

**BigCloneBench (CodeXGLUE, real published splits, 1.73M pairs)** —
published split is function-disjoint: ρ_measured = 0 (negative
control). But ρ_predicted = 1.0 under pair-level random splitting: any
re-split of BCB pairs (custom CV in follow-up papers) is fully
contaminated. → audit entry: "safe as shipped, catastrophically fragile
to protocol deviation."

## Benchmark queue (run on the R740)

| # | Benchmark | Entity key | Expected mechanism | Data source |
|---|-----------|-----------|--------------------|-------------|
| 1 | Devign (CodeXGLUE defect) | commit_id (in original function.json) | sibling derivation | orig. release (GDrive) |
| 2 | Patch assessment (APPT/Quatrain-style) | bug id (Chart_1 …) | hub reuse, k≫10 → predicted Δ larger than CodeNet | GitHub repos |
| 3 | D2A | commit | sibling | IBM release |
| 4 | SStuBs | repo/file | sibling | GitHub |
| 5 | Tufano BFP | method/repo | sibling | GitHub |
| 6 | BigCloneBench | function id | hub | DONE (negative control) |

Per benchmark: (1) converter → generic CSV, (2) `feasibility_check.py`,
(3) if ρ_pred > 0.5: retrain CodeBERT under published vs cluster split,
(4) compare measured Δ to Theorem 2 interval.

## Quick start

```bash
python scripts/validate_theory_synthetic.py
python scripts/feasibility_check.py bigclonebench /path/to/codexglue/dataset
python scripts/feasibility_check.py csv my_benchmark.csv --test-fraction 0.2
```

CSV format: `example_id,artifact_ids,label,split` with `;`-separated
artifact ids, split ∈ {train,valid,test} (optional).
