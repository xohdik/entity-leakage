# Formal Model — Entity-Level Leakage in Derived-Example Datasets

Paper Section 3 skeleton. Notation chosen to generalize the EMSE
CodeNet instance without referencing it until validation.

## 3.1 Derived-example datasets

- Artifact space **A**: raw units mined from a source system
  (submissions, functions, commits, patches, bug reports).
- Derivation function **D**: maps artifact tuples to labeled examples.
  A benchmark is a finite set X of examples; each x in X carries a
  provenance multiset art(x) ⊆ A.
- **Sharing relation**: x ~ y iff art(x) ∩ art(y) ≠ ∅ (direct sharing).
- **Leakage clusters**: connected components of the sharing graph
  G_L = (X, ~). Computed via union-find over artifact IDs.
- **Entity function** ent: X → E is a *sound clustering key* iff
  x ~ y ⟹ ent(x) = ent(y). (Problem ID is sound for CodeNet pair
  mining; commit ID for Devign-style sibling derivation; bug ID for
  patch assessment.) Soundness is checkable: ent is sound iff every
  cluster is ent-constant.

## 3.2 Split safety

**Def (leakage-safe).** A train/test split is leakage-safe iff every
leakage cluster lies entirely on one side.

**Lemma 1 (degeneracy).** Grouped CV with singleton groups (one group
per example) is equivalent to a uniform random split over examples and
is not leakage-safe whenever some cluster has size ≥ 2.

Two contamination notions:
- **strict** (cluster-level): test x contaminated iff its cluster
  intersects train — sound upper bound on any artifact-mediated
  memorization pathway;
- **direct**: test x contaminated iff ∃ train y with x ~ y — the
  mechanism actually exploited by hub memorization.
Report both; strict ≥ direct always. (BigCloneBench shows why the
distinction matters: within-split hub density makes strict clusters
enormous, but the published split is disjoint under both notions.)

## 3.3 Theorem 1 — expected contamination under example-level splitting

Random split, test fraction t, cluster sizes {k_i}. Under Bernoulli(t)
assignment (approximation of the hypergeometric split, error O(k/N)):

P(test example contaminated | cluster size k) = 1 − t^(k−1)

Expected contaminated fraction of the test set (example-weighted):

ρ_pred = Σ_k n_k·k·(1 − t^(k−1)) / Σ_k n_k·k

Monte Carlo validation (scripts/validate_theory_synthetic.py):
|ρ_pred − ρ_measured| < 0.001 across cluster regimes.
External validation: CodeNet-like regime (k∈[3,6], t=0.2) predicts
ρ≈0.991; the EMSE paper measured 97.3–98.4% (Python) under the flawed
pair-level protocol.

## 3.4 Theorem 2 — inflation as a mixture

M_obs = ρ·M_mem + (1−ρ)·M_gen  ⟹  Δ = M_obs − M_gen = ρ·(M_mem − M_gen)

- M_mem: memorization ceiling on contaminated examples, estimated
  model-free by (a) exact-match sibling-label probe (lower bound),
  (b) 1-NN label transfer in a cheap embedding (upper estimate).
  Report Δ as the interval [ρ(M_mem^a − M_gen), ρ(M_mem^b − M_gen)].
- Consequence 1: Δ increases with ρ — predictable from provenance
  metadata alone, before training anything.
- Consequence 2: Δ decreases with M_gen — matches the inverse
  difficulty/inflation pattern across the six CodeNet languages.
- Identification: given a published (leaky) M_obs and estimated ρ,
  M_mem, solve M_gen = (M_obs − ρ·M_mem)/(1 − ρ) — an *honest-score
  reconstruction* for papers we cannot rerun. Undefined at ρ = 1
  (fully contaminated benchmarks are uninformative about M_gen: a
  finding in itself).

## 3.5 Sharing-mechanism taxonomy

1. **Hub reuse** — one artifact appears in many examples
   (accepted submission × k buggy partners; function × k clone pairs;
   bug × k candidate patches).
2. **Sibling derivation** — many examples cut from one artifact
   (functions from one commit/file: Devign, D2A, SStuBs).
3. **Near-duplicate artifacts** — soft sharing not captured by exact
   provenance (clones across contest series). Frontier: bound with
   MinHash similarity ≥ θ as an approximate sharing relation; gives a
   residual-leakage estimate that entity grouping cannot remove.

## Open items to settle before writing

- [ ] Exact hypergeometric version of Theorem 1 (finite-N correction)
      for the appendix.
- [ ] Whether "direct" contamination needs asymmetric roles (memorizing
      the shared correct artifact predicts the label only when the
      artifact's role determines the label — formalize role-aware
      sharing for pairwise tasks).
- [ ] k-fold CV version of Theorem 1 (t = 1/K per fold; average over
      folds is identical by symmetry — one-line remark).
