# Results skeleton (numbers as of 2026-07-20; all runs warmup config unless noted)

## RQ1: Prevalence and predictability of entity-level exposure
Table: benchmark | mechanism | rho_pred | rho_measured (protocol) | Thm1 error
- CodeNet (EMSE, motivating)  | hub (submission) | ~0.99  | 0.973-0.984 (pair-level CV) | <0.02
- Devign (CodeXGLUE split)    | sibling (commit) | 0.673  | 0.650 (shipped split)       | 0.023
- Quatrain/Ye patches         | hub (bug)        | 0.869  | 0.873 (patch-level 70/10/20)| 0.004
- BigCloneBench (CodeXGLUE)   | hub (function)   | 1.000 (re-split) | 0.000 (shipped) | n/a (negative control)
- [D2A, SStuBs, Tufano BFP: pending feasibility]
Claim: Thm1 predicts protocol contamination from provenance metadata alone, within 0.004-0.023 on real shipped/standard protocols.

## RQ2: Does exposure inflate measured performance? (Delta)
- Quatrain: patch-level F1 0.858+-0.003 vs bug-level 0.650+-0.038 -> Delta F1 = +0.208.
  Stratified (hub examples only, the hard stratum): contam-hub F1 ~0.58 vs unseen-hub ~0.29 -> +~0.30.
  Minority recall on contaminated positives: 0.464 (headline F1 conceals it).
- Devign: pub 0.648+-0.005 vs cluster 0.608+-0.036 -> Delta acc = +0.040.
  BUT composition: pub strata contam 0.701 / clean 0.543 vs clean-stratum base rate 0.549
  -> zero skill outside contamination. Placebo: cluster model on same ids: 0.594/0.609 (flat).
Claim: exposure != exploitability; Delta ranges 0.04-0.21 (overall) and up to 0.30 (stratified) across mechanisms.

## RQ3: Mechanisms of exploitation (Quatrain deep-dive + taxonomy)
- Structural label shortcut: singleton bugs 100% developer/label-1; hub bugs 94% tool/label-0.
  Both protocols score singletons ~0.92-0.98 F1 -> construct-validity finding independent of leakage.
- Channel separation: hub label-prior oracle F1 0.474 (ruled out as F1 driver);
  exact near-dups n=62 only; soft-sibling similarity dose-response:
  acc 0.742 / 0.968 / 0.988 / 0.990 / 1.000 across Jaccard bins [0-.3/.3-.5/.5-.7/.7-.9/.9-1].
- Devign mechanism: context (commit) exposure without label leak -> skill concentration, small net Delta.

## RQ4: Estimation methodology
- Within-split stratification is confounded (Devign: contam stratum 72% qemu, base 0.611 vs clean 50/50, 0.549).
- Cluster split = unbiased estimator; the placebo (same ids, two regimes) isolates cause.
- Warmup note: 2/10 pre-warmup runs collapsed to majority; all reported runs single config; footnote.

## Pending for full paper
- [ ] D2A / SStuBs / Tufano BFP feasibility rows
- [ ] Second encoder (UniXcoder) on Devign + Quatrain
- [ ] solve_m_gen reconstruction on 3-5 published papers' reported numbers
- [ ] Hypergeometric Thm1 (appendix); role-aware sharing note
- [ ] Dup finding: 599/9135 exact (bug,patch) dups in shipped Quatrain file
