"""Feasibility check: compute rho and cluster stats for a benchmark.

Usage:
  python scripts/feasibility_check.py bigclonebench <dataset_dir>
  python scripts/feasibility_check.py csv <file.csv> [--test-fraction 0.2]

Decision rule (paper go/no-go per benchmark):
  rho_predicted_random_split > 0.5  -> strong audit candidate
  published_split_rho_measured > 0  -> confirmed leaky published protocol
  both ~0                           -> negative control (keep 1-2 of these)
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

from leakage_audit.metrics import contamination_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", choices=["bigclonebench", "csv"])
    ap.add_argument("path")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    args = ap.parse_args()

    if args.adapter == "bigclonebench":
        from leakage_audit.adapters.bigclonebench import load

        graph, split, _ = load(args.path)
        n_test = sum(1 for v in split.values() if v == "test")
        tf = n_test / len(split) if split else args.test_fraction
    else:
        from leakage_audit.adapters.generic_csv import load

        graph, split, _ = load(args.path)
        tf = args.test_fraction
        if split:
            n_test = sum(1 for v in split.values() if v == "test")
            tf = n_test / len(split)

    rep = contamination_report(graph, split, test_fraction=tf)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
