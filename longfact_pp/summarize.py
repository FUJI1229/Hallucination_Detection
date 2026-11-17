"""Mean test AUC over seeds at the validation-selected layer.

The layer is chosen by the validation AUC averaged over seeds; the reported
number is the mean (and SD) of the test AUC at that layer.
Reads <save_dir>/<model_name>/longfact_pp_<pool>_s<seed>_summary.json.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True)
    p.add_argument("--model_name", required=True)
    a = p.parse_args()
    for pool in ("max", "mean", "attention", "gated_attention"):
        files = sorted(glob.glob(os.path.join(a.save_dir, a.model_name, f"longfact_pp_{pool}_s*_summary.json")))
        if not files:
            continue
        runs = [json.load(open(f))["layers"] for f in files]
        layers = set.intersection(*(set(r) for r in runs))
        sel = max(layers, key=lambda L: np.mean([r[L]["val_auc_best"] for r in runs]))
        t = [r[sel]["test_auc_best"] for r in runs]
        print(f"{a.model_name} {pool:16s} layer idx {sel} ({len(files)} seeds): "
              f"test AUC {np.mean(t):.3f} +- {np.std(t):.3f}")


if __name__ == "__main__":
    main()
