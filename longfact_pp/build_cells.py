"""Build the LongFact++ cells: 1,000 train / 1,000 validation / 1,500 test.

Source: obalcells/longfact-augmented-annotations (Obeso et al., 2026), which
holds web-verified long-form answers of Llama-3.1-8B-Instruct and
Llama-3.3-70B-Instruct.  Every entity span of an answer was checked against
web search by the dataset's annotation pipeline; we do not relabel anything.

Label (frac10): an answer is hallucinated iff at least 10% of its checked
entity spans are "Not Supported".  Variants anyns (>= 1 such span) and frac20
(share >= 20%) are written to labels_<name>.jsonl for reference.

Splits (fixed seeds; every prompt occurs once in the dataset, and its splits
share no prompt):
  test        = the dataset's test answers       + train answers up to 1,500
  validation  = the dataset's validation answers + train answers up to 1,000
  train       = 500 factual + 500 hallucinated answers drawn from the
                remaining train answers (class-balanced, so the loss needs
                no positive-class weight)
Validation and test keep the natural class ratio.

Output (Data/<model_name>/longfact_pp/):
  records.jsonl                  one line per answer with its split and labels
  qa_train.jsonl, qa_validation.jsonl
  acc_train2.jsonl, acc_validation2.jsonl   accuracy = 1 - hallucinated
The "validation" files hold both the validation| and test| ids; the detector
separates them by id prefix (Detection/main.py --fixed_split).

Usage:
  python longfact_pp/build_cells.py --model_name llama3_8b_Instruct
  python longfact_pp/build_cells.py --model_name llama3_70b
"""
import argparse
import json
import os
import random

import pandas as pd
from huggingface_hub import snapshot_download

REPO = "obalcells/longfact-augmented-annotations"
SUBSET = {"llama3_8b_Instruct": "Meta-Llama-3.1-8B-Instruct", "llama3_70b": "Llama-3.3-70B-Instruct"}
N = {"train": 1000, "validation": 1000, "test": 1500}


def rows(root, sub, split):
    df = pd.read_parquet(os.path.join(root, sub, f"{split}-00000-of-00001.parquet"))
    out = []
    for i, r in df.iterrows():
        ann = list(r.annotations)
        ns = sum(a["label"] == "Not Supported" for a in ann)
        n = max(len(ann), 1)
        conv = list(r.conversation)
        out.append({"src": f"{split}|{i}", "subset": r.subset, "question": conv[0]["content"],
                    "answer": conv[-1]["content"], "n_spans": len(ann), "n_not_supported": ns,
                    "labels": {"frac10": int(ns / n >= 0.1), "anyns": int(ns > 0),
                               "frac20": int(ns / n >= 0.2)}})
    return out


def build(root, sub):
    rng = random.Random(42)
    tr = rows(root, sub, "train")
    rng.shuffle(tr)
    test, val = rows(root, sub, "test"), rows(root, sub, "validation")
    k = N["test"] - len(test)
    test += tr[:k]
    m = N["validation"] - len(val)
    val += tr[k:k + m]
    rest = tr[k + m:]
    neg = [r for r in rest if r["labels"]["frac10"] == 0][:N["train"] // 2]
    pos = [r for r in rest if r["labels"]["frac10"] == 1][:N["train"] // 2]
    train = neg + pos
    random.Random(43).shuffle(train)
    return {"train": train, "validation": val, "test": test}


def write(out, cells):
    os.makedirs(out, exist_ok=True)
    for split, rs in cells.items():
        for j, r in enumerate(rs):
            r["id"], r["split"] = f"{split}|{j}", split
    with open(os.path.join(out, "records.jsonl"), "w") as f:
        for rs in cells.values():
            for r in rs:
                f.write(json.dumps(r) + "\n")
    files = {"train": cells["train"], "validation": cells["validation"] + cells["test"]}
    for split, rs in files.items():
        with open(os.path.join(out, f"qa_{split}.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps({"id": r["id"], "question": r["question"],
                                    "generated_answer": [{"pred_ans": r["answer"]}],
                                    "n_spans": r["n_spans"], "n_not_supported": r["n_not_supported"]}) + "\n")
        with open(os.path.join(out, f"acc_{split}2.jsonl"), "w") as f:
            for r in rs:
                f.write(json.dumps({"id": r["id"], "accuracy": [1.0 - r["labels"]["frac10"]]}) + "\n")
    for lab in ("frac10", "anyns", "frac20"):
        with open(os.path.join(out, f"labels_{lab}.jsonl"), "w") as f:
            for rs in cells.values():
                for r in rs:
                    f.write(json.dumps({"id": r["id"], "accuracy": [1.0 - r["labels"][lab]]}) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, choices=list(SUBSET))
    p.add_argument("--data_dir", default="Data")
    a = p.parse_args()
    root = snapshot_download(REPO, repo_type="dataset")
    cells = build(root, SUBSET[a.model_name])
    write(os.path.join(a.data_dir, a.model_name, "longfact_pp"), cells)
    for split, rs in cells.items():
        n = len(rs)
        print(f"{a.model_name} {split:10s} n={n}  hallucinated (frac10) {sum(r['labels']['frac10'] for r in rs) / n:.1%}")


if __name__ == "__main__":
    main()
