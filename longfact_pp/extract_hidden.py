"""Hidden states of the LongFact++ answers, read by the model that wrote them.

Each answer is teacher-forced through its generator (Llama-3.1-8B-Instruct or
Llama-3.3-70B-Instruct, the latter in 4-bit as in the QA experiments) with the
chat template it was generated under.  For every answer token we keep the
hidden state that predicts it, at the requested transformer layers.

Output (same format as gen/job1.sh):
  Data/<model_name>/longfact_pp/{train,validation}_layer{L//2}.pkl
  {id: {"generated_answer": [{"h_out": FloatTensor(T, 1, d) fp16, "tokens": LongTensor(T)}]}}
The file index is L // 2, as for the QA cells (Detection/main.py iterates over
these indices).  Files are written in pickled chunks of 50 answers;
Detection/utils.py reads all chunks.

Usage:
  python longfact_pp/extract_hidden.py --model_name llama3_8b_Instruct --layers 0,2,4,...,32
  python longfact_pp/extract_hidden.py --model_name llama3_70b --layers 0,4,8,...,80
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gen", "generation"))
from uncertainty.models.huggingface_models import HuggingfaceModel  # noqa: E402

HF_ID = {"llama3_8b_Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
         "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct"}
CHUNK = 50


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, choices=list(HF_ID))
    p.add_argument("--data_dir", default="Data")
    p.add_argument("--layers", required=True, help="comma-separated transformer layers (0 = embeddings)")
    a = p.parse_args()
    out_dir = os.path.join(a.data_dir, a.model_name, "longfact_pp")
    recs = [json.loads(l) for l in open(os.path.join(out_dir, "records.jsonl"))]
    # the detector reads a "train" file and a "validation" file holding validation| and test| ids
    splits = {"train": [r for r in recs if r["split"] == "train"],
              "validation": [r for r in recs if r["split"] != "train"]}
    layers = [int(x) for x in a.layers.split(",")]

    hf = HuggingfaceModel(HF_ID[a.model_name], stop_sequences=None, max_new_tokens=1)
    model, tok = hf.model, hf.tokenizer
    model.eval()
    dev = model.get_input_embeddings().weight.device

    for split, rs in splits.items():
        finals = {L: os.path.join(out_dir, f"{split}_layer{L // 2}.pkl") for L in layers}
        if all(os.path.exists(f) for f in finals.values()):
            continue
        fh = {L: open(finals[L] + ".partial", "wb") for L in layers}
        buf = {L: {} for L in layers}
        t0 = time.time()
        for k, r in enumerate(rs):
            text = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                           add_generation_prompt=True, tokenize=False)
            p_ids = tok(text, add_special_tokens=False).input_ids
            a_ids = tok(r["answer"], add_special_tokens=False).input_ids
            ids = torch.tensor([list(p_ids) + a_ids[:-1]], device=dev)
            P, n = len(p_ids), len(a_ids)
            with torch.no_grad():
                o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            for L in layers:
                buf[L][r["id"]] = {"generated_answer": [{
                    "h_out": o.hidden_states[L][0, P - 1:P - 1 + n].to("cpu", torch.float16).unsqueeze(1).clone(),
                    "tokens": torch.tensor(a_ids)}]}
            del o
            if (k + 1) % CHUNK == 0 or k + 1 == len(rs):
                for L in layers:
                    pickle.dump(buf[L], fh[L])
                    buf[L] = {}
            if (k + 1) % 100 == 0:
                print(f"[{split}] {k + 1}/{len(rs)} {time.time() - t0:.0f}s", flush=True)
        for L in layers:
            fh[L].close()
            os.replace(finals[L] + ".partial", finals[L])
        print(f"[{split}] done {len(rs)} answers in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
