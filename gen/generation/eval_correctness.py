import json
import re
import os
import argparse
import pickle
import torch
from tqdm import tqdm
from uncertainty.utils import openai as oai


def load_pickle_data(path):
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


# -------------------------
# GPT-based binary correctness metric (Dataset-specific)
# -------------------------
def gpt_metric(predicted_answer, example, model,ask_ref):
    """
    Evaluate by dynamically switching prompts depending on the dataset context.
    """
    question = example.get('question', 'N/A')
    answer = example.get('answer', 'N/A')
    
    prompt = f'We are assessing the quality of answers to the following question: {question}\n'
    if ask_ref:
        prompt += f"The following are expected answers to this question: {answer}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if ask_ref:
        prompt += "On the basis of the given question, expected answer, context and your own knowledge, is the proposed answer correct? Please think carefully and"
    else:
        prompt += "Based on the context of question and your own knowledge, is the proposed answer correct? Please think carefully and"
    
    prompt += " Respond only with yes or no.\nResponse:"
    try:
        response = oai.predict(
            prompt, 
            model=model
        )

        if response is None:
            return None

        clean_resp = response.strip().lower()
        
        if 'yes' in clean_resp:
            return True
        elif 'no' in clean_resp:
            return False
        else:
            print(f"Unexpected response: {response}")
            return None

    except Exception as e:
        print(f"Error during GPT call: {e}")
        return None

def main(args, mode):
    in_jsonl = f"{args.data_dir}/qa_{mode}.jsonl"
    out_jsonl = f"{args.data_dir}/acc_{mode}2.jsonl"

    with open(in_jsonl, 'r', encoding='utf-8') as f:
        jsonl_data = [json.loads(line) for line in f]

    processed_ids = set()
    if not args.no_resume and os.path.exists(out_jsonl):
        with open(out_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except Exception:
                    pass

    for item in tqdm(jsonl_data, desc=f"Processing {mode}"):
        tid = item['id']

        if tid in processed_ids:
            continue

        gens_jsonl = item.get('generated_answer', [])
        results_list = []
        if mode == 'train':
            num_gen = 1
        else:
            num_gen = args.use_num_generations
        for m in range(num_gen):
            if m >= len(gens_jsonl):
                break

            pred_text = gens_jsonl[m].get('pred_ans', '')

            if not pred_text:
                results_list.append({"acc": -1.0, "note": "empty_pred"})
                continue

            res1 = gpt_metric(pred_text, item, args.eval_model, ask_ref=True)
            if res1 is None:
                results_list.append({"acc": -1.0, "status": "api_error"})
                continue
            
            if res1 is False:
                res2 = gpt_metric(pred_text, item, args.eval_model, ask_ref=False)
                if res2 is None:
                    results_list.append({"acc": -1.0, "status": "api_error"})
                    continue

            if res1:
                final_acc = 1.0 
                status = "consistent"
            elif res2 is True:
                final_acc = -1.0
                status = "inconsistent"
            else: 
                final_acc = 0.0
                status = "incorrect"

            results_list.append({
                "acc": final_acc,
                "status": status
            })

        if results_list:
            output = {
                "id": tid,
                "accuracy": [r['acc'] for r in results_list],
                "status": [r.get('status') for r in results_list]
            }
            with open(out_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='llama3_70b')
    parser.add_argument('--use_num_generations', type=int, default=5)
    parser.add_argument('--eval_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--double_check', action='store_true', default=True)
    parser.add_argument('--no_resume', action='store_true', default=False)
    parser.add_argument('--split', type=str, required=True)

    args = parser.parse_args()
    main(args, args.split)
