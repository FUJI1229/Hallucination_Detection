import os
import json
import hashlib
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type
from openai import AzureOpenAI
from dotenv import load_dotenv

# ================================================
# Azure OpenAI setup
project_root = Path(__file__).resolve().parents[3]
load_dotenv()
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

client = None
if AZURE_API_KEY and AZURE_ENDPOINT:
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

# ================================================
# Utility
# ================================================
def md5hash(string):
    return int(hashlib.md5(string.encode("utf-8")).hexdigest(), 16)

@retry(retry=retry_if_not_exception_type(Exception), wait=wait_random_exponential(min=1, max=10))
def azure_predict(prompt, temperature=0.0):
    if client is None or not client.api_key:
        raise RuntimeError("Azure OpenAI client not configured. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in the environment.")
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT, 
            messages=messages, 
            temperature=temperature, 
            max_tokens=64
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return None

# ================================================
# Entailment model
# ================================================
class EntailmentAzureGPT:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.prediction_cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                try:
                    return pickle.load(f)
                except:
                    return {}
        return {}

    def save_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.prediction_cache, f)

    def check_implication(self, text1, text2, question):
        # Use a strict prompt format
        prompt = (f"Context: Question asked to an AI: '{question}'\n"
                  f"Answer A: {text1}\n"
                  f"Answer B: {text2}\n"
                  f"Determine if Answer A entails Answer B. "
                  f"Respond with only one word: entailment, contradiction, or neutral.")
        
        hashed = md5hash(prompt)
        if hashed in self.prediction_cache:
            response = self.prediction_cache[hashed]
        else:
            response = azure_predict(prompt)
            if response:
                self.prediction_cache[hashed] = response
        
        if not response: 
            return 1 # neutral
        
        r = response.lower()
        if "contradiction" in r: return 0
        if "entailment" in r: return 2
        return 1 # neutral

# ================================================
# Entropy utilities
# ================================================
def get_semantic_ids(strings_list, model, question):
    semantic_ids = [-1] * len(strings_list)
    cluster_id = 0
    for i, s1 in enumerate(strings_list):
        if semantic_ids[i] == -1:
            semantic_ids[i] = cluster_id
            for j in range(i + 1, len(strings_list)):
                # Bidirectional entailment check for semantic equivalence
                imp1 = model.check_implication(s1, strings_list[j], question)
                imp2 = model.check_implication(strings_list[j], s1, question)
                if imp1 == 2 and imp2 == 2:
                    semantic_ids[j] = cluster_id
            cluster_id += 1
    return semantic_ids

def cluster_assignment_entropy(semantic_ids):
    if not semantic_ids: return 0.0
    counts = np.bincount(semantic_ids)
    probs = counts / len(semantic_ids)
    return -np.sum(probs * np.log(probs + 1e-12))

def predictive_entropy_rao(semantic_ids, log_likelihoods):
    """
    semantic_ids: Cluster ID for each generated answer
    log_likelihoods: Mean log-likelihood for each generated answer
    """
    # Convert to probability space (exp(log_lik))
    exp_ll = np.exp(log_likelihoods)
    
    # Aggregate probabilities per cluster
    unique_ids = sorted(set(semantic_ids))
    cluster_probs = []
    for uid in unique_ids:
        indices = [i for i, x in enumerate(semantic_ids) if x == uid]
        # Sum answer probabilities within cluster and normalize
        c_prob = np.sum(exp_ll[indices]) / (np.sum(exp_ll) + 1e-12)
        cluster_probs.append(c_prob)
    
    cluster_probs = np.array(cluster_probs)
    return -np.sum(cluster_probs * np.log(cluster_probs + 1e-12))

# ================================================
# Main Execution
# ================================================
def compute_entailment(args, dataset_split):
    entail_model = EntailmentAzureGPT(cache_file=f"entailment_cache_{args.data_name}.pkl")
    
    jsonl_file = f"{args.data_dir}/qa_{dataset_split}.jsonl"
    out_file = f"{args.data_dir}/se_{dataset_split}.jsonl"

    if  not os.path.exists(jsonl_file):
        print(f"❌ Missing files for {dataset_split} at {args.data_dir}")
        return

    # 1) Load answer texts and log-likelihoods (both available in qa_*.jsonl)
    # Since the first script stores qa_responses as {'pred_ans', 'log_lik'},
    # using jsonl as the primary source is safer.
    text_data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            text_data.append(json.loads(line))

    # Check already processed IDs (for resume)
    processed_ids = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                processed_ids.add(json.loads(line)["id"])

    for item in tqdm(text_data, desc=f"Processing {dataset_split}"):
        tid = item['id']
        if tid in processed_ids: continue

        question = item["question"]
        generated_data = item["generated_answer"]
        
        # Build answer-text and log-likelihood lists
        texts = [g["pred_ans"] for g in generated_data]
        
        # Handle log_lik for tensor/list/scalar cases
        log_liks = []
        for g in generated_data:
            ll = g["log_lik"]
            if isinstance(ll, list):
                log_liks.append(np.mean(ll))
            elif torch.is_tensor(ll):
                log_liks.append(ll.cpu().numpy().mean())
            else:
                log_liks.append(ll)
        
        log_liks = np.array(log_liks)

        # 2) Semantic clustering
        semantic_ids = get_semantic_ids(texts, entail_model, question)
        
        # 3) Entropy computation
        ce = cluster_assignment_entropy(semantic_ids)
        pe = predictive_entropy_rao(semantic_ids, log_liks)

        # 4) Save
        result = {
            "id": tid,
            "semantic_ids": [int(x) for x in semantic_ids],
            "cluster_assignment_entropy": float(ce),
            "semantic_entropy": float(pe)
        }

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Periodically save cache
        if len(processed_ids) % 20 == 0:
            entail_model.save_cache()
        processed_ids.add(tid)

    entail_model.save_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="squad")
    parser.add_argument("--model_name", type=str, default="llama3_70b")
    args = parser.parse_args()

    compute_entailment(args, 'train')
    compute_entailment(args, 'validation')
