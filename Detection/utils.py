import json
import numpy as np
import random
import torch
import os
import pickle
from sklearn import metrics

def load_jsonl_data(path, key_name=None):
    """JSONL loader for auxiliary data (e.g., accuracy or semantic_ids)."""
    data = {}
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if key_name:
                data[entry['id']] = entry[key_name]
            else:
                data[entry['id']] = entry
    return data

def load_pickle_data(path):
    """Pickle loader for main data including hidden states (h_out)."""
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)

def idx2prob(semantic_ids_dict, used_sample=6):
    se_prob_dict = {}
    for tid, count in semantic_ids_dict.items():
        if len(count) > used_sample:
            count = count[:used_sample]
        values, nums = np.unique(count, return_counts=True)
        probabilities = nums / len(count)
        pro_dict = dict(zip(values, probabilities))
        prob_list = [pro_dict[i] for i in count]
        se_prob_dict[tid] = prob_list
    return se_prob_dict

def load_embedding_train(args, idx_layer):
    mode = 'train'
    datapath = f"{args.data_dir}/{args.data_name}/{mode}_layer{idx_layer}.pkl"
    acc_datapath = f"{args.data_dir}/{args.data_name}/acc_{mode}2.jsonl"
    loaded_data = load_pickle_data(datapath)
    acc_loaded_data = load_jsonl_data(acc_datapath, key_name='accuracy')

    all_hidden_list = []  
    all_label_list = []   
        
    for tid, example in loaded_data.items():
        # Use the generation-side key 'generated_answer'
        generations = example.get('generated_answer', [])
        
        if tid not in acc_loaded_data:
            continue
        
        acc_list = acc_loaded_data[tid]
        if not acc_list:
            continue
        
        m = 0
        acc = acc_list[m]
        if acc == 2.0 or acc == -1.0: continue
        
        h_out_raw = generations[m].get('h_out')
        if h_out_raw is None: continue
            
        hiddens = h_out_raw if isinstance(h_out_raw, torch.Tensor) else torch.tensor(h_out_raw)
        
        hiddens = hiddens[:, args.start_layer::args.interval_layer, :] \
            if args.end_layer is None \
            else hiddens[:, args.start_layer:args.end_layer:args.interval_layer, :]
        if args.mode == 'reg':
            hiddens = hiddens / (hiddens.norm(dim=-1, keepdim=True) + 1e-8)
        label = 1.0 - acc
        all_hidden_list.append([hiddens])
        all_label_list.append([label])

    return all_hidden_list, all_label_list, list(range(len(all_hidden_list)))


def load_embedding_val(args, idx_layer):
    mode='validation'
    datapath = f"{args.data_dir}/{args.data_name}/{mode}_layer{idx_layer}.pkl"
    acc_test_datapath = f"{args.data_dir}/{args.data_name}/acc_{mode}2.jsonl"
    loaded_data_dict = load_pickle_data(datapath)
    acc_test_loaded_data = load_jsonl_data(acc_test_datapath, key_name='accuracy')
    tids = list(loaded_data_dict.keys())
    random.seed(args.seed)
    val_sampled_tids = set(random.sample(tids, min(len(tids), args.val_sample)))
    val_hidden_list, val_label_list = [], []
    test_hidden_list, test_label_list, test_qa_list = [], [], []

    for tid in tids:
        example = loaded_data_dict[tid]
        generations = example.get('generated_answer', []) # Use the corrected key name

        if tid not in acc_test_loaded_data:
            continue
            
        acc_list = acc_test_loaded_data[tid]
        if not acc_list:
            continue

        num = min(args.used_sample, len(generations))
        tmp_hidden, tmp_label, tmp_ans = [], [], []

        for m in range(num):
            if m >= len(acc_list): break
            acc = acc_list[m]
            if acc == 2.0 or acc == -1.0 : continue

            h_out_raw = generations[m].get('h_out')
            if h_out_raw is None: continue
            
            hiddens = h_out_raw if isinstance(h_out_raw, torch.Tensor) else torch.tensor(h_out_raw)
            
            hiddens = hiddens[:, args.start_layer::args.interval_layer, :] \
                if args.end_layer is None \
                else hiddens[:, args.start_layer:args.end_layer:args.interval_layer, :]
            if args.mode == 'reg':
                hiddens = hiddens / (hiddens.norm(dim=-1, keepdim=True) + 1e-8)
            ans_tokens = generations[m].get('tokens')
            label = 1.0 - acc
            
            tmp_hidden.append(hiddens)
            tmp_label.append(label)
            tmp_ans.append(ans_tokens)

        if not tmp_hidden: continue

        if tid in val_sampled_tids:
            val_hidden_list.append(tmp_hidden)
            val_label_list.append(tmp_label)
        else:
            test_hidden_list.append(tmp_hidden)
            test_label_list.append(tmp_label)
            test_qa_list.append(tmp_ans)

    return (val_hidden_list, val_label_list, 
            test_qa_list, test_hidden_list, test_label_list)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, idx_list, label_list):
        self.idx_list = idx_list
        self.label_list = label_list

    def __getitem__(self, i):
        return {
            "idx": self.idx_list[i],
            "label": self.label_list[i][0]
        }

    def __len__(self):
        return len(self.idx_list)

def my_collate(batch):
    idx = [b["idx"] for b in batch]
    return {"idx": idx}

def tabular_metrics(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Set threshold based on the proportion of normal samples (label 0)
    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thresh = np.percentile(y_score, ratio)
    y_pred = (y_score >= thresh).astype(int)
    y_true = y_true.astype(int)
    p, r, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return metrics.roc_auc_score(y_true, y_score), metrics.average_precision_score(y_true, y_score), f1
